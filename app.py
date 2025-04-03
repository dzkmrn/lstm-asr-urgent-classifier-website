import os
import logging
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_METAL_DISABLE'] = '1'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.getLogger('pymongo').setLevel(logging.WARNING)
logging.getLogger('h5py').setLevel(logging.WARNING)

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import tensorflow as tf
import numpy as np
import soundfile as sf
import librosa
from datetime import datetime
from database import MongoDB

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
db = MongoDB()

logger.info("Loading LSTM model...")
try:
    model = tf.keras.models.load_model('models/lstm_model_fold_2.h5', compile=False)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

@app.route('/process_audio', methods=['POST'])
def process_audio():
    logger.info("\n=== Starting new audio processing request ===")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request headers: {dict(request.headers)}")
    logger.info(f"Request files keys: {list(request.files.keys())}")
    logger.info(f"Request form keys: {list(request.form.keys())}")
    
    if 'audio' not in request.files:
        logger.error("No audio file received in request.files")
        logger.error(f"Available files: {request.files}")
        return jsonify({'error': 'No audio file'}), 400
    
    try:
        audio_file = request.files['audio']
        if not audio_file.filename:
            logger.error("Empty filename received")
            return jsonify({'error': 'Empty filename'}), 400
            
        logger.info(f"Audio file received: {audio_file.filename}")
        
        user_id = request.form.get('user_id', 'default_user')
        logger.info(f"Processing audio for user: {user_id}")
        
        temp_path = os.path.join('data', f'temp_{user_id}.wav')
        audio_file.save(temp_path)
        logger.info(f"Audio saved to: {temp_path}")
        
        # Verify file exists and size
        file_size = os.path.getsize(temp_path)
        logger.info(f"Saved file size: {file_size} bytes")
        
        audio_data, sr = librosa.load(temp_path, sr=16000)
        logger.info(f"Audio loaded: duration={len(audio_data)/sr:.2f}s, sr={sr}Hz, shape={audio_data.shape}")
        
        features = extract_features(audio_data, sr)
        logger.info(f"Features extracted: shape={features.shape}")
        
        logger.info("Making prediction...")
        prediction = model.predict(features, verbose=0)
        is_urgent = bool(prediction[0][0] > 0.5)
        confidence = float(prediction[0][0])
        logger.info(f"Prediction complete: Urgent={is_urgent}, Confidence={confidence:.2%}")
        
        record = {
            'user_id': user_id,
            'timestamp': datetime.now(),
            'audio_path': temp_path,
            'is_urgent': is_urgent,
            'confidence': confidence
        }
        
        db.save_record(record)
        logger.info(f"Record saved to database: {record}")
        
        socketio.emit('new_detection', record)
        logger.info("WebSocket notification sent")
        
        logger.info("=== Audio processing completed successfully ===\n")
        return jsonify({
            'status': 'success',
            'is_urgent': is_urgent,
            'confidence': confidence
        })
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def extract_features(audio_data, sr=16000):
    try:
        logger.info("Extracting MFCC features...")
        mfcc = librosa.feature.mfcc(
            y=audio_data, 
            sr=sr,
            n_mfcc=13,
            n_fft=2048,
            hop_length=512
        )
        mfcc = mfcc.T
        features = np.expand_dims(mfcc, axis=0)
        logger.info(f"Features extracted successfully: shape={features.shape}")
        return features
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        raise

# Add this new route after your existing routes
@app.route('/test_model', methods=['GET'])
def test_model():
    logger.info("=== Testing model prediction ===")
    try:
        # Load a test audio file
        test_path = 'data/temp_default_user.wav'
        if not os.path.exists(test_path):
            logger.error(f"Test file not found: {test_path}")
            return jsonify({'error': 'Test file not found'}), 404
            
        audio_data, sr = librosa.load(test_path, sr=16000)
        logger.info(f"Test audio loaded: duration={len(audio_data)/sr:.2f}s, sr={sr}Hz")
        
        features = extract_features(audio_data, sr)
        logger.info(f"Test features extracted: shape={features.shape}")
        
        prediction = model.predict(features, verbose=0)
        is_urgent = bool(prediction[0][0] > 0.5)
        confidence = float(prediction[0][0])
        logger.info(f"Test prediction: Urgent={is_urgent}, Confidence={confidence:.2%}")
        
        return jsonify({
            'status': 'success',
            'is_urgent': is_urgent,
            'confidence': confidence
        })
        
    except Exception as e:
        logger.error(f"Error in test route: {e}")
        return jsonify({'error': str(e)}), 500

# Add this new test endpoint
@app.route('/test', methods=['GET'])
def test():
    logger.info("Test endpoint called")
    return jsonify({'status': 'Server is running'})

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    socketio.run(app, debug=True)