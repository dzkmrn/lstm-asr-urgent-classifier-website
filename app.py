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
from flask_cors import CORS
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
CORS(app, resources={
    r"/user_history/*": {"origins": "*", "methods": ["GET"]},
    r"/process_audio": {"origins": "*", "methods": ["POST"]}
})
socketio = SocketIO(app, cors_allowed_origins="*")
db = MongoDB()

logger.info("Loading LSTM model...")
try:
    model = tf.keras.models.load_model('models/lstm_model_fold_2.h5', compile=False)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

@app.after_request
def add_headers(response):
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

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

# Tambahkan normalisasi MFCC seperti saat training
def extract_features(audio_data, sr=16000):
    mfcc = librosa.feature.mfcc(
        y=audio_data, 
        sr=sr,
        n_mfcc=13,
        n_fft=2048,
        hop_length=512
    )
    # Tambahkan normalisasi
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    mfcc = mfcc.T
    
    # Pastikan padding sesuai dengan training
    max_length = 94  # Sesuaikan dengan panjang saat training
    if mfcc.shape[0] < max_length:
        pad_width = max_length - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), 
                      mode='constant')
    else:
        mfcc = mfcc[:max_length, :]
    
    return np.expand_dims(mfcc, axis=0)

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
        is_urgent = bool(np.argmax(prediction[0]) == 1)  # Gunakan argmax
        confidence = float(prediction[0][1])  # Ambil probabilitas kelas 1 (darurat)
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

@app.route('/user_history/<user_id>', methods=['GET'])
def get_user_history(user_id):
    try:
        logger.info(f"Attempting to fetch history for user: {user_id}")
        
        # Add database connection check
        try:
            db.client.admin.command('ping')
            logger.info("Database connection active")
        except Exception as e:
            logger.error("Database connection failed")
            raise
        
        history = db.get_user_history(user_id)
        logger.info(f"Raw database response: {history}")
        
        if not history:
            logger.info("No history found for user")
            return jsonify([])
            
        # Convert MongoDB objects
        for record in history:
            record['_id'] = str(record['_id'])
            record['timestamp'] = record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"Processed history: {history}")
        return jsonify(history)
        
    except Exception as e:
        logger.error(f"Error in get_user_history: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/urgent_cases', methods=['GET'])
def get_urgent_cases():
    try:
        urgent_cases = db.get_all_urgent()
        for record in urgent_cases:
            record['_id'] = str(record['_id'])  # Convert ObjectId
            record['timestamp'] = record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        return jsonify(urgent_cases)
    except Exception as e:
        logger.error(f"Error fetching urgent cases: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    socketio.run(app, debug=True)
    
