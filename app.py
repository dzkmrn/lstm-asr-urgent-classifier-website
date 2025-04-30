import os
import logging
from bson import ObjectId
import speech_recognition as sr
import librosa.effects

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
    model = tf.keras.models.load_model('models/lstm_model_fold_3_29apr.h5', compile=False)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise


@app.route('/process_audio', methods=['POST'])
def process_audio():
    logger.info("\n=== Starting new audio processing request ===")
    
    if 'audio' not in request.files:
        logger.error("No audio file received")
        return jsonify({'error': 'No audio file'}), 400
    
    try:
        audio_file = request.files['audio']
        user_id = request.form.get('user_id', 'default_user')
        
        # 1. Baca audio langsung dari memory
        try:
            audio_data, sr = sf.read(audio_file)
        except Exception as e:
            logger.error(f"Error reading audio: {str(e)}")
            return jsonify({'error': 'Invalid audio file'}), 400

        # 2. Resample jika diperlukan
        if sr != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            sr = 16000

        # 3. Praproses audio
        # Trim silence
        audio_trimmed, _ = librosa.effects.trim(audio_data, top_db=20)
        
        # Pad/truncate ke 3 detik (48000 samples)
        target_length = 3 * sr
        if len(audio_trimmed) < target_length:
            pad_width = target_length - len(audio_trimmed)
            audio_processed = np.pad(audio_trimmed, (0, pad_width), mode='constant')
        elif len(audio_trimmed) > target_length:
            audio_processed = audio_trimmed[:target_length]
        else:
            audio_processed = audio_trimmed

        # 4. Simpan versi yang sudah diproses
        processed_filename = f'processed_{user_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.wav'
        processed_path = os.path.join('data', processed_filename)
        sf.write(processed_path, audio_processed, sr, subtype='PCM_16')

        # 5. Ekstrak fitur dari audio yang sudah diproses
        features = extract_features(audio_processed, sr)
        
        # 6. Lakukan prediksi
        prediction = model.predict(features, verbose=0)
        is_urgent = bool(prediction[0][0] > 0.5)
        confidence = float(prediction[0][0])

        # 7. Simpan ke database
        record = {
            'user_id': user_id,
            'timestamp': datetime.now(),
            'audio_path': processed_path,  # Simpan path ke file processed
            'is_urgent': is_urgent,
            'confidence': confidence
        }
        
        db.save_record(record)
        socketio.emit('new_detection', record)

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
            
        # 1. Load audio, force mono
        audio_data, sr = librosa.load(test_path, sr=16000, mono=True)
        logger.info(f"Audio loaded: duration={len(audio_data)/sr:.2f}s, sr={sr}Hz, shape={audio_data.shape}")

        # 2. Remove silence
        audio_data, _ = librosa.effects.trim(audio_data, top_db=30)
        logger.info(f"After silence trimming: duration={len(audio_data)/sr:.2f}s, samples={audio_data.shape}")

        # 3. Pad or truncate to exactly 3 seconds (3*16000 = 48000 samples)
        target_length = 3 * sr  # 48000 samples for 3 seconds

        if len(audio_data) < target_length:
            pad_width = target_length - len(audio_data)
            audio_data = np.pad(audio_data, (0, pad_width), mode='constant')
            logger.info(f"Audio padded: new length={len(audio_data)}, should be {target_length}")
        else:
            audio_data = audio_data[:target_length]
            logger.info(f"Audio truncated: new length={len(audio_data)}, should be {target_length}")
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
    
