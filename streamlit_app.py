import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import requests
import socketio
from datetime import datetime
import os

class AudioRecorder:
    def __init__(self):
        self.sio = socketio.Client()
        try:
            self.sio.connect('http://localhost:5000')
            self.setup_socketio()
        except Exception as e:
            st.error(f"Could not connect to server: {e}")

    def setup_socketio(self):
        @self.sio.on('connect')
        def on_connect():
            st.success('Connected to server')

def main():
    st.set_page_config(page_title="Urgent Speech Detection")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # User authentication
    if 'user_type' not in st.session_state:
        user_type = st.radio("Select User Type", ["User", "Admin"])
        if st.button("Login"):
            st.session_state.user_type = user_type
    
    if 'user_type' in st.session_state:
        if st.session_state.user_type == "User":
            user_interface()
        else:
            admin_interface()

def user_interface():
    st.title("Speech Recording")
    
    if st.button("Record"):
        try:
            with st.spinner("Recording..."):
                # Record audio
                duration = 5  # seconds
                fs = 16000  # Match the sampling rate used in model training
                st.info("Starting recording...")
                recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
                sd.wait()
                st.info("Recording completed")
                
                # Save recording
                filename = os.path.join('data', f'recording_{datetime.now().strftime("%Y%m%d_%H%M%S")}.wav')
                sf.write(filename, recording, fs)
                st.info(f"Audio saved to {filename}")
                st.audio(filename)
                
                # Send to server
                st.info("Sending to server for processing...")
                with open(filename, 'rb') as audio_file:
                    # Simplify the file upload
                    files = {'audio': audio_file}
                    data = {'user_id': st.session_state.get('user_id', 'default_user')}
                    try:
                        response = requests.post(
                            'http://127.0.0.1:5000/process_audio',
                            files=files,
                            data=data,
                            timeout=30
                        )
                        st.info(f"Server response status: {response.status_code}")
                        st.info(f"Response content: {response.content.decode() if response.content else 'No content'}")
                        
                        if response.status_code == 200:
                            result = response.json()
                            confidence = result.get('confidence', 0) * 100
                            if result['is_urgent']:
                                st.error(f"⚠️ URGENT SITUATION DETECTED! (Confidence: {confidence:.1f}%)")
                            else:
                                st.success(f"✅ No emergency detected (Confidence: {confidence:.1f}%)")
                        else:
                            st.error(f"Server Error: {response.text}")
                            st.error(f"Response Headers: {dict(response.headers)}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Connection Error: {str(e)}")
                    
        except Exception as e:
            st.error(f"Error during recording: {str(e)}")
            st.error(f"Error type: {type(e).__name__}")
            import traceback
            st.error(f"Full error: {traceback.format_exc()}")
    
    # Show history
    st.subheader("History")
    try:
        response = requests.get(f'http://localhost:5000/user_history/{st.session_state.get("user_id", "default_user")}')
        if response.status_code == 200:
            history = response.json()
            for record in history:
                st.write(f"Time: {record['timestamp']}")
                st.write(f"Status: {'Urgent' if record['is_urgent'] else 'Normal'}")
                st.write("---")
    except Exception as e:
        st.error(f"Could not load history: {e}")

def admin_interface():
    st.title("Admin Dashboard")
    
    # Real-time notifications
    st.subheader("Live Notifications")
    if 'recorder' not in st.session_state:
        st.session_state.recorder = AudioRecorder()
    
    # History of urgent cases
    st.subheader("Urgent Cases History")
    try:
        response = requests.get('http://localhost:5000/urgent_cases')
        if response.status_code == 200:
            cases = response.json()
            for case in cases:
                st.write(f"User: {case['user_id']}")
                st.write(f"Time: {case['timestamp']}")
                st.write("---")
    except Exception as e:
        st.error(f"Could not load urgent cases: {e}")

if __name__ == "__main__":
    main()