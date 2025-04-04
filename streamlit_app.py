import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import requests
import socketio
from datetime import datetime
import os
import hashlib  
import pandas as pd
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stylable_container import stylable_container

# Konfigurasi Halaman
st.set_page_config(
    page_title="Emergency Voice Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)    

# Custom CSS
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f0f2f6;
        padding: 2rem;
        
    }
    
    /* Tombol Rekam Khusus */
    .record-button {
        width: 100% !important;
        height: 80px !important;
        font-size: 1.5rem !important;
        border: 3px solid #ff4b4b !important;
    }
    
    /* Status Darurat */
    .emergency-status {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        font-size: 2.5rem !important;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# Database Pengguna
USERS_DB = {
    "admin": {
        "password": hashlib.sha256("Admin123!".encode()).hexdigest(),
        "role": "admin",
        "history": []
    },
    "user1": {
        "password": hashlib.sha256("User123!".encode()).hexdigest(),
        "role": "user",
        "history": []
    }
}

class AudioRecorder:
    def __init__(self):
        self.sio = socketio.Client()
        try:
            self.sio.connect('http://localhost:5000')
        except Exception as e:
            st.error(f"Koneksi server gagal: {e}")

# Fungsi Autentikasi
def handle_login(username, password):
    if not username or not password:
        return st.error("Username dan password harus diisi!")
    
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    
    if username in USERS_DB and USERS_DB[username]["password"] == hashed_pw:
        st.session_state.update({
            "authenticated": True,
            "user_type": USERS_DB[username]["role"],
            "current_user": username
        })
        st.rerun()
    else:
        st.error("Kredensial tidak valid!")

def handle_signup(new_user, new_pw):
    if len(new_pw) < 8:
        return st.error("Password minimal 8 karakter!")
    
    if new_user in USERS_DB:
        return st.error("Username sudah ada!")
    
    USERS_DB[new_user] = {
        "password": hashlib.sha256(new_pw.encode()).hexdigest(),
        "role": "user",
        "history": []
    }
    st.success("Akun berhasil dibuat!")
    st.balloons()

# Tampilan Autentikasi
def show_auth():
    with stylable_container(
        key="auth_box",
        css_styles="""
            {
                background: white;
                padding: 3rem;
                border-radius: 20px;
                box-shadow: 0 8px 16px rgba(0,0,0,0.1);
                max-width: 600px;
                margin: 3rem auto;
            }
        """
    ):
        st.title("🔒 Sistem Deteksi Darurat Suara")
        tab1, tab2 = st.tabs(["Masuk", "Daftar"])
        
        with tab1:
            with st.form("login"):
                user = st.text_input("Username")
                pw = st.text_input("Password", type="password")
                if st.form_submit_button("Masuk", use_container_width=True):
                    handle_login(user, pw)
        
        with tab2:
            with st.form("signup"):
                new_user = st.text_input("Username Baru")
                new_pw = st.text_input("Password Baru", type="password")
                if st.form_submit_button("Buat Akun", use_container_width=True):
                    handle_signup(new_user, new_pw)

# Tampilan Utama Pengguna
def user_interface():
    colored_header(
        label="🎙️ REKAM DARURAT",
        description="Tekan tombol untuk mulai merekam (5 detik)",
        color_name="red-70"
    )
    
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        # Panel Rekam dan Hasil
        with stylable_container(
            key="rec_box",
            css_styles="""
                {
                    background: white;
                    padding: 2rem;
                    border-radius: 20px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                }
            """
        ):
            # Tombol Rekam
            if st.button("⏺️ REKAM SEKARANG", 
                        use_container_width=True, 
                        key="rec_button",
                        type="primary"):
                handle_recording()
            
            # Hasil Analisis
            if 'result' in st.session_state:
                res = st.session_state.result
                status_style = "background: #ffebee; color: #d32f2f; border: 2px solid #d32f2f;" if res['is_urgent'] else "background: #e8f5e9; color: #2e7d32; border: 2px solid #2e7d32;"
                
                st.markdown(f"""
                <div class='emergency-status' style='{status_style}'>
                    { '🚨 DARURAT' if res['is_urgent'] else '✅ AMAN' }
                    <div style='font-size: 1.5rem; margin-top: 1rem;'>
                        Tingkat Keyakinan: {res['confidence']*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Detail Analisis
                with st.expander("🔬 DETAIL ANALISIS LENGKAP", expanded=True):
                    cols = st.columns(3)
                    cols[0].metric("Kata Kunci", "Tolong, Bantu")
                    cols[1].metric("Tingkat Stress", "78%")
                    cols[2].metric("Intensitas Suara", "85 dB")
                    st.map(pd.DataFrame({"lat": [-6.2000], "lon": [106.8167]}))
    
    with col2:
        # Riwayat Rekaman
        colored_header(
            label="📜 RIWAYAT",
            description="5 rekaman terakhir",
            color_name="blue-70"
        )
        
        with stylable_container(
            key="history_box",
            css_styles="""
                {
                    background: white;
                    padding: 1rem;
                    border-radius: 15px;
                    max-height: 600px;
                    overflow-y: auto;
                }
            """
        ):
            for item in USERS_DB[st.session_state.current_user]["history"][-5:]:
                with st.container():
                    st.caption(f"🕒 {item['time']}")
                    st.markdown(f"**{item['status']}** ({item['confidence']}%)")
                    st.progress(item['confidence']/100)
                    st.audio(item['file'], format="audio/wav")
                    st.divider()

def handle_recording():
    try:
        with st.spinner("🎙️ Sedang merekam..."):
            fs = 16000
            duration = 5
            audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            
            # Simpan rekaman
            filename = f"data/{st.session_state.current_user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            sf.write(filename, audio, fs)
            
            # Simpan hasil (simulasi)
            result = {
                'is_urgent': np.random.rand() > 0.5,
                'confidence': round(np.random.uniform(0.7, 0.95), 2),
                'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'file': filename
            }
            
            # Update state dan history
            st.session_state.result = result
            USERS_DB[st.session_state.current_user]["history"].append({
                'time': result['time'],
                'status': 'DARURAT' if result['is_urgent'] else 'AMAN',
                'confidence': int(result['confidence']*100),
                'file': filename
            })
            
    except Exception as e:
        st.error(f"Gagal merekam: {str(e)}")

# Tampilan Admin
def admin_interface():
    st.title("📊 DASHBOARD ADMIN")
    
    # Statistik Real-time
    cols = st.columns(3)
    cols[0].metric("Total Pengguna", len(USERS_DB))
    cols[1].metric("Aktivitas Hari Ini", sum(1 for u in USERS_DB if USERS_DB[u]['history']))
    cols[2].metric("Respon Rata-rata", "1.2s")
    
    # Visualisasi Data
    tab1, tab2 = st.tabs(["Statistik", "Monitor"])
    with tab1:
        hist_data = pd.DataFrame({
            "Waktu": [h['time'] for h in USERS_DB['user1']['history']],
            "Status": [h['status'] for h in USERS_DB['user1']['history']]
        })
        st.bar_chart(hist_data['Status'].value_counts())
    
    with tab2:
        for user in USERS_DB:
            if user != 'admin':
                with st.expander(f"Pengguna: {user}"):
                    for h in USERS_DB[user]['history'][-3:]:
                        st.write(f"{h['time']} - {h['status']} ({h['confidence']}%)")

# Main App
def main():
    if 'authenticated' not in st.session_state:
        show_auth()
    else:
        if st.session_state.user_type == "admin":
            admin_interface()
        else:
            user_interface()

if __name__ == "__main__":
    main()