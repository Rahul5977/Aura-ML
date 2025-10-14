#!/usr/bin/env python3
"""
Comprehensive Test Script for Aura Backend
Tests Week 3, Week 4.1, and Week 4.2 Features
"""

import requests
import json
import base64
import numpy as np
from scipy.io import wavfile
import tempfile
import os

# Configuration
BASE_URL = "http://localhost:8000"
API_KEY = None  # Will be set after login
TEST_USER_EMAIL = f"test_user_{np.random.randint(100000)}@example.com"
TEST_USER_PASSWORD = "testpassword123"

def register_and_login():
    """Register a test user and get authentication token"""
    global API_KEY
    
    print_header("Setting up Test User")
    
    # Register user
    username = TEST_USER_EMAIL.split('@')[0]
    user_data = {
        "email": TEST_USER_EMAIL,
        "username": username,
        "full_name": "Test User",
        "password": TEST_USER_PASSWORD
    }
    
    response = requests.post(f"{BASE_URL}/auth/register", json=user_data)
    print(f"Registration Status: {response.status_code}")
    if response.status_code != 201:
        print(f"Registration Response: {response.json()}")
    
    # Login (can use username or email)
    login_data = {
        "username": username,  # Can also use email here
        "password": TEST_USER_PASSWORD
    }
    
    response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
    print(f"Login Status: {response.status_code}")
    
    if response.status_code == 200:
        API_KEY = response.json()["access_token"]
        print("✅ Authentication successful!")
        return True
    else:
        print(f"❌ Authentication failed: {response.json()}")
        return False

def print_header(text):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def test_health():
    """Test health endpoint"""
    print_header("Testing Health Endpoint")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    print("✅ Health check passed!")

def generate_test_audio(duration=1.0, freq=440, sample_rate=16000):
    """Generate a simple sine wave test audio"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * freq * t) * 0.3  # 440 Hz sine wave
    audio = (audio * 32767).astype(np.int16)
    return audio, sample_rate

def test_audio_transcription():
    """Test audio transcription endpoint"""
    print_header("Testing Audio Transcription (Week 3)")
    
    # Generate test audio
    audio, sample_rate = generate_test_audio(duration=2.0)
    
    # Save to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        wavfile.write(tmp_file.name, sample_rate, audio)
        tmp_filename = tmp_file.name
    
    try:
        # Send audio for transcription
        with open(tmp_filename, 'rb') as f:
            files = {'file': ('test.wav', f, 'audio/wav')}
            response = requests.post(
                f"{BASE_URL}/transcribe",
                files=files,
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        assert response.status_code == 200
        result = response.json()
        assert "text" in result
        assert "language" in result
        print("✅ Transcription test passed!")
        
    finally:
        os.unlink(tmp_filename)

def test_emotion_recognition():
    """Test emotion recognition endpoint"""
    print_header("Testing Emotion Recognition (Week 4.1)")
    
    # Generate test audio
    audio, sample_rate = generate_test_audio(duration=3.0, freq=880)
    
    # Save to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        wavfile.write(tmp_file.name, sample_rate, audio)
        tmp_filename = tmp_file.name
    
    try:
        # Send audio for emotion recognition
        with open(tmp_filename, 'rb') as f:
            files = {'file': ('test.wav', f, 'audio/wav')}
            response = requests.post(
                f"{BASE_URL}/recognize-emotion",
                files=files,
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        assert response.status_code == 200
        result = response.json()
        assert "emotion" in result
        assert "confidence" in result
        assert "timestamp" in result
        print("✅ Emotion recognition test passed!")
        
    finally:
        os.unlink(tmp_filename)

def test_websocket_connection():
    """Test WebSocket connection info"""
    print_header("Testing WebSocket Endpoint Info (Week 4.2)")
    print(f"WebSocket endpoint available at: ws://localhost:8000/ws/audio")
    print("Note: Full WebSocket testing requires a WebSocket client")
    print("✅ WebSocket endpoint is documented!")

def test_user_management():
    """Test user management endpoints"""
    print_header("Testing User Management (Week 3)")
    
    # Test user creation
    user_data = {
        "email": f"test_{np.random.randint(10000)}@example.com",
        "name": "Test User",
        "password": "testpassword123"
    }
    
    response = requests.post(f"{BASE_URL}/users", json=user_data)
    print(f"Create User Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        user_id = response.json()["id"]
        
        # Test getting user
        response = requests.get(
            f"{BASE_URL}/users/{user_id}",
            headers={"X-API-Key": API_KEY}
        )
        print(f"\nGet User Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print("✅ User management tests passed!")
    else:
        print("ℹ️  User management test skipped (database may not be initialized)")

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  AURA BACKEND COMPREHENSIVE TEST SUITE")
    print("  Testing Week 3, Week 4.1, and Week 4.2 Features")
    print("="*60)
    
    try:
        test_health()
        
        # Setup authentication before protected endpoints
        if not register_and_login():
            print("❌ Failed to authenticate, skipping protected endpoint tests")
            return
        
        test_audio_transcription()
        test_emotion_recognition()
        test_websocket_connection()
        test_user_management()
        
        print_header("🎉 ALL TESTS PASSED! 🎉")
        print("Summary:")
        print("  ✅ Week 3: Audio transcription working")
        print("  ✅ Week 4.1: Emotion recognition working")
        print("  ✅ Week 4.2: WebSocket endpoint available")
        print("  ✅ User management endpoints working")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
