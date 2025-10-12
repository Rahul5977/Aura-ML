#!/usr/bin/env python3
"""
Audio streaming client for testing real-time transcription
Simulates live microphone by streaming audio file in chunks
"""

import asyncio
import websockets
import json
import sys
import wave
import requests
import argparse
from pathlib import Path
import time

BASE_URL = "http://localhost:8000"
WS_BASE_URL = "ws://localhost:8000"


def read_wav_file(filepath: str):
    """
    Read WAV file and return audio data with metadata.
    
    Args:
        filepath: Path to WAV file
        
    Returns:
        Tuple of (audio_data, sample_rate, channels, duration)
    """
    try:
        with wave.open(filepath, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            n_frames = wav_file.getnframes()
            audio_data = wav_file.readframes(n_frames)
            duration = n_frames / sample_rate
            
            return audio_data, sample_rate, channels, duration
    except Exception as e:
        print(f"❌ Error reading WAV file: {e}")
        return None, None, None, None


async def stream_audio_file(token: str, audio_file: str, chunk_size: int = 4096):
    """
    Stream audio file to WebSocket endpoint.
    
    Args:
        token: JWT authentication token
        audio_file: Path to audio file
        chunk_size: Size of audio chunks to send (bytes)
    """
    print(f"\n🎵 Loading audio file: {audio_file}")
    
    # Read audio file
    audio_data, sample_rate, channels, duration = read_wav_file(audio_file)
    
    if audio_data is None:
        print("❌ Failed to load audio file")
        return
    
    print(f"✅ Audio loaded:")
    print(f"   Sample rate: {sample_rate} Hz")
    print(f"   Channels: {channels}")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Size: {len(audio_data)} bytes")
    print(f"   Chunk size: {chunk_size} bytes")
    print()
    
    # Connect to WebSocket
    ws_url = f"{WS_BASE_URL}/ws/v1/audio?token={token}"
    
    print(f"🔌 Connecting to: {ws_url}")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ Connected to audio transcription service\n")
            
            # Create task to receive messages
            async def receive_messages():
                try:
                    while True:
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        msg_type = data.get("type")
                        
                        if msg_type == "transcription":
                            text = data.get("text", "")
                            duration = data.get("duration", 0)
                            print(f"\n📝 TRANSCRIPTION:")
                            print(f"   Text: \"{text}\"")
                            print(f"   Duration: {duration:.2f}s")
                            print()
                        
                        elif msg_type == "status":
                            content = data.get("content", "")
                            print(f"ℹ️  Status: {content}")
                        
                        elif msg_type == "error":
                            content = data.get("content", "")
                            print(f"⚠️  Error: {content}")
                
                except websockets.exceptions.ConnectionClosed:
                    print("\n🔌 Connection closed by server")
                except Exception as e:
                    print(f"\n❌ Error receiving message: {e}")
            
            # Start receiving messages
            receive_task = asyncio.create_task(receive_messages())
            
            # Stream audio in chunks
            print(f"🎤 Streaming audio ({len(audio_data)} bytes in {chunk_size}-byte chunks)...")
            print("=" * 60)
            
            total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
            
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                
                # Send chunk
                await websocket.send(chunk)
                
                chunk_num = (i // chunk_size) + 1
                progress = (i + len(chunk)) / len(audio_data) * 100
                print(f"Sent chunk {chunk_num}/{total_chunks} ({len(chunk)} bytes) - {progress:.1f}%")
                
                # Simulate real-time streaming with small delay
                await asyncio.sleep(0.05)  # 50ms between chunks
            
            print("\n✅ Finished streaming audio")
            print("⏳ Waiting for transcription...")
            
            # Wait a bit for final transcription
            await asyncio.sleep(3)
            
            # Cancel receive task
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass
    
    except Exception as e:
        print(f"\n❌ Connection error: {e}")


def login(username: str, password: str) -> str:
    """
    Login and get JWT token.
    
    Args:
        username: Username
        password: Password
        
    Returns:
        JWT token or None if login fails
    """
    print(f"🔑 Logging in as: {username}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/auth/login",
            json={"username": username, "password": password}
        )
        
        if response.status_code == 200:
            token = response.json()["access_token"]
            print("✅ Login successful\n")
            return token
        else:
            print(f"❌ Login failed: {response.text}\n")
            return None
    
    except Exception as e:
        print(f"❌ Login error: {e}\n")
        return None


async def main():
    parser = argparse.ArgumentParser(description="Audio streaming client for transcription testing")
    parser.add_argument("username", help="Username for authentication")
    parser.add_argument("password", help="Password for authentication")
    parser.add_argument("audio_file", help="Path to WAV audio file")
    parser.add_argument("--chunk-size", type=int, default=4096, help="Audio chunk size in bytes (default: 4096)")
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not Path(args.audio_file).exists():
        print(f"❌ Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    # Check backend health
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend is not healthy")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to backend at {BASE_URL}: {e}")
        print("Please ensure the backend is running on port 8000")
        sys.exit(1)
    
    # Login
    token = login(args.username, args.password)
    if not token:
        sys.exit(1)
    
    # Stream audio
    await stream_audio_file(token, args.audio_file, args.chunk_size)
    
    print("\n" + "=" * 60)
    print("✅ Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
