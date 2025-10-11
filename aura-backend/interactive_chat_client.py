#!/usr/bin/env python3
"""
Interactive WebSocket Chat Client
Simple command-line interface for testing real-time chat
"""

import asyncio
import websockets
import json
import sys
from datetime import datetime
import requests
import argparse


BASE_URL = "http://localhost:8000"
WS_BASE_URL = "ws://localhost:8000"


async def chat_client(username: str, password: str, conversation_id: str = None):
    """Interactive chat client"""
    
    print(f"\n🚀 Starting chat client for user: {username}\n")
    
    # Login
    print("🔑 Logging in...")
    login_data = {"username": username, "password": password}
    response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
    
    if response.status_code != 200:
        print(f"❌ Login failed: {response.text}")
        return
    
    token = response.json()["access_token"]
    print("✅ Logged in successfully\n")
    
    # Create or use existing conversation
    if not conversation_id:
        print("📝 Creating new conversation...")
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(
            f"{BASE_URL}/conversations",
            json={"title": f"{username}'s chat"},
            headers=headers
        )
        
        if response.status_code != 201:
            print(f"❌ Failed to create conversation: {response.text}")
            return
        
        conversation_id = response.json()["id"]
        print(f"✅ Created conversation: {conversation_id}\n")
    
    # Connect to WebSocket
    ws_url = f"{WS_BASE_URL}/ws/conversations/{conversation_id}?token={token}"
    print(f"🔌 Connecting to WebSocket: {conversation_id}")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ Connected to chat!\n")
            print("="*60)
            print("Commands:")
            print("  - Type a message and press Enter to send")
            print("  - /users - Show active users")
            print("  - /ping - Send a ping")
            print("  - /quit - Exit chat")
            print("="*60 + "\n")
            
            async def receive_messages():
                """Listen for incoming messages"""
                try:
                    while True:
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        msg_type = data.get("type")
                        
                        if msg_type == "message":
                            sender = data.get("sender", {})
                            sender_name = sender.get("username", "Unknown")
                            content = data.get("content", "")
                            role = data.get("role", "user")
                            timestamp = data.get("timestamp", "")
                            
                            # Format timestamp
                            try:
                                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                time_str = dt.strftime("%H:%M:%S")
                            except:
                                time_str = ""
                            
                            icon = "🤖" if role == "assistant" else "💬"
                            print(f"{icon} [{time_str}] {sender_name}: {content}")
                        
                        elif msg_type == "system":
                            content = data.get("content", "")
                            print(f"ℹ️  System: {content}")
                        
                        elif msg_type == "active_users":
                            users = data.get("users", [])
                            count = data.get("count", 0)
                            print(f"\n👥 Active Users ({count}):")
                            for user in users:
                                print(f"   - {user.get('username')} ({user.get('full_name')})")
                            print()
                        
                        elif msg_type == "error":
                            content = data.get("content", "")
                            print(f"⚠️  Error: {content}")
                        
                        elif msg_type == "pong":
                            print(f"🏓 Pong received")
                
                except websockets.exceptions.ConnectionClosed:
                    print("\n🔌 Connection closed by server")
                except Exception as e:
                    print(f"\n❌ Error receiving message: {e}")
            
            # Start receiving messages in background
            receive_task = asyncio.create_task(receive_messages())
            
            # Main input loop
            try:
                while True:
                    # Get user input
                    try:
                        user_input = await asyncio.get_event_loop().run_in_executor(
                            None, input, ""
                        )
                    except EOFError:
                        break
                    
                    user_input = user_input.strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle commands
                    if user_input == "/quit":
                        print("👋 Goodbye!")
                        break
                    
                    elif user_input == "/users":
                        await websocket.send(json.dumps({"type": "get_active_users"}))
                    
                    elif user_input == "/ping":
                        await websocket.send(json.dumps({"type": "ping"}))
                    
                    elif user_input.startswith("/"):
                        print(f"⚠️  Unknown command: {user_input}")
                    
                    else:
                        # Send as message
                        message = {
                            "type": "message",
                            "content": user_input,
                            "role": "user"
                        }
                        await websocket.send(json.dumps(message))
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
            finally:
                receive_task.cancel()
    
    except Exception as e:
        print(f"❌ Connection error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Interactive WebSocket Chat Client")
    parser.add_argument("username", help="Username to login")
    parser.add_argument("password", help="Password for the user")
    parser.add_argument("--conversation", "-c", help="Conversation ID to join (optional)")
    
    args = parser.parse_args()
    
    # Check backend
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend is not healthy")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to backend at {BASE_URL}: {e}")
        print("Please ensure the backend is running on port 8000")
        sys.exit(1)
    
    # Run chat client
    try:
        asyncio.run(chat_client(args.username, args.password, args.conversation))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
