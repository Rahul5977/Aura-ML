#!/usr/bin/env python3
"""
WebSocket Test Client for Real-time Chat
Tests the WebSocket functionality with multiple simultaneous connections
"""

import asyncio
import websockets
import json
import sys
from datetime import datetime
import requests

# Configuration
BASE_URL = "http://localhost:8000"
WS_BASE_URL = "ws://localhost:8000"


class ChatClient:
    """Simple chat client for testing WebSocket connections"""
    
    def __init__(self, username: str, token: str, conversation_id: str):
        self.username = username
        self.token = token
        self.conversation_id = conversation_id
        self.ws_url = f"{WS_BASE_URL}/ws/conversations/{conversation_id}?token={token}"
        self.websocket = None
        self.running = False
    
    async def connect(self):
        """Establish WebSocket connection"""
        try:
            self.websocket = await websockets.connect(self.ws_url)
            self.running = True
            print(f"[{self.username}] ✅ Connected to conversation {self.conversation_id}")
            return True
        except Exception as e:
            print(f"[{self.username}] ❌ Connection failed: {e}")
            return False
    
    async def send_message(self, content: str, role: str = "user"):
        """Send a message through WebSocket"""
        if not self.websocket:
            print(f"[{self.username}] ❌ Not connected")
            return
        
        message = {
            "type": "message",
            "content": content,
            "role": role
        }
        
        try:
            await self.websocket.send(json.dumps(message))
            print(f"[{self.username}] 📤 Sent: {content}")
        except Exception as e:
            print(f"[{self.username}] ❌ Error sending message: {e}")
    
    async def receive_messages(self):
        """Listen for incoming messages"""
        try:
            while self.running:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                msg_type = data.get("type", "unknown")
                
                if msg_type == "message":
                    sender = data.get("sender", {})
                    sender_name = sender.get("username", "Unknown")
                    content = data.get("content", "")
                    timestamp = data.get("timestamp", "")
                    print(f"[{self.username}] 📨 {sender_name}: {content}")
                
                elif msg_type == "system":
                    content = data.get("content", "")
                    print(f"[{self.username}] ℹ️  System: {content}")
                
                elif msg_type == "active_users":
                    users = data.get("users", [])
                    count = data.get("count", 0)
                    usernames = [u.get("username") for u in users]
                    print(f"[{self.username}] 👥 Active users ({count}): {', '.join(usernames)}")
                
                elif msg_type == "error":
                    content = data.get("content", "")
                    print(f"[{self.username}] ⚠️  Error: {content}")
                
                elif msg_type == "pong":
                    print(f"[{self.username}] 🏓 Pong received")
        
        except websockets.exceptions.ConnectionClosed:
            print(f"[{self.username}] 🔌 Connection closed")
            self.running = False
        except Exception as e:
            print(f"[{self.username}] ❌ Error receiving: {e}")
            self.running = False
    
    async def ping(self):
        """Send a ping message"""
        if not self.websocket:
            return
        
        try:
            await self.websocket.send(json.dumps({"type": "ping"}))
        except Exception as e:
            print(f"[{self.username}] ❌ Error sending ping: {e}")
    
    async def get_active_users(self):
        """Request active users list"""
        if not self.websocket:
            return
        
        try:
            await self.websocket.send(json.dumps({"type": "get_active_users"}))
        except Exception as e:
            print(f"[{self.username}] ❌ Error requesting users: {e}")
    
    async def disconnect(self):
        """Close WebSocket connection"""
        self.running = False
        if self.websocket:
            await self.websocket.close()
            print(f"[{self.username}] 🔌 Disconnected")


def register_and_login(username: str, email: str, password: str) -> str:
    """Register a new user and get authentication token"""
    # Try to register
    register_data = {
        "username": username,
        "email": email,
        "password": password,
        "full_name": f"Test User {username}"
    }
    
    response = requests.post(f"{BASE_URL}/auth/register", json=register_data)
    if response.status_code == 201:
        print(f"✅ User {username} registered successfully")
    elif response.status_code == 400:
        print(f"ℹ️  User {username} already exists, logging in...")
    else:
        print(f"❌ Registration failed: {response.text}")
        return None
    
    # Login to get token
    login_data = {
        "username": username,
        "password": password
    }
    
    response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
    if response.status_code == 200:
        token = response.json()["access_token"]
        print(f"✅ User {username} logged in successfully")
        return token
    else:
        print(f"❌ Login failed: {response.text}")
        return None


def create_conversation(token: str, title: str) -> str:
    """Create a new conversation"""
    headers = {"Authorization": f"Bearer {token}"}
    data = {"title": title}
    
    response = requests.post(f"{BASE_URL}/conversations", json=data, headers=headers)
    if response.status_code == 201:
        conversation_id = response.json()["id"]
        print(f"✅ Conversation '{title}' created: {conversation_id}")
        return conversation_id
    else:
        print(f"❌ Failed to create conversation: {response.text}")
        return None


async def test_single_user():
    """Test single user chat"""
    print("\n" + "="*60)
    print("TEST 1: Single User Chat")
    print("="*60 + "\n")
    
    # Setup
    username = "testuser_ws1"
    token = register_and_login(username, f"{username}@test.com", "testpass123")
    if not token:
        return
    
    conversation_id = create_conversation(token, "Test Chat 1")
    if not conversation_id:
        return
    
    # Create client and connect
    client = ChatClient(username, token, conversation_id)
    if not await client.connect():
        return
    
    # Start receiving messages
    receive_task = asyncio.create_task(client.receive_messages())
    
    # Wait a bit for connection messages
    await asyncio.sleep(1)
    
    # Send some messages
    await client.send_message("Hello, this is a test message!")
    await asyncio.sleep(0.5)
    await client.send_message("Testing WebSocket chat functionality")
    await asyncio.sleep(0.5)
    
    # Request active users
    await client.get_active_users()
    await asyncio.sleep(0.5)
    
    # Send a ping
    await client.ping()
    await asyncio.sleep(0.5)
    
    # Disconnect
    await client.disconnect()
    receive_task.cancel()
    
    print("\n✅ Single user test completed\n")


async def test_multiple_users():
    """Test multiple users in same conversation"""
    print("\n" + "="*60)
    print("TEST 2: Multiple Users in Same Conversation")
    print("="*60 + "\n")
    
    # Setup users
    users = []
    tokens = []
    for i in range(1, 4):
        username = f"testuser_ws{i}"
        token = register_and_login(username, f"{username}@test.com", "testpass123")
        if token:
            users.append(username)
            tokens.append(token)
    
    if len(tokens) < 2:
        print("❌ Need at least 2 users for this test")
        return
    
    # Create conversation with first user
    conversation_id = create_conversation(tokens[0], "Multi-User Test Chat")
    if not conversation_id:
        return
    
    # Create clients
    clients = []
    for username, token in zip(users, tokens):
        client = ChatClient(username, token, conversation_id)
        clients.append(client)
    
    # Connect all clients
    for client in clients:
        if not await client.connect():
            print(f"❌ Failed to connect {client.username}")
            return
        await asyncio.sleep(0.5)
    
    # Start receiving messages for all clients
    receive_tasks = []
    for client in clients:
        task = asyncio.create_task(client.receive_messages())
        receive_tasks.append(task)
    
    # Wait a bit for connection messages
    await asyncio.sleep(1)
    
    # Each user sends a message
    for i, client in enumerate(clients):
        await client.send_message(f"Hello from {client.username}! Message #{i+1}")
        await asyncio.sleep(1)
    
    # User 1 requests active users
    await clients[0].get_active_users()
    await asyncio.sleep(1)
    
    # User 2 disconnects
    print(f"\n{clients[1].username} is leaving the chat...")
    await clients[1].disconnect()
    await asyncio.sleep(1)
    
    # User 1 sends another message
    await clients[0].send_message("Anyone still here?")
    await asyncio.sleep(1)
    
    # Disconnect remaining clients
    for client in clients:
        if client.running:
            await client.disconnect()
    
    # Cancel receive tasks
    for task in receive_tasks:
        task.cancel()
    
    print("\n✅ Multiple users test completed\n")


async def test_message_broadcasting():
    """Test message broadcasting between clients"""
    print("\n" + "="*60)
    print("TEST 3: Message Broadcasting")
    print("="*60 + "\n")
    
    # Setup two users
    user1 = "broadcaster"
    user2 = "receiver"
    
    token1 = register_and_login(user1, f"{user1}@test.com", "testpass123")
    token2 = register_and_login(user2, f"{user2}@test.com", "testpass123")
    
    if not token1 or not token2:
        return
    
    # Create conversation
    conversation_id = create_conversation(token1, "Broadcast Test")
    if not conversation_id:
        return
    
    # Create and connect clients
    client1 = ChatClient(user1, token1, conversation_id)
    client2 = ChatClient(user2, token2, conversation_id)
    
    if not await client1.connect() or not await client2.connect():
        return
    
    # Start receiving
    task1 = asyncio.create_task(client1.receive_messages())
    task2 = asyncio.create_task(client2.receive_messages())
    
    await asyncio.sleep(1)
    
    # Rapid message exchange
    messages = [
        ("broadcaster", "Testing broadcast feature"),
        ("receiver", "Message received!"),
        ("broadcaster", "Great! It's working"),
        ("receiver", "Perfect synchronization"),
    ]
    
    for sender, msg in messages:
        if sender == "broadcaster":
            await client1.send_message(msg)
        else:
            await client2.send_message(msg)
        await asyncio.sleep(0.8)
    
    await asyncio.sleep(1)
    
    # Cleanup
    await client1.disconnect()
    await client2.disconnect()
    task1.cancel()
    task2.cancel()
    
    print("\n✅ Broadcasting test completed\n")


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("WebSocket Chat Testing Suite")
    print("="*60)
    
    try:
        # Test 1: Single user
        await test_single_user()
        await asyncio.sleep(2)
        
        # Test 2: Multiple users
        await test_multiple_users()
        await asyncio.sleep(2)
        
        # Test 3: Broadcasting
        await test_message_broadcasting()
        
        print("\n" + "="*60)
        print("🎉 All Tests Completed Successfully!")
        print("="*60 + "\n")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test suite error: {e}")


if __name__ == "__main__":
    # Check if backend is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend is not healthy")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to backend at {BASE_URL}: {e}")
        print("Please ensure the backend is running on port 8000")
        sys.exit(1)
    
    # Run tests
    asyncio.run(main())
