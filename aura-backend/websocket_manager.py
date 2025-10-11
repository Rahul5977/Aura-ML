"""
WebSocket Connection Manager for Real-time Chat
Manages WebSocket connections, message broadcasting, and conversation state
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Set
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for real-time chat functionality.
    Handles connection lifecycle, message broadcasting, and active user tracking.
    """
    
    def __init__(self):
        # Store active connections: {conversation_id: {user_id: websocket}}
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        # Track user info for each connection: {conversation_id: {user_id: user_data}}
        self.active_users: Dict[str, Dict[str, dict]] = {}
        
    async def connect(self, websocket: WebSocket, conversation_id: str, user_id: str, user_data: dict):
        """
        Accept and register a new WebSocket connection.
        
        Args:
            websocket: The WebSocket connection object
            conversation_id: ID of the conversation
            user_id: ID of the user connecting
            user_data: User information (username, full_name, etc.)
        """
        await websocket.accept()
        
        # Initialize conversation dict if it doesn't exist
        if conversation_id not in self.active_connections:
            self.active_connections[conversation_id] = {}
            self.active_users[conversation_id] = {}
        
        # Store the connection
        self.active_connections[conversation_id][user_id] = websocket
        self.active_users[conversation_id][user_id] = user_data
        
        logger.info(
            f"User {user_data.get('username')} ({user_id}) connected to conversation {conversation_id}. "
            f"Total active connections: {len(self.active_connections[conversation_id])}"
        )
        
        # Notify other users about the new connection
        await self.broadcast_system_message(
            conversation_id,
            f"{user_data.get('username')} joined the chat",
            exclude_user_id=user_id
        )
        
        # Send active users list to the newly connected user
        await self.send_active_users(websocket, conversation_id)
    
    async def disconnect(self, conversation_id: str, user_id: str):
        """
        Remove a WebSocket connection and clean up.
        
        Args:
            conversation_id: ID of the conversation
            user_id: ID of the user disconnecting
        """
        if conversation_id in self.active_connections:
            user_data = self.active_users[conversation_id].get(user_id, {})
            username = user_data.get('username', 'Unknown user')
            
            # Remove the connection
            if user_id in self.active_connections[conversation_id]:
                del self.active_connections[conversation_id][user_id]
            
            # Remove user data
            if user_id in self.active_users[conversation_id]:
                del self.active_users[conversation_id][user_id]
            
            logger.info(
                f"User {username} ({user_id}) disconnected from conversation {conversation_id}. "
                f"Remaining connections: {len(self.active_connections[conversation_id])}"
            )
            
            # Notify other users about the disconnection
            await self.broadcast_system_message(
                conversation_id,
                f"{username} left the chat"
            )
            
            # Clean up empty conversation
            if not self.active_connections[conversation_id]:
                del self.active_connections[conversation_id]
                del self.active_users[conversation_id]
                logger.info(f"Conversation {conversation_id} has no active connections, cleaned up")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """
        Send a message to a specific WebSocket connection.
        
        Args:
            message: The message data to send
            websocket: The target WebSocket connection
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
    
    async def broadcast_message(self, conversation_id: str, message: dict, exclude_user_id: str = None):
        """
        Broadcast a message to all connections in a conversation.
        
        Args:
            conversation_id: ID of the conversation
            message: The message data to broadcast
            exclude_user_id: Optional user ID to exclude from broadcast (e.g., the sender)
        """
        if conversation_id not in self.active_connections:
            logger.warning(f"Attempted to broadcast to non-existent conversation {conversation_id}")
            return
        
        # Get all connections for this conversation
        connections = self.active_connections[conversation_id]
        disconnected_users = []
        
        # Send to all connections except the excluded user
        for user_id, websocket in connections.items():
            if exclude_user_id and user_id == exclude_user_id:
                continue
            
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to user {user_id}: {e}")
                disconnected_users.append(user_id)
        
        # Clean up disconnected users
        for user_id in disconnected_users:
            await self.disconnect(conversation_id, user_id)
    
    async def broadcast_system_message(self, conversation_id: str, content: str, exclude_user_id: str = None):
        """
        Broadcast a system message to all connections in a conversation.
        
        Args:
            conversation_id: ID of the conversation
            content: The system message content
            exclude_user_id: Optional user ID to exclude from broadcast
        """
        system_message = {
            "type": "system",
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast_message(conversation_id, system_message, exclude_user_id)
    
    async def send_active_users(self, websocket: WebSocket, conversation_id: str):
        """
        Send the list of active users to a specific connection.
        
        Args:
            websocket: The target WebSocket connection
            conversation_id: ID of the conversation
        """
        if conversation_id not in self.active_users:
            return
        
        active_users_list = [
            {
                "user_id": user_id,
                "username": user_data.get("username"),
                "full_name": user_data.get("full_name")
            }
            for user_id, user_data in self.active_users[conversation_id].items()
        ]
        
        message = {
            "type": "active_users",
            "users": active_users_list,
            "count": len(active_users_list)
        }
        
        await self.send_personal_message(message, websocket)
    
    def get_active_users_count(self, conversation_id: str) -> int:
        """
        Get the number of active users in a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Number of active connections
        """
        if conversation_id not in self.active_connections:
            return 0
        return len(self.active_connections[conversation_id])
    
    def is_user_connected(self, conversation_id: str, user_id: str) -> bool:
        """
        Check if a user is connected to a conversation.
        
        Args:
            conversation_id: ID of the conversation
            user_id: ID of the user
            
        Returns:
            True if user is connected, False otherwise
        """
        if conversation_id not in self.active_connections:
            return False
        return user_id in self.active_connections[conversation_id]


# Global connection manager instance
manager = ConnectionManager()
