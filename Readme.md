# Aura - AI-Powered Chat Application

## Project Overview

Aura is a modern, real-time chat application built with FastAPI (backend) and React (frontend), featuring WebSocket-based communication, user authentication, and AI-powered conversations.

## 🚀 Current Status: Week 3 COMPLETED ✅

### Week 1: Project Setup & Database Design ✅

- PostgreSQL database with Prisma ORM
- User, Conversation, and Message models
- Docker containerization (backend, frontend, database)
- Database migrations and schema management

### Week 2: User Authentication & Basic API ✅

- JWT-based authentication system
- User registration and login endpoints
- Password hashing with bcrypt
- Protected API routes
- Conversation and message CRUD operations
- Comprehensive API testing

### Week 3: Real-time Communication & Chat Logic ✅

- **WebSocket endpoint for real-time chat**
- **Connection manager for multiple simultaneous users**
- **Message broadcasting to all connected clients**
- **Graceful connection/disconnection handling**
- **Message persistence to database**
- **System notifications (join/leave events)**
- **Active users tracking**
- **Comprehensive test suite and interactive client**

## 🔧 Tech Stack

### Backend

- **FastAPI**: Modern Python web framework
- **Uvicorn**: ASGI server with WebSocket support
- **Prisma**: Next-generation ORM for Python
- **PostgreSQL**: Relational database
- **JWT**: Authentication tokens
- **WebSockets**: Real-time bidirectional communication

## 🚀 Quick Start

```bash
# Start all services
docker-compose up --build

# Backend API: http://localhost:8000
# WebSocket: ws://localhost:8000/ws/conversations/{id}
```

## 📡 WebSocket Chat

### Quick Test

```bash
docker exec -it ml_proj-backend-1 python interactive_chat_client.py username password
```

See [WEBSOCKET_QUICKSTART.md](WEBSOCKET_QUICKSTART.md) and [WEEK3_COMPLETION_REPORT.md](WEEK3_COMPLETION_REPORT.md) for details.

---

**Last Updated**: October 11, 2025  
**Status**: Week 3 Complete - Real-time Chat Fully Functional ✅
