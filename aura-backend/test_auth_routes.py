#!/usr/bin/env python3
"""
Test script for additional auth routes in api/auth.py
These routes are not currently integrated into main.py but can be tested separately.
"""

import asyncio
import json
import sys
import os
sys.path.append('/Users/rahulraj/Desktop/ML_Proj/aura-backend')

from fastapi import FastAPI
from api.auth import router as auth_router
import uvicorn

# Create a temporary FastAPI app for testing
test_app = FastAPI(title="Auth Routes Test App")
test_app.include_router(auth_router)

@test_app.on_event("startup")
async def startup():
    # Initialize database connection
    from database import connect_db
    await connect_db()

@test_app.on_event("shutdown") 
async def shutdown():
    from database import disconnect_db
    await disconnect_db()

if __name__ == "__main__":
    print("Starting test server for additional auth routes on port 8001...")
    print("Available routes:")
    print("- PUT /auth/me - Update user profile")
    print("- POST /auth/change-password - Change user password") 
    print("- POST /auth/logout - Logout user")
    print("- GET /auth/me - Get current user (also available)")
    print("- POST /auth/register - Register new user (also available)")
    print("- POST /auth/login - Login user (also available)")
    
    uvicorn.run(test_app, host="0.0.0.0", port=8001, log_level="info")
