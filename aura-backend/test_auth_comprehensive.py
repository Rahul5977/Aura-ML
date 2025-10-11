#!/usr/bin/env python3
"""
Comprehensive test script for all auth routes including the additional ones
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8001"  # Test server port
MAIN_URL = "http://localhost:8000"  # Main application port

def test_main_auth_routes():
    """Test the working auth routes from main.py"""
    print("=== Testing Main Application Auth Routes (Port 8000) ===")
    
    # Test user registration
    print("\n1. Testing user registration...")
    register_data = {
        "username": f"testuser_{int(time.time())}",
        "email": f"test_{int(time.time())}@example.com", 
        "password": "testpassword123",
        "full_name": "Test User"
    }
    
    response = requests.post(f"{MAIN_URL}/auth/register", json=register_data)
    print(f"Registration status: {response.status_code}")
    if response.status_code == 201:
        print("✅ Registration successful")
        user_data = response.json()
        print(f"User ID: {user_data['id']}")
    else:
        print(f"❌ Registration failed: {response.text}")
        return None, None
    
    # Test user login
    print("\n2. Testing user login...")
    login_data = {
        "username": register_data["username"],
        "password": register_data["password"]
    }
    
    response = requests.post(f"{MAIN_URL}/auth/login", json=login_data)
    print(f"Login status: {response.status_code}")
    if response.status_code == 200:
        print("✅ Login successful")
        token_data = response.json()
        token = token_data["access_token"]
        print(f"Token received (first 20 chars): {token[:20]}...")
    else:
        print(f"❌ Login failed: {response.text}")
        return None, None
    
    # Test getting current user info
    print("\n3. Testing get current user...")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{MAIN_URL}/auth/me", headers=headers)
    print(f"Get user status: {response.status_code}")
    if response.status_code == 200:
        print("✅ Get current user successful")
        current_user = response.json()
        print(f"Current user: {current_user['username']}")
    else:
        print(f"❌ Get current user failed: {response.text}")
    
    return register_data, token

def test_additional_auth_routes(register_data, token):
    """Test the additional auth routes from api/auth.py"""
    print("\n=== Testing Additional Auth Routes (Port 8001) ===")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test updating user profile
    print("\n1. Testing update user profile...")
    update_data = {
        "full_name": "Updated Test User",
        "email": f"updated_{int(time.time())}@example.com"
    }
    
    response = requests.put(f"{BASE_URL}/auth/me", json=update_data, headers=headers)
    print(f"Update profile status: {response.status_code}")
    if response.status_code == 200:
        print("✅ Profile update successful")
        updated_user = response.json()
        print(f"Updated name: {updated_user.get('full_name')}")
        print(f"Updated email: {updated_user.get('email')}")
    else:
        print(f"❌ Profile update failed: {response.text}")
    
    # Test changing password
    print("\n2. Testing change password...")
    password_data = {
        "current_password": register_data["password"],
        "new_password": "newpassword123"
    }
    
    response = requests.post(f"{BASE_URL}/auth/change-password", json=password_data, headers=headers)
    print(f"Change password status: {response.status_code}")
    if response.status_code == 200:
        print("✅ Password change successful")
        result = response.json()
        print(f"Message: {result.get('message')}")
    else:
        print(f"❌ Password change failed: {response.text}")
    
    # Test logout
    print("\n3. Testing logout...")
    response = requests.post(f"{BASE_URL}/auth/logout", headers=headers)
    print(f"Logout status: {response.status_code}")
    if response.status_code == 200:
        print("✅ Logout successful")
        result = response.json()
        print(f"Message: {result.get('message')}")
    else:
        print(f"❌ Logout failed: {response.text}")

def test_error_cases():
    """Test error handling for the additional routes"""
    print("\n=== Testing Error Cases ===")
    
    # Test without authentication
    print("\n1. Testing routes without authentication...")
    
    routes_to_test = [
        ("PUT", f"{BASE_URL}/auth/me"),
        ("POST", f"{BASE_URL}/auth/change-password"),
        ("POST", f"{BASE_URL}/auth/logout")
    ]
    
    for method, url in routes_to_test:
        if method == "PUT":
            response = requests.put(url, json={"full_name": "Test"})
        else:
            response = requests.post(url, json={})
        
        print(f"{method} {url}: {response.status_code}")
        if response.status_code == 401:
            print("✅ Correctly requires authentication")
        else:
            print(f"❌ Expected 401, got {response.status_code}: {response.text}")

def main():
    print("Starting comprehensive auth routes test...")
    
    # First test the main working routes
    register_data, token = test_main_auth_routes()
    
    if not register_data or not token:
        print("❌ Cannot proceed with additional tests - main auth failed")
        return
    
    # Test if the additional auth server is running
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            print(f"\n✅ Additional auth test server is running on port 8001")
            
            # Test the additional routes
            test_additional_auth_routes(register_data, token)
            test_error_cases()
        else:
            print(f"\n❌ Test server returned {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Test server not running on port 8001: {e}")
        print("To test additional routes, please run: python test_auth_routes.py")
    
    print("\n=== Test Summary ===")
    print("✅ Main auth routes (register, login, get user) are working")
    print("⚠️  Additional routes (update profile, change password, logout) need test server")

if __name__ == "__main__":
    main()
