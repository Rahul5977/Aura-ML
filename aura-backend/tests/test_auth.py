import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from main import app
from core.security import get_password_hash, verify_password, create_access_token
from db.prisma import connect_db, disconnect_db
from datetime import timedelta

# Test configuration
TEST_USER_DATA = {
    "username": "testuser",
    "email": "test@example.com",
    "password": "testpass123",
    "full_name": "Test User"
}

TEST_USER_LOGIN = {
    "username": "testuser",
    "password": "testpass123"
}

@pytest.fixture
async def async_client():
    """Create an async HTTP client for testing"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def client():
    """Create a test client"""
    return TestClient(app)

@pytest.fixture
async def setup_database():
    """Setup test database"""
    await connect_db()
    yield
    await disconnect_db()

class TestPasswordSecurity:
    """Test password hashing and verification"""
    
    def test_password_hashing(self):
        """Test that passwords are properly hashed"""
        password = "testpassword123"
        hashed = get_password_hash(password)
        
        # Hash should not be the same as original password
        assert hashed != password
        # Hash should be a string
        assert isinstance(hashed, str)
        # Hash should not be empty
        assert len(hashed) > 0
    
    def test_password_verification_success(self):
        """Test that correct passwords are verified successfully"""
        password = "testpassword123"
        hashed = get_password_hash(password)
        
        # Correct password should verify
        assert verify_password(password, hashed) is True
    
    def test_password_verification_failure(self):
        """Test that incorrect passwords fail verification"""
        password = "testpassword123"
        wrong_password = "wrongpassword"
        hashed = get_password_hash(password)
        
        # Wrong password should not verify
        assert verify_password(wrong_password, hashed) is False
    
    def test_password_length_limit(self):
        """Test that passwords longer than 72 characters are handled"""
        long_password = "a" * 100  # 100 character password
        hashed = get_password_hash(long_password)
        
        # Should still be able to hash and verify
        assert verify_password(long_password, hashed) is True
        # Should also verify with truncated version
        assert verify_password(long_password[:72], hashed) is True

class TestJWTTokens:
    """Test JWT token creation and verification"""
    
    def test_token_creation(self):
        """Test that JWT tokens are created correctly"""
        data = {"sub": "user123"}
        token = create_access_token(data)
        
        # Token should be a string
        assert isinstance(token, str)
        # Token should not be empty
        assert len(token) > 0
        # Token should contain dots (JWT format)
        assert token.count('.') == 2
    
    def test_token_with_expiration(self):
        """Test token creation with custom expiration"""
        data = {"sub": "user123"}
        expires_delta = timedelta(minutes=60)
        token = create_access_token(data, expires_delta)
        
        # Should create a valid token
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_token_verification_success(self):
        """Test that valid tokens are verified correctly"""
        from core.security import verify_token
        
        data = {"sub": "user123"}
        token = create_access_token(data)
        
        # Should verify successfully
        decoded_data = verify_token(token)
        assert decoded_data["user_id"] == "user123"
    
    def test_token_verification_invalid(self):
        """Test that invalid tokens raise exceptions"""
        from core.security import verify_token
        from fastapi import HTTPException
        
        invalid_token = "invalid.token.here"
        
        # Should raise HTTPException
        with pytest.raises(HTTPException):
            verify_token(invalid_token)

class TestAuthenticationEndpoints:
    """Test authentication API endpoints"""
    
    @pytest.mark.asyncio
    async def test_user_registration_success(self, async_client, setup_database):
        """Test successful user registration"""
        response = await async_client.post("/auth/register", json=TEST_USER_DATA)
        
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == TEST_USER_DATA["username"]
        assert data["email"] == TEST_USER_DATA["email"]
        assert data["full_name"] == TEST_USER_DATA["full_name"]
        assert data["is_active"] is True
        assert "id" in data
        assert "created_at" in data
        # Password should not be in response
        assert "password" not in data
        assert "password_hash" not in data
    
    @pytest.mark.asyncio
    async def test_user_registration_duplicate_username(self, async_client, setup_database):
        """Test registration with duplicate username"""
        # First registration
        await async_client.post("/auth/register", json=TEST_USER_DATA)
        
        # Second registration with same username
        duplicate_data = TEST_USER_DATA.copy()
        duplicate_data["email"] = "different@example.com"
        
        response = await async_client.post("/auth/register", json=duplicate_data)
        assert response.status_code == 400
        assert "Username already registered" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_user_registration_duplicate_email(self, async_client, setup_database):
        """Test registration with duplicate email"""
        # First registration
        await async_client.post("/auth/register", json=TEST_USER_DATA)
        
        # Second registration with same email
        duplicate_data = TEST_USER_DATA.copy()
        duplicate_data["username"] = "differentuser"
        
        response = await async_client.post("/auth/register", json=duplicate_data)
        assert response.status_code == 400
        assert "Email already registered" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_user_registration_invalid_data(self, async_client):
        """Test registration with invalid data"""
        invalid_data = {
            "username": "ab",  # Too short
            "email": "invalid-email",  # Invalid email
            "password": "123",  # Too short
            "full_name": ""  # Empty
        }
        
        response = await async_client.post("/auth/register", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_user_login_success(self, async_client, setup_database):
        """Test successful user login"""
        # Register user first
        await async_client.post("/auth/register", json=TEST_USER_DATA)
        
        # Login
        response = await async_client.post("/auth/login", json=TEST_USER_LOGIN)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
        # Token should be a valid string
        assert isinstance(data["access_token"], str)
        assert len(data["access_token"]) > 0
    
    @pytest.mark.asyncio
    async def test_user_login_wrong_password(self, async_client, setup_database):
        """Test login with wrong password"""
        # Register user first
        await async_client.post("/auth/register", json=TEST_USER_DATA)
        
        # Login with wrong password
        wrong_login = TEST_USER_LOGIN.copy()
        wrong_login["password"] = "wrongpassword"
        
        response = await async_client.post("/auth/login", json=wrong_login)
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_user_login_nonexistent_user(self, async_client):
        """Test login with non-existent user"""
        response = await async_client.post("/auth/login", json=TEST_USER_LOGIN)
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_user_login_with_email(self, async_client, setup_database):
        """Test login using email instead of username"""
        # Register user first
        await async_client.post("/auth/register", json=TEST_USER_DATA)
        
        # Login with email
        email_login = {
            "username": TEST_USER_DATA["email"],
            "password": TEST_USER_DATA["password"]
        }
        
        response = await async_client.post("/auth/login", json=email_login)
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data

class TestProtectedEndpoints:
    """Test protected API endpoints that require authentication"""
    
    async def get_auth_token(self, async_client):
        """Helper method to get authentication token"""
        # Register and login user
        await async_client.post("/auth/register", json=TEST_USER_DATA)
        login_response = await async_client.post("/auth/login", json=TEST_USER_LOGIN)
        return login_response.json()["access_token"]
    
    @pytest.mark.asyncio
    async def test_get_current_user_success(self, async_client, setup_database):
        """Test getting current user info with valid token"""
        token = await self.get_auth_token(async_client)
        
        response = await async_client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == TEST_USER_DATA["username"]
        assert data["email"] == TEST_USER_DATA["email"]
    
    @pytest.mark.asyncio
    async def test_get_current_user_no_token(self, async_client):
        """Test accessing protected endpoint without token"""
        response = await async_client.get("/auth/me")
        assert response.status_code == 403  # Forbidden
    
    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self, async_client):
        """Test accessing protected endpoint with invalid token"""
        response = await async_client.get(
            "/auth/me",
            headers={"Authorization": "Bearer invalid_token_here"}
        )
        assert response.status_code == 401  # Unauthorized
    
    @pytest.mark.asyncio
    async def test_get_current_user_expired_token(self, async_client, setup_database):
        """Test accessing protected endpoint with expired token"""
        # Create an expired token
        from datetime import datetime, timedelta
        data = {"sub": "user123", "exp": datetime.utcnow() - timedelta(minutes=30)}
        expired_token = create_access_token(data)
        
        response = await async_client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {expired_token}"}
        )
        assert response.status_code == 401  # Unauthorized
    
    @pytest.mark.asyncio
    async def test_protected_conversations_endpoint(self, async_client, setup_database):
        """Test that conversations endpoint requires authentication"""
        # Without token
        response = await async_client.get("/conversations")
        assert response.status_code == 403
        
        # With valid token
        token = await self.get_auth_token(async_client)
        response = await async_client.get(
            "/conversations",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        # Should return empty list for new user
        assert response.json() == []

if __name__ == "__main__":
    pytest.main([__file__])
