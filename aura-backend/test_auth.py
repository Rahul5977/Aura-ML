import pytest
from fastapi.testclient import TestClient
from main import app
import asyncio
from database import connect_db, disconnect_db, prisma

client = TestClient(app)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session", autouse=True)
async def setup_database():
    """Setup and teardown database for tests."""
    await connect_db()
    yield
    await disconnect_db()

@pytest.fixture
async def cleanup_users():
    """Clean up users after each test."""
    yield
    # Clean up test users
    await prisma.user.delete_many(
        where={"username": {"in": ["testuser", "testuser2"]}}
    )

class TestAuth:
    def test_health_endpoint(self):
        """Test health endpoint is accessible."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_user_registration(self, cleanup_users):
        """Test user registration."""
        user_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password": "testpassword123"
        }
        response = client.post("/auth/register", json=user_data)
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == user_data["email"]
        assert data["username"] == user_data["username"]
        assert "id" in data
        assert "hashed_password" not in data

    @pytest.mark.asyncio
    async def test_duplicate_username_registration(self, cleanup_users):
        """Test registration with duplicate username fails."""
        user_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password": "testpassword123"
        }
        # First registration should succeed
        response = client.post("/auth/register", json=user_data)
        assert response.status_code == 201

        # Second registration with same username should fail
        user_data2 = {
            "email": "test2@example.com",
            "username": "testuser",
            "password": "testpassword123"
        }
        response = client.post("/auth/register", json=user_data2)
        assert response.status_code == 400
        assert "Username already registered" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_user_login(self, cleanup_users):
        """Test user login and token generation."""
        # First register a user
        user_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password": "testpassword123"
        }
        client.post("/auth/register", json=user_data)

        # Then login
        login_data = {
            "username": "testuser",
            "password": "testpassword123"
        }
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_invalid_login(self, cleanup_users):
        """Test login with invalid credentials fails."""
        login_data = {
            "username": "nonexistent",
            "password": "wrongpassword"
        }
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_protected_endpoint_without_token(self):
        """Test accessing protected endpoint without token fails."""
        response = client.get("/auth/me")
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_protected_endpoint_with_valid_token(self, cleanup_users):
        """Test accessing protected endpoint with valid token."""
        # Register and login
        user_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password": "testpassword123"
        }
        client.post("/auth/register", json=user_data)
        
        login_response = client.post("/auth/login", json={
            "username": "testuser",
            "password": "testpassword123"
        })
        token = login_response.json()["access_token"]

        # Access protected endpoint
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/auth/me", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"

    @pytest.mark.asyncio
    async def test_create_conversation(self, cleanup_users):
        """Test creating a conversation."""
        # Register and login
        user_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password": "testpassword123"
        }
        client.post("/auth/register", json=user_data)
        
        login_response = client.post("/auth/login", json={
            "username": "testuser",
            "password": "testpassword123"
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Create conversation
        conversation_data = {"title": "Test Conversation"}
        response = client.post("/conversations", json=conversation_data, headers=headers)
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Test Conversation"
        assert "id" in data
