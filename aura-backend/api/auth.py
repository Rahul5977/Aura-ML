from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from core.dependencies import get_current_user
from core.security import create_access_token, verify_password, ACCESS_TOKEN_EXPIRE_MINUTES
from db.crud import create_user, authenticate_user, get_user_by_username, get_user_by_email, get_user_by_id, update_user_profile, update_user_password
from schemas.user import UserCreate, UserResponse, UserLogin, UserUpdate, PasswordChange
from schemas.token import Token
from datetime import timedelta
from typing import Optional

router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate):
    """
    Register a new user account.
    
    - **username**: Must be unique, 3-50 characters, alphanumeric with dots, dashes, underscores
    - **email**: Must be unique and valid email format
    - **password**: At least 8 characters with letters and numbers
    - **full_name**: User's full name, 2-100 characters
    """
    # Check if username already exists
    existing_user = await get_user_by_username(user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    existing_email = await get_user_by_email(user_data.email)
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    try:
        # Create new user
        user = await create_user(user_data)
        return user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user account"
        )

@router.post("/login", response_model=Token)
async def login(user_credentials: UserLogin):
    """
    Authenticate user and return JWT token.
    
    - **username**: Can be username or email
    - **password**: User's password
    
    Returns JWT token for authenticated requests.
    """
    # Authenticate user
    user = await authenticate_user(user_credentials.username, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user account is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is deactivated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.id, "username": user.username}, 
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60  # in seconds
    }

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user=Depends(get_current_user)):
    """
    Get current authenticated user's information.
    
    Requires valid JWT token in Authorization header.
    """
    return current_user

@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user=Depends(get_current_user)
):
    """
    Update current user's profile information.
    
    - **email**: New email address (optional)
    - **full_name**: New full name (optional)
    """
    # Check if email is being updated and if it's already taken
    if user_update.email and user_update.email != current_user.email:
        existing_email = await get_user_by_email(user_update.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    # Update user (implement this in crud.py)
    try:
        # This would need to be implemented in crud.py
        updated_user = await update_user_profile(current_user.id, user_update)
        return updated_user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user profile"
        )

@router.post("/change-password")
async def change_password(
    password_change: PasswordChange,
    current_user=Depends(get_current_user)
):
    """
    Change user's password.
    
    - **current_password**: Current password for verification
    - **new_password**: New password (must meet security requirements)
    """
    # Verify current password
    if not verify_password(password_change.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Update password (implement this in crud.py)
    try:
        # This would need to be implemented in crud.py
        await update_user_password(current_user.id, password_change.new_password)
        return {"message": "Password updated successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update password"
        )

@router.post("/logout")
async def logout(current_user=Depends(get_current_user)):
    """
    Logout user (client should discard the JWT token).
    
    Note: JWTs are stateless, so actual logout is handled client-side.
    This endpoint can be used for logging purposes.
    """
    # In a real application, you might want to maintain a blacklist of tokens
    # or implement token refresh logic
    return {"message": "Logged out successfully"}
