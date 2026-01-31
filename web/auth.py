"""
Authentication module for INSIGHT Web Application.

Provides:
- Password hashing and verification
- JWT token generation and validation
- User authentication middleware
"""

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Try to use jose for JWT, fallback to simple tokens
try:
    from jose import JWTError, jwt
    JOSE_AVAILABLE = True
except ImportError:
    JOSE_AVAILABLE = False

from web.database import Database, User, get_db


# =============================================================================
# Configuration
# =============================================================================

SECRET_KEY = os.environ.get("INSIGHT_SECRET_KEY", secrets.token_hex(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

security = HTTPBearer(auto_error=False)


# =============================================================================
# Password Hashing
# =============================================================================

class AuthManager:
    """
    Authentication manager for password hashing and verification.
    """

    def __init__(self, secret_key: str = SECRET_KEY):
        """Initialize auth manager."""
        self.secret_key = secret_key

    # OWASP 2023 recommended minimum for PBKDF2-SHA256
    PBKDF2_ITERATIONS = 310000

    def hash_password(self, password: str) -> str:
        """
        Hash a password using PBKDF2-SHA256.

        Uses OWASP 2023 recommended iteration count for PBKDF2.

        Args:
            password: Plain text password

        Returns:
            Hashed password string with version prefix
        """
        salt = secrets.token_hex(16)
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            self.PBKDF2_ITERATIONS
        )
        # Include version prefix for future algorithm upgrades
        return f"v2${salt}${key.hex()}"

    def verify_password(self, password: str, password_hash: str) -> bool:
        """
        Verify a password against its hash.

        Supports both legacy (v1/no version) and current (v2) hash formats.

        Args:
            password: Plain text password
            password_hash: Stored password hash

        Returns:
            True if password matches
        """
        try:
            parts = password_hash.split('$')

            # Detect hash version
            if parts[0] == 'v2':
                # New format: v2$salt$hash
                salt = parts[1]
                stored_key = parts[2]
                iterations = self.PBKDF2_ITERATIONS
            else:
                # Legacy format: salt$hash (100k iterations)
                salt = parts[0]
                stored_key = parts[1]
                iterations = 100000

            key = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                iterations
            )
            return hmac.compare_digest(key.hex(), stored_key)
        except (ValueError, AttributeError, IndexError):
            return False


# =============================================================================
# JWT Token Management
# =============================================================================

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: Data to encode in token
        expires_delta: Optional expiration time delta

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire.isoformat(), "iat": datetime.utcnow().isoformat()})

    if JOSE_AVAILABLE:
        # Convert back to datetime for jose
        to_encode["exp"] = expire
        to_encode["iat"] = datetime.utcnow()
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    else:
        # Fallback with HMAC signing for security
        import base64
        import json

        payload_bytes = json.dumps(to_encode, default=str).encode()
        payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode()

        # Create HMAC signature
        signature = hmac.new(
            SECRET_KEY.encode(),
            payload_bytes,
            hashlib.sha256
        ).hexdigest()

        # Combine payload and signature
        encoded_jwt = f"{payload_b64}.{signature}"

    return encoded_jwt


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and validate a JWT token.

    Args:
        token: JWT token string

    Returns:
        Decoded token data or None if invalid
    """
    try:
        if JOSE_AVAILABLE:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        else:
            import base64
            import json

            # Split token into payload and signature
            parts = token.split('.')
            if len(parts) != 2:
                return None

            payload_b64, received_signature = parts

            # Decode payload
            payload_bytes = base64.urlsafe_b64decode(payload_b64.encode())

            # Verify HMAC signature
            expected_signature = hmac.new(
                SECRET_KEY.encode(),
                payload_bytes,
                hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(received_signature, expected_signature):
                return None

            payload = json.loads(payload_bytes.decode())

            # Check expiration
            if "exp" in payload:
                exp = datetime.fromisoformat(payload["exp"])
                if exp < datetime.utcnow():
                    return None

        return payload

    except Exception:
        return None


# =============================================================================
# Authentication Dependency
# =============================================================================

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Database = Depends(get_db)
) -> Optional[User]:
    """
    Get current authenticated user from request.

    Args:
        credentials: HTTP bearer credentials
        db: Database instance

    Returns:
        User object if authenticated, None otherwise
    """
    if not credentials:
        return None

    token = credentials.credentials
    payload = decode_token(token)

    if not payload:
        return None

    username = payload.get("sub")
    if not username:
        return None

    user = await db.get_user_by_username(username)
    return user


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Database = Depends(get_db)
) -> User:
    """
    Require authentication for endpoint.

    Args:
        credentials: HTTP bearer credentials
        db: Database instance

    Returns:
        Authenticated user

    Raises:
        HTTPException: If not authenticated
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"}
        )

    token = credentials.credentials
    payload = decode_token(token)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    username = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )

    user = await db.get_user_by_username(username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    return user


async def require_admin(user: User = Depends(require_auth)) -> User:
    """
    Require admin privileges.

    Args:
        user: Authenticated user

    Returns:
        Admin user

    Raises:
        HTTPException: If not admin
    """
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return user


# =============================================================================
# API Key Authentication
# =============================================================================

class APIKeyAuth:
    """
    API key authentication for programmatic access.
    """

    @staticmethod
    def generate_api_key() -> tuple[str, str]:
        """
        Generate a new API key.

        Returns:
            Tuple of (plain_key, key_hash)
        """
        plain_key = f"insight_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(plain_key.encode()).hexdigest()
        return plain_key, key_hash

    @staticmethod
    def verify_api_key(key: str, key_hash: str) -> bool:
        """
        Verify an API key.

        Args:
            key: Plain API key
            key_hash: Stored key hash

        Returns:
            True if key is valid
        """
        computed_hash = hashlib.sha256(key.encode()).hexdigest()
        return hmac.compare_digest(computed_hash, key_hash)


async def get_api_key_user(
    api_key: Optional[str] = None,
    db: Database = Depends(get_db)
) -> Optional[User]:
    """
    Get user from API key.

    Args:
        api_key: API key from header
        db: Database instance

    Returns:
        User if valid API key, None otherwise
    """
    if not api_key:
        return None

    # Query database for API key
    row = await db.fetchone(
        """
        SELECT u.* FROM users u
        JOIN api_keys ak ON u.id = ak.user_id
        WHERE ak.key_hash = ? AND ak.is_active = 1
        """,
        (hashlib.sha256(api_key.encode()).hexdigest(),)
    )

    if row:
        # Update last used
        await db.execute(
            "UPDATE api_keys SET last_used = CURRENT_TIMESTAMP WHERE key_hash = ?",
            (hashlib.sha256(api_key.encode()).hexdigest(),)
        )
        await db.commit()

        return User(
            id=row["id"],
            username=row["username"],
            email=row["email"],
            password_hash=row["password_hash"],
            is_admin=bool(row["is_admin"]),
            created_at=datetime.fromisoformat(row["created_at"])
        )

    return None
