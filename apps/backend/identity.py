"""
The Acting identity module: who is acting (a user id), as a rep or as an admin, and on whose data.

Every way a request proves who it is goes through here:

    Authorization: Bearer <access JWT>      the signed-in account (signature, expiry, token_version)
    + X-Admin-Console: 1                    acting as an admin over everyone's data (403 for non-admins)
    + ?user_id= on the legacy app routes    must be the token's own account — nobody, admins included,
                                            acts as another rep through the rep routes (403)
    ?ticket= on the console's SSE feed      a one-use, 60-second ticket an admin got with their token

    acting(authorization, console=...)  → Acting(user_id, as_admin); .owner = whose data (None = everyone's)
    admin_user(authorization)           → the admin's id (401 / 403)
    legacy_user(authorization, user_id) → the rep the legacy route acts for (401 / 403)
    data_owner(me, explicit, ...)       → whose rows a signed-in rep or the console works on

The FastAPI dependencies the routes use (api_routes.require_admin / request_user_id / require_admin_stream,
app.verify_api_key / get_current_user_id / _chat_identity) are thin shims over these.
The JWTs themselves (HS256, dependency-free) are made and checked here too.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

from fastapi import HTTPException
from sqlalchemy import select

from database import UserDB, get_session_factory

# Returned (401) when a token predates the user's current token_version. Clients match on it to sign out.
SESSION_REVOKED_DETAIL = "Your session was signed out. Please sign in again."
# The apps refresh their access token and retry once on this exact detail.
TOKEN_EXPIRED_DETAIL = "Authentication token expired"
OTHER_ACCOUNT_DETAIL = "That belongs to another account"


# ---------------------------------------------------------------- JWT (HS256)

# HS256 signing secret. Must NOT be APP_API_KEY: that key ships inside the mobile app, so anyone could
# forge tokens (incl. admin) with it. Unset JWT_SECRET → a random secret is generated once and kept in
# JWT_SECRET_FILE (default ./jwt_secret.key) so sessions survive restarts on the same disk.
def _load_jwt_secret() -> str:
    env_secret = os.environ.get("JWT_SECRET", "").strip()
    if env_secret:
        return env_secret
    path = os.environ.get("JWT_SECRET_FILE") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "jwt_secret.key")
    try:
        with open(path, encoding="utf-8") as fh:
            stored = fh.read().strip()
        if len(stored) >= 32:
            return stored
    except OSError:
        pass
    generated = secrets.token_urlsafe(48)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(generated)
        print(f"[CONFIG] JWT_SECRET unset — generated one and saved it to {path}")
    except OSError as err:
        print(f"[CONFIG] JWT_SECRET unset and {path} not writable ({err}) — sessions end on every restart")
    return generated


JWT_SECRET = _load_jwt_secret()
JWT_ACCESS_TTL = timedelta(hours=int(os.environ.get("JWT_ACCESS_TTL_HOURS", "12")))
JWT_REFRESH_TTL = timedelta(days=int(os.environ.get("JWT_REFRESH_TTL_DAYS", "30")))


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def create_jwt_token(claims: dict, ttl: timedelta) -> str:
    """Create a signed HS256 JWT. `claims` is merged with standard iat/exp fields."""
    header = {"alg": "HS256", "typ": "JWT"}
    now = datetime.now(timezone.utc)
    payload = {**claims, "iat": int(now.timestamp()), "exp": int((now + ttl).timestamp())}
    segments = [
        _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8")),
        _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
    ]
    signing_input = ".".join(segments).encode("ascii")
    signature = hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
    segments.append(_b64url_encode(signature))
    return ".".join(segments)


def decode_jwt_token(token: str, verify_exp: bool = True) -> dict:
    """Verify signature (and optionally expiry) and return the JWT payload. Raises 401 on failure."""
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
    except (ValueError, AttributeError):
        raise HTTPException(status_code=401, detail="Malformed authentication token")

    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected_sig = hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
    try:
        actual_sig = _b64url_decode(signature_b64)
    except Exception:
        raise HTTPException(status_code=401, detail="Malformed authentication token")

    if not hmac.compare_digest(expected_sig, actual_sig):
        # Signed with another key (e.g. JWT_SECRET rotated): the session is over. The apps sign out
        # on this exact detail instead of retrying forever.
        raise HTTPException(status_code=401, detail=SESSION_REVOKED_DETAIL)

    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception:
        raise HTTPException(status_code=401, detail="Malformed authentication token")

    if verify_exp:
        exp = payload.get("exp")
        if exp and datetime.now(timezone.utc).timestamp() > float(exp):
            raise HTTPException(status_code=401, detail=TOKEN_EXPIRED_DETAIL)

    return payload


def issue_auth_tokens(user_id: str, email: str, is_admin: bool, token_version: int = 0) -> dict:
    """Issue an access + refresh JWT pair for a user. Shape matches what the mobile client expects.

    `token_version` must be the user's current UserDB.token_version, or the tokens are born revoked.
    """
    tv = int(token_version or 0)
    access_token = create_jwt_token(
        {"sub": user_id, "userId": user_id, "email": email, "is_admin": is_admin, "type": "access", "tv": tv},
        JWT_ACCESS_TTL,
    )
    refresh_token = create_jwt_token({"sub": user_id, "userId": user_id, "type": "refresh", "tv": tv}, JWT_REFRESH_TTL)
    return {"token": access_token, "refresh_token": refresh_token, "expires_in": int(JWT_ACCESS_TTL.total_seconds())}


# ---------------------------------------------------------------- the account behind a token

async def _account(uid: uuid.UUID) -> Optional[Tuple[int, bool]]:
    """(token_version, is_admin) of an existing account, None when it's gone."""
    factory = get_session_factory()
    if factory is None:
        raise HTTPException(status_code=503, detail="Database not available. Check DATABASE_URL environment variable.")
    async with factory() as session:
        row = (await session.execute(select(UserDB.token_version, UserDB.is_admin).where(UserDB.id == uid))).first()
    return (int(row[0] or 0), bool(row[1])) if row else None


async def assert_token_current(user_id, token_version) -> None:
    """Reject a token issued before the user's sessions were revoked (token_version bumped)."""
    try:
        uid = uuid.UUID(str(user_id))
    except ValueError:
        return
    account = await _account(uid)
    # No row = the account was deleted: its tokens are dead too.
    if account is None or int(token_version or 0) < account[0]:
        raise HTTPException(status_code=401, detail=SESSION_REVOKED_DETAIL)


async def assert_admin(user_id) -> None:
    """403 unless `user_id` is an existing admin account (the token's is_admin claim is never trusted)."""
    try:
        uid = uuid.UUID(str(user_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    account = await _account(uid)
    if not account or not account[1]:
        raise HTTPException(status_code=403, detail="Admin access required")


def bearer_token(authorization: Optional[str]) -> Optional[str]:
    """The token of an `Authorization: Bearer …` header, None without one."""
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    return authorization.split(" ", 1)[1].strip()


async def signed_in_user(authorization: Optional[str], verify_exp: bool = True,
                         missing_detail: str = "Sign in required") -> uuid.UUID:
    """The account of a valid, current access token. 401: no token (`missing_detail`), bad signature /
    revoked (SESSION_REVOKED_DETAIL), expired (TOKEN_EXPIRED_DETAIL, unless verify_exp=False), not an
    access token."""
    token = bearer_token(authorization)
    if token is None:
        raise HTTPException(status_code=401, detail=missing_detail)
    payload = decode_jwt_token(token, verify_exp=verify_exp)
    if payload.get("type") != "access" or not payload.get("sub"):
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    try:
        uid = uuid.UUID(str(payload["sub"]))
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    await assert_token_current(uid, payload.get("tv"))
    return uid


# ---------------------------------------------------------------- who is acting, on whose data

@dataclass(frozen=True)
class Acting:
    """A signed-in account, acting as a rep (its own data) or through the admin console (everyone's)."""
    user_id: uuid.UUID
    as_admin: bool = False

    @property
    def owner(self) -> Optional[uuid.UUID]:
        """Whose data the request works on: the rep's own, or None = every user's (admin console)."""
        return None if self.as_admin else self.user_id


async def acting(authorization: Optional[str], console: bool = False) -> Acting:
    """The /api identity. The mobile app always sends its token, so everything it reads or writes is
    scoped to that user. The admin console sends `X-Admin-Console: 1` (console=True) with an admin's
    token and gets the cross-user view; a non-admin's token is refused (403)."""
    uid = await signed_in_user(authorization)
    if console:
        await assert_admin(uid)
    return Acting(uid, as_admin=console)


async def admin_user(authorization: Optional[str]) -> uuid.UUID:
    """The signed-in admin (expiry enforced: the console refreshes its token). 401 / 403."""
    uid = await signed_in_user(authorization, missing_detail="Admin sign-in required")
    await assert_admin(uid)
    return uid


def _claimed_id(claimed) -> uuid.UUID:
    try:
        return uuid.UUID(str(claimed))
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid user_id")


async def legacy_user(authorization: Optional[str], claimed) -> uuid.UUID:
    """The account a legacy app route (`?user_id=` + API key) acts for. The login token decides: the API key
    ships inside the app, so a `user_id` alone proves nothing. No / unusable token → 401 (the app signs out on
    SESSION_REVOKED_DETAIL, refreshes on TOKEN_EXPIRED_DETAIL); someone else's id → 403, for admins too —
    the console works on other reps' data through /api with X-Admin-Console, never by impersonating a rep."""
    want = _claimed_id(claimed)
    if bearer_token(authorization) is None:
        raise HTTPException(status_code=401, detail="Sign in required")
    try:
        uid = await signed_in_user(authorization)
    except HTTPException as e:
        if e.status_code == 401 and e.detail in (SESSION_REVOKED_DETAIL, TOKEN_EXPIRED_DETAIL):
            raise
        # malformed / pre-JWT token: the session can't be trusted, so end it
        raise HTTPException(status_code=401, detail=SESSION_REVOKED_DETAIL)
    if uid != want:
        raise HTTPException(status_code=403, detail=OTHER_ACCOUNT_DETAIL)
    return want


async def check_presented_token(authorization: Optional[str]) -> None:
    """For API-key routes that don't act for an account: a token the app sends anyway must not be revoked
    (signed-out devices really stop) or expired (the app refreshes); a legacy / malformed one is ignored."""
    token = bearer_token(authorization)
    if token is None:
        return
    try:
        payload = decode_jwt_token(token)
    except HTTPException as err:
        if err.detail in (SESSION_REVOKED_DETAIL, TOKEN_EXPIRED_DETAIL):
            raise  # a real token we no longer accept (key rotated) / that needs a refresh
        return
    if payload.get("type") == "access" and payload.get("sub"):
        await assert_token_current(str(payload["sub"]), payload.get("tv"))


async def data_owner(me: Optional[uuid.UUID], explicit: Optional[str], session=None,
                     missing_status: int = 401, missing_detail: str = "Sign in required",
                     invalid_detail: str = "Invalid userId") -> uuid.UUID:
    """Whose rows a request works on: the signed-in rep's own (`me`), else — the admin console, me None —
    an explicit userId. With `session` the account must exist (404)."""
    if me is not None:
        return me
    if not explicit:
        raise HTTPException(status_code=missing_status, detail=missing_detail)
    try:
        owner = uuid.UUID(explicit)
    except ValueError:
        raise HTTPException(status_code=400, detail=invalid_detail)
    if session is not None and not (await session.execute(select(UserDB.id).where(UserDB.id == owner))).scalar_one_or_none():
        raise HTTPException(status_code=404, detail="User not found")
    return owner


def require_rep(me: Optional[uuid.UUID]) -> uuid.UUID:
    """Routes that only make sense for a signed-in rep (sync): the console (me None) gets 401."""
    if me is None:
        raise HTTPException(status_code=401, detail="Sign in required")
    return me


# ---------------------------------------------------------------- the console's live feed (SSE)

# One-use, 60-second tickets for the console's live feed: browsers' EventSource can't send headers,
# and an access token in the URL ends up in server / proxy logs.
STREAM_TICKET_SECONDS = 60
_stream_tickets: Dict[str, tuple] = {}


def issue_stream_ticket(admin_id: uuid.UUID) -> str:
    ticket = secrets.token_urlsafe(32)
    _stream_tickets[ticket] = (admin_id, datetime.now(timezone.utc) + timedelta(seconds=STREAM_TICKET_SECONDS))
    return ticket


async def stream_admin(authorization: Optional[str], ticket: Optional[str], access_token: Optional[str]) -> uuid.UUID:
    """The admin opening the live feed: a header, a `?ticket=` (issue_stream_ticket), or (older consoles)
    `?access_token=`."""
    if bearer_token(authorization) is None and ticket:
        now = datetime.now(timezone.utc)
        for k in [k for k, (_, exp) in _stream_tickets.items() if exp < now]:
            _stream_tickets.pop(k, None)
        found = _stream_tickets.pop(ticket, None)
        if not found:
            raise HTTPException(status_code=401, detail="Live feed ticket expired")
        return found[0]
    if bearer_token(authorization) is None and access_token:
        authorization = f"Bearer {access_token}"
    return await admin_user(authorization)
