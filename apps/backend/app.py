import os
from dotenv import load_dotenv
load_dotenv()
os.environ["USE_TF"] = "0"
import re
import json
import uuid
import asyncio
import qrcode
from qrcode.image.pil import PilImage
try:
    from pyzbar.pyzbar import decode as decode_qr
except ImportError:
    print("[WARNING] pyzbar not found. QR scanning will be disabled.")
    def decode_qr(image):
        return []
from PIL import Image
import base64
import hmac
import hashlib
import secrets
import smtplib
import ssl
from email.message import EmailMessage
import traceback
from urllib.parse import urlencode
import numpy as np
import faiss
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Header, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from html import escape as html_escape
from pydantic import BaseModel, EmailStr

from sentence_transformers import SentenceTransformer

import bcrypt

import ai_client
from ai_client import AIError
import score_levels
import identity
import event_identity
import contact_view
import contact_store
import scoring
from scoring import ScoringUnavailable, admin_ai_profile  # score_contact_with_ai raises it; pipeline / assistant use it
from event_identity import MAX_PROFILE_EVENTS, MAX_EVENT_NAME_LENGTH
from identity import decode_jwt_token, issue_auth_tokens, JWT_SECRET, SESSION_REVOKED_DETAIL  # the JWTs live in identity.py

from sqlalchemy import select, update, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from database import (
    UserDB, ContactDB, SharedContactDB, UserProfileDB, ExhibitorDB,
    ContactFileDB, ContactPipelineDB, PasswordResetDB, AdminContactScoreDB, Base,
    get_engine, get_session_factory, get_backup_session_factory, init_db, dispose_engines,
    ASYNC_DATABASE_URL, ASYNC_BACKUP_URL, IS_PRODUCTION
)

import httpx

# --- CONFIG ---
APP_API_KEY = os.environ.get("APP_API_KEY", "1234")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")  # n8n webhook endpoint
OPENROUTER_API_KEY = ai_client.OPENROUTER_API_KEY  # all AI goes through OpenRouter (ai_client.py)
# Emails allowed to become admin automatically — ONLY when the identity provider vouches for the email
# (LinkedIn email_verified). Password sign-ups are never auto-promoted: there is no email verification,
# so anyone could register a listed address first. Grant admin with `python manage_admins.py grant <email>`.
ADMIN_EMAILS = [e.strip().lower() for e in os.environ.get("ADMIN_EMAILS", "").split(",") if e.strip()]

# --- JWT CONFIG ---
# The HS256 secret (JWT_SECRET, else a generated JWT_SECRET_FILE) and token lifetimes are loaded by identity.py.

if IS_PRODUCTION:
    # Fail at boot rather than run with secrets anyone can guess or that vanish on the next redeploy.
    _missing = []
    if not os.environ.get("APP_API_KEY") or len(APP_API_KEY) < 16:
        _missing.append("APP_API_KEY (16+ characters)")
    if len(os.environ.get("JWT_SECRET", "").strip()) < 32:
        _missing.append("JWT_SECRET (32+ characters)")
    if _missing:
        raise RuntimeError("Production config incomplete — set: " + ", ".join(_missing))
    if not os.environ.get("PUBLIC_BASE_URL"):
        print("[CONFIG] WARNING: PUBLIC_BASE_URL is unset — share / card links use the request host.")
else:
    if not os.environ.get("APP_API_KEY") or len(APP_API_KEY) < 16:
        print("[CONFIG] WARNING: APP_API_KEY is unset or shorter than 16 characters — set a strong key before deploying.")
    if not os.environ.get("JWT_SECRET"):
        print("[CONFIG] WARNING: JWT_SECRET is unset — using the generated key file. Set JWT_SECRET in production "
              "(containers lose the file on redeploy, which signs everyone out).")

# Comma-separated list of allowed browser origins (admin console / web app). Unset = any origin
# (safe here: auth is header-based, no cookies).
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()] or ["*"]

# LinkedIn OAuth config
LINKEDIN_CLIENT_ID = os.environ.get("LINKEDIN_CLIENT_ID", "")
LINKEDIN_CLIENT_SECRET = os.environ.get("LINKEDIN_CLIENT_SECRET", "")
LINKEDIN_REDIRECT_URI = os.environ.get("LINKEDIN_REDIRECT_URI", "http://localhost:8081/callback")
LINKEDIN_ENABLED = bool(LINKEDIN_CLIENT_ID and LINKEDIN_CLIENT_SECRET)
# The installed app can't receive LinkedIn's https redirect itself: LinkedIn returns to this API's page
# (register it in the LinkedIn app too), which hands the code to the app via its eventscout:// scheme.
LINKEDIN_NATIVE_RETURN_URI = os.environ.get("LINKEDIN_NATIVE_REDIRECT_URI", "") or (
    f"{os.environ.get('PUBLIC_BASE_URL', '').rstrip('/')}/auth/linkedin/app-return" if os.environ.get("PUBLIC_BASE_URL") else "")
LINKEDIN_APP_CALLBACK = "eventscout://callback"


def linkedin_redirect_for(platform: Optional[str]) -> str:
    """The redirect_uri LinkedIn must see — the same one for the authorize URL and the code exchange."""
    if (platform or "").lower() == "native":
        if not LINKEDIN_NATIVE_RETURN_URI:
            raise HTTPException(status_code=501, detail="LinkedIn sign-in from the app needs PUBLIC_BASE_URL on the server")
        return LINKEDIN_NATIVE_RETURN_URI
    return LINKEDIN_REDIRECT_URI

# Outgoing email (password reset codes). Any SMTP provider works: SES, SendGrid, Resend, Gmail…
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("SMTP_FROM", "") or SMTP_USERNAME
SMTP_USE_SSL = os.environ.get("SMTP_USE_SSL", "false").lower() == "true"  # true = implicit TLS (port 465)
# A company mail server with a self-signed certificate: trust its certificate file (SMTP_CA_FILE, PEM),
# or as a last resort skip the check (SMTP_TLS_VERIFY=false). The connection stays encrypted either way.
SMTP_CA_FILE = os.environ.get("SMTP_CA_FILE", "").strip()
SMTP_TLS_VERIFY = os.environ.get("SMTP_TLS_VERIFY", "true").lower() != "false"
EMAIL_ENABLED = bool(SMTP_HOST and SMTP_FROM)
# Local development only: print reset codes to the server log instead of emailing them.
PASSWORD_RESET_DEV_LOG = os.environ.get("PASSWORD_RESET_DEV_LOG", "false").lower() == "true"

# Pipeline config
PRESENTON_API_URL = os.environ.get("PRESENTON_API_URL", "")  # Self-hosted Presenton instance (legacy)
PRESENTON_API_KEY = os.environ.get("PRESENTON_API_KEY", "")
GAMMA_API_KEY = os.environ.get("GAMMA_API_KEY", "")  # Gamma.app for AI deck generation
GAMMA_API_BASE = "https://public-api.gamma.app/v1.0"
GAMMA_THEME = "ash"  # Professional B2B theme
AUTO_PIPELINE_ENABLED = os.environ.get("AUTO_PIPELINE_ENABLED", "false").lower() == "true"

# Optional per-user chat model (profile.preferred_ai_model); unset = ai_client.CHAT_MODEL.
# Names match the model ids actually called (they used to say "Opus 4.6" / "GPT-5.2").
AVAILABLE_MODELS = {
    "claude-opus-4": {"id": "anthropic/claude-opus-4", "name": "Claude Opus 4", "provider": "Anthropic"},
    "gpt-4.1": {"id": "openai/gpt-4.1", "name": "GPT-4.1", "provider": "OpenAI"},
    "gemini-2.5-pro": {"id": "google/gemini-2.5-pro", "name": "Gemini 2.5 Pro", "provider": "Google"},
}
# keys saved in profiles before the keys matched the models ("gpt-5" was really GPT-4.1)
LEGACY_MODEL_KEYS = {"claude-opus": "claude-opus-4", "gpt-5": "gpt-4.1", "gemini-pro": "gemini-2.5-pro"}


def model_label(model_id: str) -> str:
    """Friendly name for the model that answered."""
    for m in AVAILABLE_MODELS.values():
        if m["id"] == model_id:
            return m["name"]
    return model_id

BASE_DIR = os.getcwd()
TEMP_DIR = os.path.join(BASE_DIR, "temp_images")
QR_DIR = os.path.join(BASE_DIR, "saved_qr")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(QR_DIR, exist_ok=True)

# --- AI provider: OpenRouter (ai_client.py) ---
ai_configured = ai_client.ai_configured
print(f"[INIT] AI via OpenRouter: {'configured' if ai_configured else 'NOT configured - set OPENROUTER_API_KEY'} (model {ai_client.DEFAULT_MODEL})")


# --- Pydantic models ---
class Contact(BaseModel):
    name: str
    email: str
    phone: str
    linkedin: str
    company_name: str = "N/A"
    job_title: Optional[str] = None  # read from the card / vCard TITLE when present
    notes: str = ""
    links: List[Dict[str, str]] = []
    event_name: Optional[str] = None


class SearchQuery(BaseModel):
    query: str
    user_id: str


class ConverseQuery(BaseModel):
    query: str
    user_id: str
    conversation_history: Optional[List[Dict[str, str]]] = None
    top_k: Optional[int] = 4
    admin_mode: Optional[bool] = False  # Cross-user search for admins
    ui_format: Optional[str] = None  # "openui": answer may include OpenUI Lang blocks (Ask AI renders them)


class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class LinkedInAuth(BaseModel):
    code: str
    state: Optional[str] = None
    platform: Optional[str] = None  # "native" = the installed app (see LINKEDIN_NATIVE_RETURN_URI)

class LinkedInTokenResponse(BaseModel):
    token: str
    refresh_token: str
    expires_in: int
    user_id: str
    email: Optional[str] = None
    name: Optional[str] = None


class AddContactRequest(BaseModel):
    full_name: Optional[str] = None
    job_title: Optional[str] = None
    company_name: Optional[str] = None
    company_website: Optional[str] = None
    linkedin_url: Optional[str] = None
    products: Optional[List[Dict[str, str]]] = None
    target_industries: Optional[List[str]] = None
    target_company_sizes: Optional[List[str]] = None
    target_roles: Optional[List[str]] = None
    target_geographies: Optional[List[str]] = None
    pitch_style: Optional[str] = None
    value_propositions: Optional[List[str]] = None
    common_objections: Optional[List[Dict[str, str]]] = None
    case_studies: Optional[List[Dict[str, str]]] = None
    preferred_follow_up_delay_days: Optional[int] = None
    auto_research_on_scan: Optional[bool] = None
    auto_score_on_add: Optional[bool] = None
    email_signature: Optional[str] = None
    current_event_name: Optional[str] = None
    current_event_description: Optional[str] = None
    event_goals: Optional[List[str]] = None
    preferred_ai_model: Optional[str] = None  # a key of AVAILABLE_MODELS (claude-opus-4, gpt-4.1, gemini-2.5-pro)
    # Your profile photo: a data URI / base64 image to set, "" to remove it.
    avatar: Optional[str] = None


class LeadScoreResult(BaseModel):
    contact_id: str
    score: int
    temperature: str
    reasoning: str
    recommended_actions: List[str]
    breakdown: Dict[str, int]


class EnrichRequest(BaseModel):
    notes_append: Optional[str] = None
    links: Optional[List[Dict[str, str]]] = None


class UserProfileUpdate(BaseModel):
    """Update user's personalization profile for event intelligence."""
    full_name: Optional[str] = None
    job_title: Optional[str] = None
    company_name: Optional[str] = None
    company_website: Optional[str] = None
    linkedin_url: Optional[str] = None
    products: Optional[List[Dict[str, str]]] = None
    target_industries: Optional[List[str]] = None
    target_company_sizes: Optional[List[str]] = None
    target_roles: Optional[List[str]] = None
    target_geographies: Optional[List[str]] = None
    pitch_style: Optional[str] = None
    value_propositions: Optional[List[str]] = None
    common_objections: Optional[List[Dict[str, str]]] = None
    case_studies: Optional[List[Dict[str, str]]] = None
    preferred_follow_up_delay_days: Optional[int] = None
    auto_research_on_scan: Optional[bool] = None
    auto_score_on_add: Optional[bool] = None
    email_signature: Optional[str] = None
    current_event_name: Optional[str] = None  # the ACTIVE event: new scans are tagged with it
    current_event_description: Optional[str] = None
    event_goals: Optional[List[str]] = None
    # Every event the user is scouting (picked from the imported list or typed in by hand).
    events: Optional[List[str]] = None
    # Per-event goals / notes: {event name: {"goals": [...], "description": "..."}}. The current
    # event's entry is mirrored into event_goals / current_event_description (what the AI reads).
    event_details: Optional[Dict[str, Dict[str, Any]]] = None
    # Events the user has ended (app → event page → End event). They stay in `events`, marked done.
    ended_events: Optional[List[str]] = None
    preferred_ai_model: Optional[str] = None  # a key of AVAILABLE_MODELS (claude-opus-4, gpt-4.1, gemini-2.5-pro)
    # Your profile photo: a data URI / base64 image to set, "" to remove it.
    avatar: Optional[str] = None
    phone: Optional[str] = None  # shown on your QR card


PITCH_STYLES = ("consultative", "direct", "challenger", "relationship")

# Admin vs rep AI — two separate worlds (scoring.py): the rep's own profile vs ONLY the admin sales profile
# (scoring.admin_ai_profile) for everything the admin console's AI does.


def normalize_profile_sales(update_data: dict) -> None:
    """Clean the "what you sell" fields in an incoming profile update, in place.

    These go straight into the AI chat / scoring / research / pitch prompts, so strings are
    trimmed, blanks dropped and sizes capped. Products need a name; pitch_style must be one
    of PITCH_STYLES ("" clears it).
    """
    def s(v, n):
        return " ".join(str(v or "").split())[:n]

    def text(v, n):  # keeps line breaks (descriptions)
        return "\n".join(" ".join(line.split()) for line in str(v or "").strip().splitlines()).strip()[:n]

    def str_list(key, max_items, max_len):
        if update_data.get(key) is None:
            return
        out, seen = [], set()
        for raw in update_data[key]:
            v = s(raw, max_len)
            if v and v.lower() not in seen:
                seen.add(v.lower())
                out.append(v)
        update_data[key] = out[:max_items]

    def pair_list(key, fields, max_items, max_len):
        if update_data.get(key) is None:
            return
        out = []
        for raw in update_data[key]:
            item = {f: text((raw or {}).get(f), max_len) for f in fields}
            if item[fields[0]]:
                out.append(item)
        update_data[key] = out[:max_items]

    if update_data.get("products") is not None:
        products = []
        for raw in update_data["products"]:
            p = {"name": s((raw or {}).get("name"), 120),
                 "description": text((raw or {}).get("description"), 600),
                 "ideal_customer": text((raw or {}).get("ideal_customer"), 300)}
            if p["name"]:
                products.append(p)
        if len(products) > 12:
            raise HTTPException(status_code=400, detail="You can add up to 12 products or services")
        update_data["products"] = products
    str_list("value_propositions", 10, 200)
    str_list("event_goals", 10, 200)
    str_list("target_company_sizes", 6, 40)
    str_list("target_geographies", 20, 60)
    pair_list("common_objections", ("objection", "response"), 10, 400)
    pair_list("case_studies", ("title", "result"), 10, 400)
    if update_data.get("current_event_description") is not None:
        update_data["current_event_description"] = text(update_data["current_event_description"], 1000)
    if update_data.get("full_name") is not None:
        name = s(update_data["full_name"], 120)
        if not name:
            raise HTTPException(status_code=400, detail="Your name can't be empty")
        update_data["full_name"] = name
    if update_data.get("event_details") is not None:
        raw = update_data["event_details"]
        if not isinstance(raw, dict):
            raise HTTPException(status_code=400, detail="event_details must be an object keyed by event name")
        details = {}
        for key, val in raw.items():
            name = s(key, MAX_EVENT_NAME_LENGTH)
            if not name or not isinstance(val, dict):
                continue
            goals, seen = [], set()
            for g in val.get("goals") or []:
                g = s(g, 200)
                if g and g.lower() not in seen:
                    seen.add(g.lower())
                    goals.append(g)
            details[name] = {"goals": goals[:10], "description": text(val.get("description"), 1000)}
            if len(details) >= MAX_PROFILE_EVENTS:
                break
        update_data["event_details"] = details
    if update_data.get("pitch_style") is not None:
        style = str(update_data["pitch_style"]).strip().lower()
        if style and style not in PITCH_STYLES:
            raise HTTPException(status_code=400, detail="Pitch style must be consultative, direct, challenger or relationship")
        update_data["pitch_style"] = style


# --- AUTH ---
# Who is acting (tokens, revocation, admin checks, the legacy ?user_id= rule) lives in identity.py; these are
# the FastAPI dependencies the app.py routes use, as thin shims over it.

async def verify_api_key(request: Request, x_api_key: str = Header(None), authorization: Optional[str] = Header(None)):
    """The legacy app routes: the shared API key + (when the route acts for an account, `?user_id=`) that
    account's own login token. The key ships inside the app, so it only proves "an EventScout client"."""
    if x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    claimed = request.query_params.get("user_id")
    if claimed:
        await identity.legacy_user(authorization, claimed)
    else:
        await identity.check_presented_token(authorization)
    return x_api_key


# --- PASSWORD HASHING ---
def hash_password(password: str) -> str:
    password_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        password_bytes = plain_password.encode('utf-8')[:72]
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except ValueError:
        return False


async def get_current_user_id(authorization: Optional[str] = Header(None)) -> str:
    """Resolve the signed-in user from an `Authorization: Bearer <access JWT>` header."""
    return str(await identity.signed_in_user(authorization))


# --- LINKEDIN OAUTH HELPERS ---
async def get_linkedin_auth_url(state: Optional[str] = None, redirect_uri: Optional[str] = None) -> str:
    """Generate a LinkedIn 'Sign In with LinkedIn using OpenID Connect' authorization URL."""
    if not LINKEDIN_ENABLED:
        raise HTTPException(status_code=501, detail="LinkedIn OAuth not configured")

    state = state or str(uuid.uuid4())
    params = {
        'response_type': 'code',
        'client_id': LINKEDIN_CLIENT_ID,
        'redirect_uri': redirect_uri or LINKEDIN_REDIRECT_URI,
        'state': state,
        # OpenID Connect scopes (the legacy r_liteprofile/r_emailaddress products are deprecated)
        'scope': 'openid profile email',
    }
    # urlencode so spaces in `scope` and reserved chars in redirect_uri are escaped correctly.
    return f"https://www.linkedin.com/oauth/v2/authorization?{urlencode(params)}"


async def exchange_linkedin_code(code: str, redirect_uri: Optional[str] = None) -> dict:
    """Exchange a LinkedIn authorization code for the user's profile via OpenID Connect."""
    if not LINKEDIN_ENABLED:
        raise HTTPException(status_code=501, detail="LinkedIn OAuth not configured")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Exchange the authorization code for an access token + id_token
            token_response = await client.post(
                'https://www.linkedin.com/oauth/v2/accessToken',
                data={
                    'grant_type': 'authorization_code',
                    'code': code,
                    'redirect_uri': redirect_uri or LINKEDIN_REDIRECT_URI,
                    'client_id': LINKEDIN_CLIENT_ID,
                    'client_secret': LINKEDIN_CLIENT_SECRET,
                },
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
            )

            token_data = token_response.json()
            if 'error' in token_data:
                detail = token_data.get('error_description') or token_data['error']
                raise HTTPException(status_code=400, detail=f"LinkedIn token error: {detail}")

            access_token = token_data.get('access_token')
            if not access_token:
                raise HTTPException(status_code=400, detail="LinkedIn did not return an access token")

            # OpenID Connect userinfo endpoint returns sub/name/given_name/family_name/email/picture
            userinfo_response = await client.get(
                'https://api.linkedin.com/v2/userinfo',
                headers={'Authorization': f'Bearer {access_token}'},
            )
            userinfo = userinfo_response.json()

            name = userinfo.get('name') or " ".join(
                filter(None, [userinfo.get('given_name'), userinfo.get('family_name')])
            ).strip()

            return {
                'access_token': access_token,
                'expires_in': token_data.get('expires_in', 3600),
                'profile': userinfo,
                'email': userinfo.get('email'),
                'name': name,
                'picture': userinfo.get('picture'),
            }
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"LinkedIn API error: {str(e)}")


# --- DATABASE SESSION HELPER ---
async def get_db_session() -> AsyncSession:
    factory = get_session_factory()
    if factory is None:
        raise HTTPException(
            status_code=503, 
            detail="Database not available. Check DATABASE_URL environment variable."
        )
    try:
        async with factory() as session:
            return session
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection failed: {str(e)}"
        )


# --- EMBEDDING MODEL ---
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def get_embedding(text: str) -> np.ndarray:
    return embedding_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]


# --- IN-MEMORY FAISS INDEX (rebuilt from Postgres on startup) ---
def _contact_summary(c: dict) -> str:
    summary = f"{c.get('name','')}, {c.get('email','')}, {c.get('phone','')}, {c.get('linkedin','')}, {c.get('company_name','')}"
    if c.get('notes'):
        summary += f", {c['notes'][:200]}"
    return summary


def _embed_many(texts: List[str]) -> np.ndarray:
    return embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")


class FAISSIndex:
    """In-memory FAISS index for semantic search. Rebuilt from Postgres on startup.

    Each user's vectors are cached, so adding / editing / removing one contact embeds only that
    contact (it used to re-embed every contact the user had on each change)."""

    def __init__(self):
        self.indices: Dict[str, faiss.IndexFlatIP] = {}  # user_id -> FAISS index
        self.texts: Dict[str, List[str]] = {}  # user_id -> text summaries
        self.metadata: Dict[str, List[dict]] = {}  # user_id -> contact metadata
        self.vectors: Dict[str, Optional[np.ndarray]] = {}  # user_id -> (n, dim) embeddings, same order
        print("[FAISS] In-memory index manager initialized")

    def _reindex(self, user_id: str):
        vecs = self.vectors.get(user_id)
        if vecs is None or len(vecs) == 0:
            self.indices[user_id] = None
            return
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        self.indices[user_id] = index

    def build_for_user(self, user_id: str, contacts: List[dict]):
        """Build FAISS index for a user from a list of contact dicts."""
        if not contacts:
            self.indices[user_id] = None
            self.texts[user_id] = []
            self.metadata[user_id] = []
            self.vectors[user_id] = None
            return

        texts = [_contact_summary(c) for c in contacts]
        self.texts[user_id] = texts
        self.metadata[user_id] = list(contacts)
        self.vectors[user_id] = _embed_many(texts)
        self._reindex(user_id)
        print(f"[FAISS] Built index for user {user_id[:8]}... with {len(contacts)} contacts")

    def add_contact(self, user_id: str, contact_dict: dict):
        """Add a single contact to user's index."""
        summary = _contact_summary(contact_dict)
        if user_id not in self.texts:
            self.texts[user_id] = []
            self.metadata[user_id] = []
            self.vectors[user_id] = None
        vec = _embed_many([summary])
        self.texts[user_id].append(summary)
        self.metadata[user_id].append(contact_dict)
        prev = self.vectors.get(user_id)
        self.vectors[user_id] = vec if prev is None or len(prev) == 0 else np.vstack([prev, vec])
        self._reindex(user_id)

    def update_contact(self, user_id: str, contact_id: str, contact_dict: dict):
        """Update a contact in user's index."""
        if user_id not in self.metadata:
            return False
        for i, m in enumerate(self.metadata[user_id]):
            if m.get("id") == contact_id:
                summary = _contact_summary(contact_dict)
                self.texts[user_id][i] = summary
                self.metadata[user_id][i] = contact_dict
                vecs = self.vectors.get(user_id)
                if vecs is None or len(vecs) != len(self.texts[user_id]):
                    self.vectors[user_id] = _embed_many(self.texts[user_id])
                else:
                    vecs[i] = _embed_many([summary])[0]
                self._reindex(user_id)
                return True
        return False

    def delete_contact(self, user_id: str, contact_id: str):
        """Delete a contact from user's index."""
        if user_id not in self.metadata:
            return False
        for i, m in enumerate(self.metadata[user_id]):
            if m.get("id") == contact_id:
                self.texts[user_id].pop(i)
                self.metadata[user_id].pop(i)
                vecs = self.vectors.get(user_id)
                if self.texts[user_id] and vecs is not None and len(vecs) == len(self.texts[user_id]) + 1:
                    self.vectors[user_id] = np.delete(vecs, i, axis=0)
                elif self.texts[user_id]:
                    self.vectors[user_id] = _embed_many(self.texts[user_id])
                else:
                    self.vectors[user_id] = None
                self._reindex(user_id)
                return True
        return False

    def delete_user(self, user_id: str):
        """Remove all FAISS data for a user."""
        self.indices.pop(user_id, None)
        self.texts.pop(user_id, None)
        self.metadata.pop(user_id, None)
        self.vectors.pop(user_id, None)

    def search_scored(self, user_id: str, query: str, k: int = 4, q_vec: Optional[np.ndarray] = None) -> List[tuple]:
        """Like search(), with the similarity: [(score, text, meta)], best first."""
        index = self.indices.get(user_id)
        if index is None or index.ntotal == 0:
            return []
        texts = self.texts.get(user_id, [])
        metas = self.metadata.get(user_id, [])
        if q_vec is None:
            q_vec = get_embedding(query).astype("float32").reshape(1, -1)
        D, I = index.search(q_vec, min(k, len(texts)))
        return [(float(d), texts[idx], metas[idx]) for d, idx in zip(D[0], I[0]) if 0 <= idx < len(texts)]

    def search(self, user_id: str, query: str, k: int = 4) -> List[tuple]:
        """Search user's contacts semantically."""
        return [(text, meta) for _, text, meta in self.search_scored(user_id, query, k)]


faiss_index = FAISSIndex()


# --- INTELLIGENCE ENGINE (Multi-Provider AI Chat) ---
class IntelligenceEngine:
    """Chat assistant. Every answer comes from OpenRouter (ai_client): the user's preferred
    model if they picked one, else OPENROUTER_CHAT_MODEL, then the fallback models."""

    def _build_system_prompt(self, user_profile: dict = None, exhibitors: list = None) -> str:
        base_prompt = """You are Event Scout AI — a strategic event intelligence partner. You don't just retrieve contacts, you help the user WIN at this event. You think like a sales strategist, relationship builder, and event tactician.

YOUR CAPABILITIES:
1. CONTACT INTELLIGENCE: Analyze who the user has met and provide strategic context
2. PITCH GENERATION: Create tailored pitch angles using the user's products/value props
3. LEAD PRIORITIZATION: Rank contacts by strategic value, not just score numbers
4. RELATIONSHIP MAPPING: Identify connections between contacts (same company, industry, role)
5. ACTION PLANNING: Give specific next-step advice (e.g. "Visit booth X to meet Y because Z")
6. BRIEFINGS: Prepare the user before meetings with key talking points
7. EXHIBITOR INTELLIGENCE: Recommend which exhibitors to visit based on the user's goals

RESPONSE STYLE:
- Be strategic and direct, like a trusted advisor at the event
- Lead with the most important insight
- Use the user's own language (their products, value props, pitch style)
- Give SPECIFIC actions, not vague advice
- Reference actual contacts and data, never make things up
- When asked "who should I prioritize?", give a ranked list with clear reasoning tied to their goals
- When asked about pitch angles, tailor to the specific contact's company/role/industry
- Format responses with clear headers and bullet points for easy mobile reading
- Keep responses concise but actionable — the user is at an event and needs quick answers
- MIRROR the user's communication style. If they're direct, be direct. If they're casual, be casual.
- Be a PARTNER, not just an assistant — anticipate needs, connect dots between contacts, think ahead
- Write like you're texting a trusted colleague at the event, not writing a formal report
- When the user has a LinkedIn profile or company website, weave those insights naturally into advice
- Always frame advice in terms of the user's specific products, value props, and goals — make it feel like you truly understand their business

Important: Only use information from the provided contact/exhibitor context. Do not make up contact details or exhibitor info."""

        if user_profile and any(user_profile.values()):
            profile_context = "\n\nUSER CONTEXT (use this to deeply personalize ALL advice):"
            if user_profile.get("full_name"):
                profile_context += f"\n- User: {user_profile['full_name']}"
            if user_profile.get("job_title"):
                profile_context += f"\n- Role: {user_profile['job_title']}"
            if user_profile.get("company_name"):
                profile_context += f"\n- Company: {user_profile['company_name']}"
            if user_profile.get("products"):
                for i, p in enumerate(user_profile["products"], 1):
                    profile_context += f"\n- Product {i}: {p.get('name', '')} — {p.get('description', '')}"
                    if p.get("ideal_customer"):
                        profile_context += f" (Ideal customer: {p['ideal_customer']})"
            if user_profile.get("target_industries"):
                profile_context += f"\n- Target Industries: {', '.join(user_profile['target_industries'])}"
            if user_profile.get("target_roles"):
                profile_context += f"\n- Target Roles: {', '.join(user_profile['target_roles'])}"
            if user_profile.get("target_company_sizes"):
                profile_context += f"\n- Target Company Sizes: {', '.join(user_profile['target_company_sizes'])}"
            if user_profile.get("value_propositions"):
                profile_context += f"\n- Value Propositions:"
                for vp in user_profile["value_propositions"]:
                    profile_context += f"\n  * {vp}"
            if user_profile.get("pitch_style"):
                styles = {
                    "consultative": "Consultative (discover needs first, ask questions, position as partner)",
                    "direct": "Direct (lead with solution and ROI, get to the point fast)",
                    "challenger": "Challenger (reframe their thinking, share insights they haven't considered)",
                    "relationship": "Relationship (build trust first, find common ground, long-term play)",
                }
                profile_context += f"\n- Pitch Style: {styles.get(user_profile['pitch_style'], user_profile['pitch_style'])}"
            if user_profile.get("current_event_name"):
                profile_context += f"\n- Current Event: {user_profile['current_event_name']}"
            if user_profile.get("current_event_description"):
                profile_context += f"\n- Event Details: {user_profile['current_event_description']}"
            if user_profile.get("event_goals"):
                profile_context += f"\n- Event Goals:"
                for goal in user_profile["event_goals"]:
                    profile_context += f"\n  * {goal}"
            if user_profile.get("linkedin_url"):
                profile_context += f"\n- LinkedIn: {user_profile['linkedin_url']}"
            if user_profile.get("company_website"):
                profile_context += f"\n- Company Website: {user_profile['company_website']}"
            if user_profile.get("target_geographies"):
                profile_context += f"\n- Target Geographies: {', '.join(user_profile['target_geographies'])}"
            other_events = [e for e in (user_profile.get("events") or []) if e != user_profile.get("current_event_name")]
            if other_events:
                profile_context += f"\n- Other events they're scouting: {', '.join(other_events)}"

            profile_context += "\n\nCRITICAL: Use ALL of the above context when giving advice. Every pitch suggestion must reference the user's actual products and value props. Every prioritization must consider their target market. Every briefing must connect to their event goals. Speak as if you truly know and understand this person — their business, their style, their goals."
            base_prompt += profile_context

        if exhibitors:
            exhibitor_context = f"\n\nEXHIBITOR DATA ({len(exhibitors)} exhibitors at the event):"
            for ex in exhibitors[:30]:  # Limit to 30 to keep prompt manageable
                ex_line = f"\n- {ex['name']}"
                if ex.get('booth'):
                    ex_line += f" (Booth: {ex['booth']}"
                    if ex.get('hall'):
                        ex_line += f", Hall: {ex['hall']}"
                    ex_line += ")"
                if ex.get('category'):
                    ex_line += f" — {ex['category']}"
                if ex.get('country'):
                    ex_line += f" [{ex['country']}]"
                if ex.get('description'):
                    ex_line += f" | {ex['description'][:200]}"
                exhibitor_context += ex_line
            base_prompt += exhibitor_context

        return base_prompt

    def _build_context_from_contacts(self, contacts: List[tuple], all_contacts: list = None) -> str:
        if not contacts and not all_contacts:
            return "No contacts found in the database matching your query."

        context_parts = []

        if contacts:
            context_parts.append(f"RELEVANT CONTACTS (semantic match for your query, {len(contacts)} results):\n")
            for i, (text, meta) in enumerate(contacts, 1):
                contact_info = f"Contact {i}: {meta.get('name', 'N/A')}"
                contact_info += f" | Company: {meta.get('company_name', 'N/A')}"
                contact_info += f" | Email: {meta.get('email', 'N/A')}"
                contact_info += f" | Phone: {meta.get('phone', 'N/A')}"
                if meta.get('linkedin') and meta['linkedin'] != 'N/A':
                    contact_info += f" | LinkedIn: {meta['linkedin']}"
                if meta.get('notes'):
                    contact_info += f" | Notes: {meta['notes'][:300]}"
                if meta.get('lead_score') is not None:
                    contact_info += f" | Score: {meta['lead_score']}/100 ({meta.get('lead_temperature', 'unscored')})"
                if meta.get('lead_score_reasoning'):
                    contact_info += f" | Reasoning: {meta['lead_score_reasoning']}"
                if meta.get('lead_recommended_actions'):
                    actions = meta['lead_recommended_actions']
                    if isinstance(actions, list) and actions:
                        contact_info += f" | Actions: {'; '.join(actions[:3])}"
                context_parts.append(contact_info)

        if all_contacts and len(all_contacts) > len(contacts or []):
            context_parts.append(f"\nALL CONTACTS SUMMARY ({len(all_contacts)} total):")
            for c in all_contacts:
                summary = f"- {c.get('name', 'N/A')} at {c.get('company_name', 'N/A')}"
                if c.get('lead_score') is not None:
                    summary += f" (Score: {c['lead_score']}, {c.get('lead_temperature', '?')})"
                context_parts.append(summary)

        return "\n".join(context_parts)

    def _generate_fallback_response(self, query: str, retrieved_contacts: List[tuple]) -> str:
        if not retrieved_contacts:
            return f"No contacts found matching your query: '{query}'. Try adding some contacts first."
        response_parts = [f"Found {len(retrieved_contacts)} contact(s):\n"]
        for i, (text, meta) in enumerate(retrieved_contacts, 1):
            response_parts.append(f"**{i}. {meta.get('name', 'N/A')}** — {meta.get('company_name', 'N/A')}\n"
                                  f"- Email: {meta.get('email', 'N/A')} | Phone: {meta.get('phone', 'N/A')}")
        return "\n".join(response_parts)

    def _build_request(self, query: str, retrieved_contacts: List[tuple],
                       conversation_history: list = None, user_profile: dict = None,
                       all_contacts: list = None, exhibitors: list = None,
                       attachments: list = None, ui_format: Optional[str] = None) -> dict:
        """The ai_client.chat / chat_stream arguments for one answer.
        Attachments: images and PDFs go as native parts, text files are inlined."""
        system_prompt = self._build_system_prompt(user_profile, exhibitors)
        if ui_format == "openui" and OPENUI_PROMPT:
            system_prompt += "\n\n=== RICH ANSWERS (OpenUI) ===\n" + OPENUI_PROMPT + OPENUI_REQUIRED
        contact_context = self._build_context_from_contacts(retrieved_contacts, all_contacts)

        pref = (user_profile or {}).get("preferred_ai_model")
        pref = LEGACY_MODEL_KEYS.get(pref, pref)
        model_id = AVAILABLE_MODELS[pref]["id"] if pref in AVAILABLE_MODELS else ai_client.CHAT_MODEL

        messages: list = [{"role": "system", "content": system_prompt}]
        for msg in conversation_history or []:
            content = msg.get("content", "")
            if content:
                messages.append({"role": "assistant" if msg.get("role") == "assistant" else "user", "content": content})

        user_text = f"{contact_context}\n\n---\nUser Query: {query}" if contact_context else query
        if attachments:
            parts = [ai_client.text_part(user_text)]
            for att in attachments:
                if att.get("text") is not None:
                    parts.append(ai_client.text_part(f"\n\nATTACHED FILE \"{att['name']}\":\n{att['text']}"))
                elif str(att.get("mime_type", "")).startswith("image/"):
                    parts.append(ai_client.image_part(att["data"], att["mime_type"]))
                else:
                    parts.append(ai_client.file_part(att.get("name") or "file.pdf", att["data"], att.get("mime_type") or "application/pdf"))
            messages.append({"role": "user", "content": parts})
        else:
            messages.append({"role": "user", "content": user_text})

        return {
            "messages": messages, "model": model_id,
            "max_tokens": 2500 if ui_format == "openui" else 2000,
            "temperature": 0.5 if ui_format == "openui" else 0.7,
            "timeout": 75, "title": "Event Scout Chat",
        }

    async def generate_response(self, **kwargs) -> tuple:
        """Answer via OpenRouter. Returns (text, model_label) — the model that actually answered.
        Raises AIUnavailable when no model could answer."""
        try:
            text, used = await ai_client.chat(**self._build_request(**kwargs))
        except AIError as e:
            print(f"[AI] Chat failed: {e}")
            raise AIUnavailable(str(e), e.reason)
        return text, model_label(used)

    async def stream_response(self, **kwargs):
        """Like generate_response, streamed: yields ("model", model_label) then ("delta", text) pieces.
        Raises AIUnavailable when no model could answer (or the answering model failed mid-answer)."""
        try:
            async for kind, value in ai_client.chat_stream(**self._build_request(**kwargs)):
                yield kind, (model_label(value) if kind == "model" else value)
        except AIError as e:
            print(f"[AI] Chat stream failed: {e}")
            raise AIUnavailable(str(e), e.reason)


# OpenUI component prompt for Ask AI rich answers. Generated from the mobile app's component set
# (apps/mobile/src/features/ask/openui/spec.ts) by `node scripts/gen-openui-prompt.mjs`.
try:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "openui_prompt.txt"), encoding="utf-8") as _f:
        OPENUI_PROMPT = _f.read().strip()
except OSError:
    OPENUI_PROMPT = ""
    print("[INIT] openui_prompt.txt missing - Ask AI rich answers off (plain text only)")

# Without this the model mostly follows RESPONSE STYLE ("headers and bullet points") and answers in
# plain Markdown, so the app has nothing rich to render. Appended after the OpenUI prompt so it wins.
OPENUI_REQUIRED = """

=== WHEN TO USE A UI BLOCK (this overrides the Markdown formatting in RESPONSE STYLE) ===
You MUST include exactly one ```openui-lang block whenever the answer contains any of:
- a message for the user to send (email, LinkedIn note, text, opening lines) -> Draft (one Draft per message)
- a plan, a checklist, steps or a ranked to-do list -> Steps
- specific contacts -> ContactCard / ContactList
- counts, scores or other numbers from the data -> Stats or BarChart
- exhibitors or booths to visit -> Exhibitor
- a key warning or tip -> Callout
In those answers do NOT write the same content as Markdown bullets or headers: keep the text around the block
to one or two short sentences and put the structure in the block.
Plain text only (no block) for greetings, small talk and one-line factual answers."""

intelligence_engine = IntelligenceEngine()


# --- GENERATIVE UI COMPONENT GENERATOR ---
def generate_ui_components(query: str, response_text: str, retrieved_contacts: list,
                           all_contacts: list, exhibitors: list, user_profile: dict) -> list:
    """Analyze query/response context and generate rich UI components for the frontend."""
    components = []
    query_lower = query.lower()

    # 1. Contact Cards -- show when contacts are retrieved or mentioned
    if retrieved_contacts:
        components.append({
            "type": "contact_cards",
            "data": retrieved_contacts[:4]
        })

    # 2. Exhibitor Cards -- show when query mentions exhibitors/booths/visit
    exhibitor_keywords = ['exhibitor', 'booth', 'visit', 'hall', 'vendor', 'supplier', 'who should i see']
    if any(kw in query_lower for kw in exhibitor_keywords) and exhibitors:
        resp_lower = response_text.lower()
        matched = [e for e in exhibitors if e.get('name', '').lower() in resp_lower][:4]
        if not matched:
            matched = exhibitors[:4]
        components.append({"type": "exhibitor_cards", "data": matched})

    # 3. Score Summary -- when discussing scores or leads
    score_keywords = ['score', 'lead', 'hot', 'warm', 'cold', 'rating', 'qualify', 'pipeline', 'best contact']
    if any(kw in query_lower for kw in score_keywords):
        scored = [c for c in all_contacts if c.get('lead_score') is not None]
        if scored:
            components.append({
                "type": "score_summary",
                "data": {
                    "total": len(all_contacts),
                    "scored": len(scored),
                    "hot": len([c for c in scored if c.get('lead_temperature') == 'hot']),
                    "warm": len([c for c in scored if c.get('lead_temperature') == 'warm']),
                    "cold": len([c for c in scored if c.get('lead_temperature') == 'cold']),
                    "top_contacts": sorted(scored, key=lambda x: x.get('lead_score', 0), reverse=True)[:3]
                }
            })

    # 4. Action Buttons -- contextual actions based on retrieved contacts
    actions = []
    if retrieved_contacts:
        first = retrieved_contacts[0]
        cid = first.get('id', '')
        name = first.get('name', '')
        if not first.get('lead_score'):
            actions.append({"label": f"Score {name}", "action": "score", "contact_id": cid, "icon": "fa-chart-bar"})
        actions.append({"label": f"Research {name}", "action": "research", "contact_id": cid, "icon": "fa-search"})
        actions.append({"label": f"Pitch for {name}", "action": "pitch", "contact_id": cid, "icon": "fa-file-powerpoint"})
    if actions:
        components.append({"type": "action_buttons", "data": actions})

    # 5. Quick Replies -- suggested follow-up questions
    quick_replies = []
    if retrieved_contacts:
        name = retrieved_contacts[0].get('name', '')
        company = retrieved_contacts[0].get('company_name', '')
        quick_replies.append(f"Research {company}")
        quick_replies.append(f"Generate pitch for {name}")
        if not retrieved_contacts[0].get('lead_score'):
            quick_replies.append(f"Score {name} as a lead")
        quick_replies.append("Who should I visit next?")
    else:
        quick_replies = ["Show my hot leads", "Who should I visit next?", "Summarize my contacts", "Which exhibitors should I visit?"]
    components.append({"type": "quick_replies", "data": quick_replies[:4]})

    return components


def detect_intent(query: str) -> str:
    """Detect special intents in user queries for enhanced processing."""
    q = query.lower().strip()
    research_keywords = ['research', 'look up', 'find out about', 'investigate', 'tell me about', 'what do you know about', 'company info']
    pitch_keywords = ['pitch', 'pitch deck', 'presentation', 'slides', 'proposal', 'generate pitch', 'create pitch']

    if any(kw in q for kw in pitch_keywords):
        return 'pitch'
    if any(kw in q for kw in research_keywords):
        return 'research'
    return 'chat'


RESEARCH_PROMPT_SUPPLEMENT = """
IMPORTANT: The user wants you to RESEARCH this company/person in depth. Provide a comprehensive analysis with these sections:

## Company Overview
What they do, their size, market position, key products/services

## Key Insights
Recent developments, competitive positioning, growth signals, challenges

## Role-Specific Analysis
What challenges someone in this contact's role typically faces, their likely priorities and KPIs

## Strategic Fit
How this contact/company aligns with the user's products and services. Identify specific pain points the user can address.

## Recommended Approach
Best pitch angle, talking points, potential objections to prepare for, and ideal next steps.

Be specific and actionable. This research will be used to prepare for a real sales conversation.
"""

PITCH_PROMPT_SUPPLEMENT = """
IMPORTANT: The user wants you to generate a PITCH DECK for this contact. Create a compelling, personalized pitch.

Return your response as regular text with clear slide headers. Structure it as an 8-slide pitch:

## Slide 1: Opening
Personalized opening addressing the contact by name and their company. Hook them with a relevant insight about their industry.

## Slide 2: The Challenge
Their role-specific pain points backed by industry context. Show you understand their world.

## Slide 3: By The Numbers
Relevant industry data, market trends, competitor moves that create urgency.

## Slide 4: Cost of Inaction
What happens if they don't address these challenges. Make it concrete with examples.

## Slide 5: Our Approach
The user's methodology and philosophy. What makes their approach different.

## Slide 6: The Solution
Specific product/service mapping to the contact's pain points. Be concrete about value.

## Slide 7: Proof Points
Most relevant case studies, testimonials, or results from the user's profile.

## Slide 8: Next Steps
Clear CTA with a proposed meeting, demo, or follow-up action.

Make each slide compelling, concise, and personalized to this specific contact and their company.
"""


# --- QR GENERATOR ---
def create_qr(contact: Contact, contact_id: str = None):
    if contact_id is None:
        contact_id = str(uuid.uuid4())
    vcard = f"""BEGIN:VCARD
VERSION:3.0
N:{contact.name}
FN:{contact.name}
ORG:{contact.company_name}
TEL;TYPE=WORK,VOICE:{contact.phone}
EMAIL;TYPE=PREF,INTERNET:{contact.email}
URL:{contact.linkedin}
END:VCARD"""
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L)
    qr.add_data(vcard)
    qr.make(fit=True)
    img = qr.make_image(image_factory=PilImage, fill_color="black", back_color="white")
    qr_path = os.path.join(QR_DIR, f"qr_{contact_id}.png")
    img.save(qr_path)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    qr_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {"qr_path": qr_path, "qr_base64": qr_base64, "contact_id": contact_id}


# --- QR SCANNER ---
def parse_vcard(vcard_text: str) -> dict:
    fields = {"name": "N/A", "email": "N/A", "phone": "N/A", "linkedin": "N/A", "company_name": "N/A", "job_title": "N/A"}
    fn_match = re.search(r"FN[;:]([^\r\n]+)", vcard_text)
    if fn_match:
        fields["name"] = fn_match.group(1).strip()
    else:
        n_match = re.search(r"^N[;:]([^\r\n]+)", vcard_text, re.MULTILINE)
        if n_match:
            fields["name"] = n_match.group(1).strip()
    email_match = re.search(r"EMAIL[^:]*:([^\r\n]+)", vcard_text)
    if email_match:
        fields["email"] = email_match.group(1).strip()
    tel_match = re.search(r"TEL[^:]*:([^\r\n]+)", vcard_text)
    if tel_match:
        fields["phone"] = tel_match.group(1).strip()
    org_match = re.search(r"ORG[;:]([^\r\n]+)", vcard_text)
    if org_match:
        fields["company_name"] = org_match.group(1).strip()
    url_match = re.search(r"URL[^:]*:([^\r\n]+)", vcard_text)
    if url_match:
        fields["linkedin"] = url_match.group(1).strip()
    title_match = re.search(r"^TITLE[;:]([^\r\n]+)", vcard_text, re.MULTILINE)
    if title_match:
        fields["job_title"] = title_match.group(1).strip()
    return fields


SCAN_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".gif"}


def scan_temp_path(upload: UploadFile) -> str:
    """A unique temp path for an uploaded scan. Never uses the client's filename (two uploads
    named "card.jpg" collided, and a name like "../x" could escape TEMP_DIR) — only a
    whitelisted extension is kept."""
    ext = os.path.splitext(upload.filename or "")[1].lower()
    if ext not in SCAN_IMAGE_EXTENSIONS:
        ext = ".jpg"
    return os.path.join(TEMP_DIR, f"scan_{uuid.uuid4().hex}{ext}")


def remove_temp_file(path: str) -> None:
    """Best-effort cleanup: a lingering handle (e.g. an OCR thread that outlived its
    timeout) must not turn a finished scan into a 500."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as e:
        print(f"[SCAN] Could not remove temp file {path}: {e}")


def load_image_copy(image_path: str) -> Image.Image:
    """Read an image fully into memory and release the file handle immediately."""
    with Image.open(image_path) as src:
        src.load()
        return src.copy()


def scan_qr_image(image_path: str) -> str:
    img = load_image_copy(image_path)
    decoded_objects = decode_qr(img)
    if not decoded_objects:
        return None
    return decoded_objects[0].data.decode("utf-8")


# --- CARD READING (vision model via OpenRouter) ---
CARD_PROMPT = """You are an expert at reading business cards. Look at this image carefully and extract ALL contact information you can find.

This is a business card or contact information image. Extract the following fields:
- name: The person's full name (first and last name)
- email: Their email address (look for @ symbol)
- phone: Their phone number (look for digits, +, -, parentheses)
- linkedin: Their LinkedIn URL or profile (look for linkedin.com or "LinkedIn:")
- company_name: Their company or organization name
- job_title: Their role or job title (e.g. "Head of Data", "VP Sales")

Return ONLY a valid JSON object with exactly these keys. Use "N/A" for any field you cannot find:
{"name": "...", "email": "...", "phone": "...", "linkedin": "...", "company_name": "...", "job_title": "..."}

CRITICAL: Return ONLY the raw JSON object. No markdown, no backticks, no explanation."""

CARD_UNAVAILABLE_MESSAGES = {
    "not_configured": "Card reading isn't set up on the server (no OpenRouter API key).",
    "bad_key": "Card reading is unavailable right now: the server's OpenRouter API key isn't working. Ask your admin to check it.",
    "no_credits": "Card reading is paused: the server's OpenRouter account is out of credits. Ask your admin to top it up.",
    "rate_limited": "Card reading is busy right now (AI rate limit). Please try again in a minute.",
    "timeout": "Reading the card took too long. Please try again.",
    "unavailable": "Card reading couldn't reach the AI service. Please try again.",
}


def _card_jpeg(image_path: str) -> bytes:
    img = load_image_copy(image_path)  # don't hold the temp file open during the AI call
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    if max(img.size) > 1600:  # business cards don't need 4K
        img.thumbnail((1600, 1600), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=90)
    return buf.getvalue()


async def extract_contact_from_image(image_path: str) -> dict:
    """Read a business card photo with the OpenRouter vision model. Raises AIError when no
    model could be reached (setup / quota problem), returns "N/A" fields when nothing was read."""
    fields = {"name": "N/A", "email": "N/A", "phone": "N/A", "linkedin": "N/A", "company_name": "N/A", "job_title": "N/A"}
    jpeg = await asyncio.to_thread(_card_jpeg, image_path)
    text, used = await ai_client.chat(
        [{"role": "user", "content": [ai_client.text_part(CARD_PROMPT), ai_client.image_part(jpeg)]}],
        model=ai_client.VISION_MODEL, max_tokens=600, temperature=0.1, timeout=35, title="Event Scout Card",
    )
    print(f"[OCR] {used}: {text[:300]}")
    try:
        extracted = ai_client.parse_json(text)
    except (json.JSONDecodeError, ValueError):
        print("[OCR] Reply wasn't JSON")
        return fields
    for key in fields:
        v = extracted.get(key)
        if v and str(v).strip() and str(v).strip() != "N/A":
            fields[key] = str(v).strip()
    fields["_source"] = f"openrouter:{used}"
    return fields


# --- LINKEDIN AUTO-LOOKUP ---
def lookup_linkedin_with_ai(name: str, company: str) -> Optional[str]:
    """Guess a LinkedIn URL (disabled at the scan route: models invent URLs)."""
    if not ai_configured or not name or name == "N/A":
        return None
    prompt = f"""Given this person: {name} at {company if company != 'N/A' else 'unknown company'}.

What is their most likely LinkedIn profile URL?

Rules:
- Return ONLY the URL (e.g. https://linkedin.com/in/username)
- If you cannot determine it with reasonable confidence, return exactly: N/A
- Do NOT make up URLs. Only return a URL if you're reasonably confident.
- Return ONLY the URL or N/A, nothing else."""
    try:
        url, _ = ai_client.chat_sync([{"role": "user", "content": prompt}], max_tokens=100, temperature=0, timeout=20, title="Event Scout LinkedIn")
        url = url.strip()
        if "linkedin.com" in url.lower() and url.startswith("http"):
            return url
    except AIError as e:
        print(f"[LINKEDIN] Lookup failed for {name}: {e}")
    return None


# --- WEBHOOK HELPER ---
async def fire_webhook(contact_data: dict, user_data: dict, contact_id: str):
    """Fire-and-forget webhook to n8n after a contact is scanned."""
    if not WEBHOOK_URL:
        return

    payload = {
        "event": "contact_scanned",
        "contact_id": contact_id,
        "contact": {
            "name": contact_data.get("name", "N/A"),
            "email": contact_data.get("email", "N/A"),
            "phone": contact_data.get("phone", "N/A"),
            "linkedin": contact_data.get("linkedin", "N/A"),
            "company_name": contact_data.get("company_name", "N/A"),
            "notes": contact_data.get("notes", ""),
        },
        "scanned_by": user_data,
        # n8n posts enrichment back here (needs X-Webhook-Secret = ENRICH_WEBHOOK_SECRET)
        "callback_url": f"{os.environ.get('PUBLIC_BASE_URL', '').rstrip('/')}/contact/{contact_id}/enrich",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(WEBHOOK_URL, json=payload, headers={"Content-Type": "application/json"})
            print(f"[WEBHOOK] Sent to {WEBHOOK_URL}: status={resp.status_code}")

            # Update shared_contacts webhook status
            session = await get_db_session()
            try:
                stmt = (
                    update(SharedContactDB)
                    .where(SharedContactDB.original_contact_id == uuid.UUID(contact_id))
                    .values(webhook_sent=True, webhook_sent_at=datetime.now(timezone.utc))
                )
                await session.execute(stmt)
                await session.commit()
            finally:
                await session.close()
    except Exception as e:
        print(f"[WEBHOOK] Failed to send: {e}")


# --- DUPLICATE CHECK ---
async def check_duplicate_contact(user_id: str, email: str, phone: str) -> dict | None:
    """Check if a contact with matching email or phone already exists for this user."""
    session = await get_db_session()
    try:
        conditions = []
        if email and email != "N/A":
            conditions.append(func.lower(ContactDB.email) == email.strip().lower())
        if phone and phone != "N/A":
            conditions.append(ContactDB.phone == phone)

        if not conditions:
            return None  # No matchable fields, can't dedup

        result = await session.execute(
            select(ContactDB).where(
                ContactDB.user_id == uuid.UUID(user_id),
                or_(*conditions)
            ).limit(1)
        )
        existing = result.scalar_one_or_none()
        if existing:
            return {
                "contact_id": str(existing.id),
                "name": existing.name,
                "matched_on": "email" if (email and email != "N/A" and (existing.email or "").lower() == email.strip().lower()) else "phone"
            }
        return None
    finally:
        await session.close()


async def _api_routes_live(event_type: str, data: dict) -> None:
    """Tell the admin console (SSE /api/live) about a change made through an app.py route."""
    try:
        from api_routes import broadcast_live_event
        await broadcast_live_event(event_type, data)
    except Exception as e:  # never fail the request because the live feed couldn't be told
        print(f"[LIVE] couldn't send {event_type}: {e}")


async def find_existing_client_contact(user_id: str, client_id: Optional[str]) -> Optional[str]:
    """Idempotent capture: a phone retrying a scan sends the same client_id. Returns that contact's id
    when it's already saved for this user (live or in Recently deleted), so the card isn't read twice."""
    if not client_id:
        return None
    try:
        cid = uuid.UUID(client_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="client_id must be a UUID")
    session = await get_db_session()
    try:
        live, archived = await contact_store.find_by_id(session, cid)
    finally:
        await session.close()
    row = live or archived
    if row is None:
        return None
    if row.user_id != uuid.UUID(str(user_id)):
        raise HTTPException(status_code=409, detail="client_id is already used")
    return str(cid)


def _parse_captured_at(value: Optional[str]) -> Optional[datetime]:
    """When an offline scan was taken (ISO 8601). Future / unparseable values are ignored."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return dt if dt <= now + timedelta(minutes=5) else None


async def _scan_webhook(contact: ContactDB) -> None:
    """contact_store's webhook adapter: a scanned contact goes to n8n in the background (never blocks)."""
    if not WEBHOOK_URL:
        return
    session = await get_db_session()
    try:
        user_row = (await session.execute(select(UserDB).where(UserDB.id == contact.user_id))).scalar_one_or_none()
    finally:
        await session.close()
    user_data = {"user_id": str(contact.user_id), "name": user_row.name if user_row else "N/A", "email": user_row.email if user_row else "N/A"}
    asyncio.create_task(fire_webhook(contact_view.index_view(contact), user_data, str(contact.id)))


contact_store.effects.webhook = _scan_webhook


# --- CONTACT LOGIC ---
async def add_contact_logic(contact: Contact, user_id: str, source: str = "manual", photo_base64: str = None,
                            contact_id: Optional[str] = None, created_at: Optional[datetime] = None) -> dict:
    """Save a scanned / added contact through the contact store — tolerant like any capture (a bad LinkedIn
    from the card is dropped, a nameless card is "Unnamed contact", "N/A" fields stay empty) with the scan
    options: the rep's active event when none was given, a copy in the shared contacts pool, the n8n
    webhook. Search index and live feed follow as for every write. Returns its id and vCard QR."""
    session = await get_db_session()
    try:
        saved = await contact_store.create_contact(
            session, uuid.UUID(user_id),
            {"name": contact.name, "email": contact.email, "phone": contact.phone, "linkedin": contact.linkedin,
             "company": contact.company_name, "title": contact.job_title, "notes": contact.notes,
             "links": contact.links or [], "eventId": contact.event_name or ""},
            strict=False, contact_id=uuid.UUID(contact_id) if contact_id else None,
            created_at=created_at,  # an offline scan keeps the time it was taken
            source=source, photo_base64=photo_base64, event_from_profile=True, shared_pool=True, webhook=True,
        )
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
    contact_id = str(saved.contact.id)
    qr_result = create_qr(contact, contact_id)
    return {"contact_id": contact_id, "qr_base64": qr_result["qr_base64"]}


async def add_contact_from_qr(file: UploadFile, user_id: str, client_id: Optional[str] = None,
                              event_name: Optional[str] = None, captured_at: Optional[str] = None):
    already = await find_existing_client_contact(user_id, client_id)
    if already:
        return {"status": "success", "already_saved": True, "message": "Already saved", "contact_id": already, "extracted_fields": {}}
    temp_filename = scan_temp_path(file)
    content = await file.read()
    with open(temp_filename, "wb") as f:
        f.write(content)
    try:
        qr_data = scan_qr_image(temp_filename)
        if not qr_data:
            raise HTTPException(status_code=400, detail="No QR code found in image")
        if "BEGIN:VCARD" not in qr_data.upper():
            raise HTTPException(status_code=400, detail="QR code does not contain vCard data")
        fields = parse_vcard(qr_data)

        # Same duplicate check as card scans: scanning a badge twice shouldn't add it twice.
        duplicate = await check_duplicate_contact(user_id, fields.get("email", "N/A"), fields.get("phone", "N/A"))
        if duplicate:
            return {
                "status": "duplicate",
                "message": f"Contact already exists: {duplicate['name']}",
                "existing_contact_id": duplicate["contact_id"],
                "existing_name": duplicate["name"],
                "matched_on": duplicate["matched_on"],
                "extracted_fields": fields,
            }

        contact_obj = Contact(**fields)
        if event_name and event_name.strip():
            contact_obj.event_name = event_name.strip()[:255]
        result = await add_contact_logic(contact_obj, user_id, source="qr", contact_id=client_id,
                                         created_at=_parse_captured_at(captured_at))

        # Same auto-pipeline as card scans (the console says it runs after every card or QR scan).
        if AUTO_PIPELINE_ENABLED and OPENROUTER_API_KEY:
            asyncio.create_task(run_contact_pipeline(result["contact_id"], user_id))

        # "Auto-score on scan" (onboarding / profile setting): score in the background.
        auto_score_started = await is_auto_score_enabled(user_id)
        if auto_score_started:
            start_auto_score(result["contact_id"], user_id)

        return {
            "status": "success",
            "message": "Contact added from QR code",
            "extracted_fields": fields,
            "contact_id": result["contact_id"],
            "qr_base64": result["qr_base64"],
            "auto_score_started": auto_score_started,
        }
    finally:
        remove_temp_file(temp_filename)


async def add_contact_from_image(file: UploadFile, user_id: str, client_id: Optional[str] = None,
                                 event_name: Optional[str] = None, captured_at: Optional[str] = None):
    already = await find_existing_client_contact(user_id, client_id)
    if already:
        # a retried upload of a card that was already read: don't read (or save) it twice
        return {"status": "success", "already_saved": True, "message": "Already saved", "contact_id": already, "extracted_fields": {}}
    temp_filename = scan_temp_path(file)
    content = await file.read()
    print(f"[SCAN] Received image: {len(content)} bytes, filename: {file.filename}, content_type: {file.content_type}")
    if len(content) < 1000:
        raise HTTPException(status_code=400, detail="Image file is empty or corrupted. Please try again.")
    with open(temp_filename, "wb") as f:
        f.write(content)

    # Create compressed thumbnail from scanned card for contact photo
    photo_base64 = None
    try:
        thumb_img = Image.open(BytesIO(content))
        if thumb_img.mode == 'RGBA':
            bg = Image.new('RGB', thumb_img.size, (255, 255, 255))
            bg.paste(thumb_img, mask=thumb_img.split()[3])
            thumb_img = bg
        elif thumb_img.mode != 'RGB':
            thumb_img = thumb_img.convert('RGB')
        thumb_img.thumbnail((200, 200), Image.LANCZOS)
        thumb_buffer = BytesIO()
        thumb_img.save(thumb_buffer, format='JPEG', quality=60)
        photo_base64 = base64.b64encode(thumb_buffer.getvalue()).decode('utf-8')
        print(f"[SCAN] Photo thumbnail created: {len(photo_base64)} chars")
    except Exception as photo_err:
        print(f"[SCAN] Photo thumbnail creation failed (non-fatal): {photo_err}")

    try:
        try:
            fields = await extract_contact_from_image(temp_filename)
        except AIError as e:
            # A server setup / quota problem isn't the user's photo: say so instead of
            # "no contact information found" (which tells them to retake the picture).
            print(f"[SCAN] AI card reading failed ({e.reason}): {e}")
            raise HTTPException(status_code=503, detail=CARD_UNAVAILABLE_MESSAGES.get(e.reason, CARD_UNAVAILABLE_MESSAGES["unavailable"]))
        fields.pop("_source", None)
        has_info = any(v != "N/A" for k, v in fields.items() if k != "linkedin")
        if not has_info:
            raise HTTPException(status_code=400, detail="No contact information found in image. Try a sharper, well-lit photo.")

        # DISABLED: LinkedIn auto-lookup generates hallucinated/fake URLs
        # Only use LinkedIn if actually found on the business card via OCR
        # if fields.get("linkedin", "N/A") == "N/A":
        #     linkedin_url = lookup_linkedin_with_ai(fields["name"], fields.get("company_name", "N/A"))
        #     if linkedin_url:
        #         fields["linkedin"] = linkedin_url
        #         fields["linkedin_source"] = "ai_detected"
        #         print(f"[LINKEDIN] Auto-detected: {linkedin_url}")

        # Check for duplicates before adding
        duplicate = await check_duplicate_contact(
            user_id,
            fields.get("email", "N/A"),
            fields.get("phone", "N/A")
        )
        if duplicate:
            print(f"[SCAN] Duplicate detected: {duplicate['name']} (matched on {duplicate['matched_on']})")
            return {
                "status": "duplicate",
                "message": f"Contact already exists: {duplicate['name']}",
                "existing_contact_id": duplicate["contact_id"],
                "existing_name": duplicate["name"],
                "matched_on": duplicate["matched_on"],
                "extracted_fields": fields,
            }

        contact_obj = Contact(**{k: v for k, v in fields.items() if k in Contact.model_fields})
        if event_name and event_name.strip():
            contact_obj.event_name = event_name.strip()[:255]
        result = await add_contact_logic(contact_obj, user_id, source="scan", photo_base64=photo_base64,
                                         contact_id=client_id, created_at=_parse_captured_at(captured_at))

        # Trigger AI pipeline in background (research → pitch → deck)
        if AUTO_PIPELINE_ENABLED and OPENROUTER_API_KEY:
            asyncio.create_task(run_contact_pipeline(result["contact_id"], user_id))
            print(f"[PIPELINE] Auto-triggered for contact {result['contact_id']}")

        # "Auto-score on scan" (onboarding / profile setting): score in the background.
        auto_score_started = await is_auto_score_enabled(user_id)
        if auto_score_started:
            start_auto_score(result["contact_id"], user_id)

        return {
            "status": "success",
            "message": "Contact added from image",
            "extracted_fields": fields,
            "contact_id": result["contact_id"],
            "qr_base64": result["qr_base64"],
            "photo_base64": photo_base64,
            "pipeline_started": AUTO_PIPELINE_ENABLED and bool(OPENROUTER_API_KEY),
            "auto_score_started": auto_score_started,
        }
    finally:
        remove_temp_file(temp_filename)


# --- LEAD SCORING ---
def _scoring_failure_reason(errors: List[str]) -> str:
    joined = " ".join(errors).lower()
    if any(s in joined for s in ("api key not valid", "api_key_invalid", "api key expired", "permission_denied", "unauthenticated")):
        return "bad_key"
    if any(s in joined for s in ("429", "quota", "resource_exhausted", "resourceexhausted", "rate")):
        return "rate_limited"
    return "unavailable"


class AIUnavailable(Exception):
    """The chat assistant / transcription couldn't get an answer from any AI provider.
    Routes turn this into a 503 with AI_UNAVAILABLE_MESSAGES[reason] — never a fake answer."""

    def __init__(self, message: str, reason: str = "unavailable"):
        super().__init__(message)
        self.reason = reason  # "not_configured" | "bad_key" | "rate_limited" | "timeout" | "unavailable"


AI_UNAVAILABLE_MESSAGES = {
    "not_configured": "The AI assistant isn't set up on the server (no OpenRouter API key).",
    "bad_key": "The AI assistant is unavailable: the server's OpenRouter API key isn't working. Ask your admin to check it.",
    "no_credits": "The AI assistant is paused: the server's OpenRouter account is out of credits. Ask your admin to top it up.",
    "rate_limited": "The AI assistant is busy right now (AI rate limit). Please try again in a minute.",
    "timeout": "The AI took too long to answer. Please try again.",
    "unavailable": "The AI assistant couldn't reach the AI service. Please try again.",
}


SCORE_DIMENSIONS = ("role_fit", "company_fit", "need_fit", "goal_fit", "intent")
# One AI answer isn't exactly repeatable (provider routing, "thinking" models), so each score
# asks SCORING_RUNS times in parallel and keeps the median of every dimension.
SCORING_RUNS = max(1, min(5, int(os.environ.get("SCORING_RUNS", "3"))))
def _list(v) -> str:
    return ", ".join(str(x) for x in v) if v else "not set"


def score_contact_with_ai(contact_meta: dict, user_profile: dict) -> dict:
    """The AI call behind scoring.score (injected as scoring.ai_scorer). Blocking (it runs in a thread).
    Raises ScoringUnavailable, never a placeholder score. contact_meta: scoring.scoring_facts(contact).
    The total is the sum of the five 0-20 dimensions."""
    if not ai_configured:
        raise ScoringUnavailable("OPENROUTER_API_KEY is not set", "not_configured")

    c = dict(contact_meta or {})
    c.setdefault("company", c.get("company_name"))  # older callers
    up = user_profile or {}
    products = up.get("products") or []
    products_text = "\n".join(
        f"  - {p.get('name', '')}: {p.get('description', '') or 'no description'}"
        + (f" (ideal customer: {p['ideal_customer']})" if p.get("ideal_customer") else "")
        for p in products if isinstance(p, dict) and p.get("name")
    ) or "  - not specified"
    event = c.get("event") or up.get("current_event_name")
    goals = event_identity.event_goals(up, c.get("event"))

    prompt = f"""You score how valuable a trade-show contact is as a sales lead FOR THIS USER. Be specific and calibrated.

THE USER (who is selling):
- Role: {up.get('job_title') or 'not set'} at {up.get('company_name') or 'not set'}
- Sells:
{products_text}
- Value propositions: {_list(up.get('value_propositions'))}
- Target industries: {_list(up.get('target_industries'))}
- Target roles: {_list(up.get('target_roles'))}
- Target company sizes: {_list(up.get('target_company_sizes'))}
- Target regions: {_list(up.get('target_geographies'))}
- Goals at {event or 'this event'}: {_list(goals)}

THE CONTACT:
- Name: {c.get('name') or 'unknown'}
- Title: {c.get('title') or 'unknown'}
- Company: {c.get('company') or 'unknown'}
- Company web domain: {c.get('company_domain') or 'unknown'}
- Met at: {c.get('event') or 'unknown'}
- How added: {c.get('met_via') or 'unknown'}
- LinkedIn on file: {'yes' if c.get('has_linkedin') else 'no'} · email: {'yes' if c.get('has_email') else 'no'} · phone: {'yes' if c.get('has_phone') else 'no'}
- The user's notes (their own words — the strongest signal of interest): {c.get('notes') or 'none'}

Score five dimensions, each 0-20:
1. role_fit — does the title match the target roles, and can this person buy or influence a purchase? (C-level / VP / Head / Director high; junior / intern / student low. No targets set: judge seniority and decision power.)
2. company_fit — does the company match the target industries, sizes and regions? Use the company name / domain and what is commonly known about well-known companies.
3. need_fit — how likely is it that this company / role needs what the user sells?
4. goal_fit — does meeting them advance the user's goals for this event?
5. intent — interest shown in the notes (asked for a demo, pricing or follow-up = high; "not interested", a student, a vendor pitching to us = low).

Rules:
- Unknown is not bad: when a fact is missing, give that dimension 8-12 and say what's missing. Go below 6 only for a clear mismatch. No notes = intent 8.
- Use the whole 0-20 range. Contacts with different titles or companies should almost never get identical breakdowns.
- Never invent facts (location, size, funding, event location) that are not given above or commonly known.
- reasoning: 2-3 sentences that name the contact's title and company and the deciding factors.
- recommended_actions: 3 short, specific next steps.

Return ONLY this JSON:
{{"breakdown": {{"role_fit": 0, "company_fit": 0, "need_fit": 0, "goal_fit": 0, "intent": 0}}, "reasoning": "...", "recommended_actions": ["...", "...", "..."]}}"""

    def once() -> dict:
        try:
            text, used = ai_client.chat_sync([{"role": "user", "content": prompt}], max_tokens=700, temperature=0,
                                             timeout=25, title="Event Scout Scoring")
        except AIError as e:
            raise ScoringUnavailable(str(e), e.reason)
        try:
            result = ai_client.parse_json(text)
            raw = result.get("breakdown") or {}
            if all(k in raw for k in SCORE_DIMENSIONS):
                breakdown = {k: max(0, min(20, int(round(float(raw[k]))))) for k in SCORE_DIMENSIONS}
                total = sum(breakdown.values())  # the total always matches the parts
            else:
                breakdown = {k: int(v) for k, v in raw.items() if isinstance(v, (int, float))}
                total = int(round(float(result["score"])))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            raise ScoringUnavailable(f"{used} returned no usable score: {text[:200]}", "unavailable")
        return {"breakdown": breakdown, "total": max(0, min(100, total)), "model": used,
                "reasoning": str(result.get("reasoning") or "").strip()[:1200],
                "actions": [str(a).strip() for a in (result.get("recommended_actions") or []) if str(a).strip()][:3]}

    from concurrent.futures import ThreadPoolExecutor
    runs, errors = [], []
    with ThreadPoolExecutor(max_workers=SCORING_RUNS) as pool:
        for fut in [pool.submit(once) for _ in range(SCORING_RUNS)]:
            try:
                runs.append(fut.result())
            except ScoringUnavailable as e:
                errors.append(e)
    if not runs:
        # No placeholder score: the contact stays "not scored" rather than showing a fake 50.
        e = errors[0] if errors else ScoringUnavailable("no answer", "unavailable")
        print(f"[SCORING] {e}")
        raise e

    import statistics
    if all(set(r["breakdown"]) >= set(SCORE_DIMENSIONS) for r in runs):
        breakdown = {k: int(round(statistics.median(r["breakdown"][k] for r in runs))) for k in SCORE_DIMENSIONS}
        score = sum(breakdown.values())
    else:
        breakdown = runs[0]["breakdown"]
        score = int(round(statistics.median(r["total"] for r in runs)))
    score = max(0, min(100, score))
    totals = [r["total"] for r in runs]
    best = min(runs, key=lambda r: abs(r["total"] - score))  # reasoning from the run closest to the final score
    print(f"[SCORING] {len(runs)} run(s) {totals} -> {score} ({best['model']})")
    return {
        "score": score,
        "temperature": score_levels.temperature_for(score),
        "reasoning": best["reasoning"],
        "recommended_actions": best["actions"],
        "breakdown": breakdown,
        "model": best["model"],
        "runs": len(runs),
        "spread": max(totals) - min(totals),
    }


# --- INTELLIGENCE PIPELINE ---

def build_research_prompt(contact_name: str, company_name: str, email: str, linkedin: str, user_profile: dict,
                          contact_title: str = "", contact_event: str = "") -> str:
    """Build a dynamic research prompt personalised to the user's products, market, and event context.
    The event (and its goals) is the one the contact was met at when known, else the rep's current one."""
    company = user_profile.get("company_name", "our company")
    job_title = user_profile.get("job_title", "sales representative")
    event = contact_event or user_profile.get("current_event_name") or "the event"
    products = user_profile.get("products", [])
    target_industries = user_profile.get("target_industries", [])
    target_roles = user_profile.get("target_roles", [])
    value_props = user_profile.get("value_propositions", [])
    # that event's own goals (profile.event_details), falling back to the current event's
    event_goals = ((user_profile.get("event_details") or {}).get(contact_event) or {}).get("goals") if contact_event else None
    event_goals = event_goals or user_profile.get("event_goals", [])

    products_text = "\n".join(
        f"  - {p.get('name','')}: {p.get('description','')} (Ideal customer: {p.get('ideal_customer','')})"
        for p in products if p.get("name")
    ) or "  - Not specified"

    target_industries_text = ", ".join(target_industries) if target_industries else "All industries"
    target_roles_text = ", ".join(target_roles) if target_roles else "Decision makers"
    value_props_text = "\n".join(f"  - {vp}" for vp in value_props) if value_props else "  - Not specified"
    event_goals_text = "\n".join(f"  - {g}" for g in event_goals) if event_goals else "  - Close new deals"

    return f"""You are a B2B sales intelligence analyst. Research this contact on behalf of {job_title} at {company} attending {event}.

CONTACT TO RESEARCH:
- Name: {contact_name}
- Job title: {contact_title or "Unknown"}
- Company: {company_name}
- Email: {email}
- LinkedIn: {linkedin}

YOUR COMPANY CONTEXT:
- Company: {company}
- Products/Services:
{products_text}
- Target Industries: {target_industries_text}
- Target Roles: {target_roles_text}
- Value Propositions:
{value_props_text}
- Event Goals:
{event_goals_text}

RESEARCH REQUIREMENTS:
1. PERSON PROFILE: Likely role, seniority level, decision-making authority, key responsibilities
2. COMPANY ANALYSIS: Industry vertical, estimated size, core products/services, growth signals, tech stack indicators
3. PAIN POINTS: What specific challenges does this contact/company face that YOUR company's products can address?
4. OPPORTUNITIES: How do YOUR specific products/services map to their needs? Be concrete.
5. CONVERSATION STARTERS: 3 specific, natural talking points tailored to your products and their likely situation
6. COMPETITIVE LANDSCAPE: Who else might be pitching them? How are you differentiated?

Return ONLY valid JSON (no markdown fences):
{{"person": {{"name": "", "likely_title": "", "seniority": "", "authority": ""}}, "company": {{"name": "", "industry": "", "size_estimate": "", "products": "", "growth_signals": ""}}, "industry": "", "pain_points": [], "opportunities": [], "talking_points": [], "competitive_notes": "", "confidence_level": "high|medium|low"}}"""


def build_pitch_prompt(research_data: dict, user_profile: dict) -> str:
    """Build a dynamic pitch prompt using the user's actual brand voice and products."""
    company = user_profile.get("company_name", "our company")
    products = user_profile.get("products", [])
    value_props = user_profile.get("value_propositions", [])
    pitch_style = user_profile.get("pitch_style", "consultative")
    case_studies = user_profile.get("case_studies", [])
    common_objections = user_profile.get("common_objections", [])

    products_text = "\n".join(
        f"  - {p.get('name','')}: {p.get('description','')} (Ideal: {p.get('ideal_customer','')})"
        for p in products if p.get("name")
    ) or "  - Not specified"

    value_props_text = "\n".join(f"  - {vp}" for vp in value_props) if value_props else "  - Not specified"

    case_studies_text = "\n".join(
        f"  - {cs.get('title','')}: {cs.get('result','')}"
        for cs in (case_studies or [])
    ) or "  - Use industry-relevant examples"

    objections_text = "\n".join(
        f"  - Objection: {o.get('objection','')} → Response: {o.get('response','')}"
        for o in (common_objections or [])
    ) or "  - Address cost, timeline, integration concerns"

    pitch_style_desc = {
        "consultative": "Consultative — ask questions first, position as trusted partner",
        "direct": "Direct — lead with solution and ROI, get to the point fast",
        "challenger": "Challenger — reframe their thinking, share counterintuitive insights",
        "relationship": "Relationship — build trust first, find common ground, long-term play",
    }.get(pitch_style, pitch_style)

    return f"""You are a senior pitch strategist for {company}. Generate a personalized pitch for this contact.

BRAND CONTEXT:
- Company: {company}
- Pitch Style: {pitch_style_desc}
- Products/Services:
{products_text}
- Value Propositions:
{value_props_text}
- Case Studies / Proof Points:
{case_studies_text}
- Common Objections & Responses:
{objections_text}

CONTACT RESEARCH:
{json.dumps(research_data, indent=2)}

Generate TWO outputs:

OUTPUT 1 — PITCH DECK (8 slides):
Each slide as: {{"title": "...", "content": "2-3 bullet points or short paragraphs", "speaker_notes": "what to say"}}
Structure (personalised to the contact's specific situation from the research above):
1. Opening Hook — Personalized to their company/industry challenge
2. The Challenge — Their specific pain points from research
3. Market Reality — Industry urgency with supporting data
4. Cost of Inaction — What they risk by not acting
5. Our Approach — {company}'s methodology and philosophy
6. The Solution — Specific {company} product mapping to their pain points
7. Proof Points — Case studies and results most relevant to their situation
8. Next Steps — Clear CTA: meeting, demo, or pilot suggestion

OUTPUT 2 — EMAIL PITCH:
- Subject (under 50 chars, personalized to their company/role)
- Body (3-4 paragraphs: hook referencing their specific situation, value prop, proof point, CTA)
- Professional sign-off from {company}

Return ONLY valid JSON (no markdown fences):
{{"slides": [...], "email_subject": "...", "email_body": "..."}}"""


async def _pipeline_call_openrouter(prompt: str, model_id: str = None, max_tokens: int = 3000) -> str:
    """Call OpenRouter for pipeline tasks (research/pitch). Returns raw text."""
    text, _ = await ai_client.chat([{"role": "user", "content": prompt}], model=model_id or ai_client.DEFAULT_MODEL,
                                   max_tokens=max_tokens, temperature=0.4, timeout=90, title="Event Scout Pipeline")
    return text


async def _pipeline_update_status(pipeline_id, status: str, step: str = "", **kwargs):
    """Update pipeline status in DB."""
    session = await get_db_session()
    try:
        result = await session.execute(select(ContactPipelineDB).where(ContactPipelineDB.id == pipeline_id))
        pipeline = result.scalar_one_or_none()
        if not pipeline:
            return
        pipeline.status = status
        # current_step is VARCHAR(100): Postgres rejects longer text, and then even a failure
        # couldn't be recorded (the run would look "running" forever). The full error goes to error_message.
        step = step or ""
        pipeline.current_step = step if len(step) <= 100 else step[:99] + "…"
        pipeline.updated_at = datetime.now(timezone.utc)
        for k, v in kwargs.items():
            if hasattr(pipeline, k):
                setattr(pipeline, k, v)
        await session.commit()
    finally:
        await session.close()


async def pipeline_step_research(pipeline_id, contact_id: str, user_id: str, admin: bool = False):
    """Step 1: Research the contact using Claude via OpenRouter.
    admin=True (run from the admin console): the research stays on the pipeline record only;
    the rep's contact notes are not touched."""
    await _pipeline_update_status(pipeline_id, "researching", "Researching contact...")

    session = await get_db_session()
    try:
        result = await session.execute(select(ContactDB).where(ContactDB.id == uuid.UUID(contact_id)))
        contact = result.scalar_one_or_none()
        if not contact:
            raise Exception(f"Contact {contact_id} not found")

        # Personalise the research prompt: the rep's own profile, or (admin run) only the admin's
        if admin:
            user_profile = await admin_ai_profile(session)
        else:
            prof_result = await session.execute(
                select(UserProfileDB).where(UserProfileDB.user_id == uuid.UUID(user_id))
            )
            profile = prof_result.scalar_one_or_none()
            user_profile = event_identity.read_profile(profile)

        present = contact_view.present  # "N/A" placeholders count as missing
        prompt = build_research_prompt(
            contact_name=contact.name or "Unknown",
            company_name=present(contact.company_name) or "Unknown",
            email=present(contact.email) or "Unknown",
            linkedin=present(contact.linkedin) or "Unknown",
            user_profile=user_profile,
            contact_title=present(contact.job_title) or "",
            contact_event=(contact.event_name or "").strip(),
        )

        raw_response = await _pipeline_call_openrouter(prompt)

        # Parse JSON from response (handle markdown fences)
        clean = raw_response.strip()
        if clean.startswith("```"):
            clean = re.sub(r'^```(?:json)?\s*', '', clean)
            clean = re.sub(r'\s*```$', '', clean)

        try:
            research_data = json.loads(clean)
        except json.JSONDecodeError:
            research_data = {"raw_research": raw_response, "parse_error": True}

        # Build human-readable summary
        summary_parts = []
        if isinstance(research_data, dict) and not research_data.get("parse_error"):
            person = research_data.get("person", {})
            company = research_data.get("company", {})
            if person.get("likely_title"):
                summary_parts.append(f"Role: {person['likely_title']} ({person.get('seniority', 'unknown')} level)")
            if company.get("industry"):
                summary_parts.append(f"Industry: {company['industry']}")
            if company.get("size_estimate"):
                summary_parts.append(f"Company size: {company['size_estimate']}")
            for pp in research_data.get("pain_points", [])[:3]:
                summary_parts.append(f"Pain point: {pp}")
            for opp in research_data.get("opportunities", [])[:2]:
                summary_parts.append(f"Opportunity: {opp}")

        research_summary = "\n".join(summary_parts) if summary_parts else raw_response[:500]

        # Update pipeline
        await _pipeline_update_status(
            pipeline_id, "researching", "Research complete",
            research_data=research_data,
            research_summary=research_summary,
        )

        # Also append research to contact notes — only for the rep's own (auto) pipeline
        if not admin:
            existing_notes = contact.notes or ""
            research_block = f"\n\n--- AI Research ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}) ---\n{research_summary}"
            contact.notes = existing_notes + research_block
            await contact_store.save(session, contact, "contact_updated")

        return research_data
    finally:
        await session.close()


async def pipeline_step_score(pipeline_id, contact_id: str, user_id: str, admin: bool = False):
    """Step 2: Score the lead (scoring.score, the same scorer as the app; an up-to-date score is kept).
    admin=True: saved as the admin console's score (admin_contact_scores); the rep's score is untouched."""
    await _pipeline_update_status(pipeline_id, "scoring", "Scoring lead...")
    try:
        result = await scoring.score(contact_id, "admin" if admin else "rep")
    except HTTPException as e:
        # only a real score is saved, never a placeholder
        print(f"[PIPELINE] Scoring unavailable ({e.detail}) — no score saved")
        result = None
    if result is not None:
        step = "Scoring complete (admin score)" if admin else "Scoring complete"
    else:
        step = "Scoring unavailable — no admin score saved" if admin else "Scoring unavailable — contact left unscored"
    await _pipeline_update_status(pipeline_id, "scoring", step, score_completed=result is not None)
    return result


async def pipeline_step_pitch(pipeline_id, contact_id: str, user_id: str, research_data: dict, admin: bool = False):
    """Step 3: Generate personalized pitch + email using Claude.
    admin=True: pitches from the admin console's own profile only (no rep profile is read)."""
    await _pipeline_update_status(pipeline_id, "pitching", "Generating pitch...")

    session = await get_db_session()
    try:
        if admin:
            user_profile = await admin_ai_profile(session)
        else:
            prof_result = await session.execute(
                select(UserProfileDB).where(UserProfileDB.user_id == uuid.UUID(user_id))
            )
            profile = prof_result.scalar_one_or_none()
            user_profile = event_identity.read_profile(profile)

        prompt = build_pitch_prompt(research_data, user_profile)

        raw_response = await _pipeline_call_openrouter(prompt, max_tokens=4000)

        # Parse JSON
        clean = raw_response.strip()
        if clean.startswith("```"):
            clean = re.sub(r'^```(?:json)?\s*', '', clean)
            clean = re.sub(r'\s*```$', '', clean)

        try:
            pitch_data = json.loads(clean)
        except json.JSONDecodeError:
            # keep what the AI wrote as the email body; no made-up subject
            pitch_data = {"slides": [], "email_subject": "", "email_body": raw_response}
        if not isinstance(pitch_data, dict):
            pitch_data = {"slides": [], "email_subject": "", "email_body": raw_response}

        slides = pitch_data.get("slides", [])
        email_subject = pitch_data.get("email_subject", "")
        email_body = pitch_data.get("email_body", "")

        # Build pitch angle narrative from slides
        pitch_angle = "\n".join(
            f"Slide {i+1}: {s.get('title', 'Untitled')}" for i, s in enumerate(slides)
        )

        await _pipeline_update_status(
            pipeline_id, "pitching", "Pitch generated",
            pitch_angle=pitch_angle,
            pitch_email_subject=email_subject,
            pitch_email_body=email_body,
            pitch_slides_content=slides,
        )

        return pitch_data
    finally:
        await session.close()


def _slides_to_gamma_text(slides_content: list, contact_name: str) -> str:
    """Convert pipeline slide objects to Gamma-compatible markdown with --- slide breaks."""
    parts = []
    for slide in slides_content:
        title = slide.get("title", "Slide")
        content = slide.get("content", "")
        speaker_notes = slide.get("speaker_notes", "")
        slide_text = f"## {title}\n{content}"
        if speaker_notes:
            slide_text += f"\n\n*Note: {speaker_notes}*"
        parts.append(slide_text)
    return "\n\n---\n\n".join(parts)


async def pipeline_step_deck(pipeline_id, slides_content: list, contact_name: str):
    """Step 4: Generate PPTX via Gamma API. Returns (pptx_bytes, gamma_url, why_no_deck);
    why_no_deck says why there's no deck (Gamma not set up / failed) so the run can report it."""
    if not GAMMA_API_KEY:
        print("[PIPELINE] Gamma API key not configured — skipping deck generation")
        await _pipeline_update_status(pipeline_id, "generating_deck", "Skipped (Gamma not configured)")
        return None, None, "Gamma isn't set up (GAMMA_API_KEY)"

    await _pipeline_update_status(pipeline_id, "generating_deck", "Creating presentation with Gamma...")

    try:
        input_text = _slides_to_gamma_text(slides_content, contact_name)

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Kick off async generation
            resp = await client.post(
                f"{GAMMA_API_BASE}/generations",
                headers={"X-API-KEY": GAMMA_API_KEY, "Content-Type": "application/json"},
                json={
                    "inputText": input_text,
                    "textMode": "preserve",
                    "cardSplit": "inputTextBreaks",
                    "format": "presentation",
                    "exportAs": "pptx",
                    "themeId": GAMMA_THEME,
                },
            )
            if resp.status_code != 200 and resp.status_code != 201:
                raise Exception(f"Gamma generation failed: {resp.status_code} - {resp.text[:200]}")

            generation_id = resp.json().get("generationId")
            if not generation_id:
                raise Exception(f"No generationId in Gamma response: {resp.text[:200]}")

            print(f"[PIPELINE] Gamma generationId={generation_id}, polling...")

        # Poll for completion (up to 3 minutes)
        gamma_url = None
        export_url = None
        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(36):  # 36 × 5s = 3 min max
                await asyncio.sleep(5)
                poll = await client.get(
                    f"{GAMMA_API_BASE}/generations/{generation_id}",
                    headers={"X-API-KEY": GAMMA_API_KEY},
                )
                poll_data = poll.json()
                status = poll_data.get("status")
                print(f"[PIPELINE] Gamma poll [{attempt+1}]: {status}")

                if status == "completed":
                    gamma_url = poll_data.get("gammaUrl")
                    export_url = poll_data.get("exportUrl")
                    credits_left = poll_data.get("credits", {}).get("remaining", "?")
                    print(f"[PIPELINE] Gamma complete — {gamma_url} | credits remaining: {credits_left}")
                    break
                elif status == "failed":
                    raise Exception(f"Gamma generation failed: {poll_data.get('error', {})}")

        if not gamma_url:
            raise Exception("Gamma generation timed out or returned no URL")

        # Download PPTX bytes if export URL available
        pptx_bytes = None
        if export_url:
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    dl = await client.get(export_url)
                    if dl.status_code == 200:
                        pptx_bytes = dl.content
                        print(f"[PIPELINE] PPTX downloaded: {len(pptx_bytes):,} bytes")
                    else:
                        print(f"[PIPELINE] PPTX download failed: {dl.status_code} — using Gamma URL only")
            except Exception as dl_err:
                print(f"[PIPELINE] PPTX download error: {dl_err} — using Gamma URL only")

        await _pipeline_update_status(
            pipeline_id, "generating_deck", "Deck created via Gamma",
            gamma_deck_url=gamma_url,
        )

        return pptx_bytes, gamma_url, None

    except Exception as e:
        print(f"[PIPELINE] Deck generation failed: {e}")
        await _pipeline_update_status(pipeline_id, "generating_deck", f"Deck failed: {str(e)[:100]}")
        return None, None, f"the deck failed ({str(e)[:120]})"


ADMIN_FILE_CATEGORY = "admin_pitch"  # files only the admin console sees (never listed to the rep)


async def pipeline_step_attach(pipeline_id, contact_id: str, user_id: str,
                               pptx_bytes: Optional[bytes], contact_name: str, gamma_url: Optional[str] = None,
                               admin: bool = False):
    """Step 5: Attach deck to contact — store PPTX as file and add Gamma URL to links.
    admin=True: the deck is stored as an admin-only file and the rep's links are not touched
    (the Gamma URL stays on the pipeline record)."""
    await _pipeline_update_status(pipeline_id, "attaching", "Attaching deck to contact...")

    session = await get_db_session()
    try:
        file_id = None
        safe_name = re.sub(r'[^a-zA-Z0-9_\- ]', '', contact_name or "contact").strip().replace(' ', '_')

        # Store PPTX bytes as a contact file
        if pptx_bytes:
            filename = f"pitch_deck_{safe_name}.pptx"
            db_file = ContactFileDB(
                contact_id=uuid.UUID(contact_id),
                filename=filename,
                original_filename=filename,
                file_type="pptx",
                mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                file_size=len(pptx_bytes),
                file_data=pptx_bytes,
                description="AI-generated pitch deck (Gamma)" + (" · admin only" if admin else ""),
                category=ADMIN_FILE_CATEGORY if admin else "pitch",
                uploaded_by=uuid.UUID(user_id),
            )
            session.add(db_file)
            await session.flush()
            file_id = db_file.id

        # Append Gamma shareable URL to contact's links[] (rep's own pipeline only)
        contact = None
        if gamma_url and not admin:
            result = await session.execute(select(ContactDB).where(ContactDB.id == uuid.UUID(contact_id)))
            contact = result.scalar_one_or_none()
            if contact:
                existing_links = list(contact.links or [])
                # Remove any previous deck link
                existing_links = [l for l in existing_links if l.get("label") != "AI Pitch Deck"]
                existing_links.append({
                    "url": gamma_url,
                    "label": "AI Pitch Deck",
                    "added_by": "pipeline",
                })
                contact.links = existing_links

        if contact is not None:
            await contact_store.save(session, contact, "contact_updated")  # the deck file is committed with it
        else:
            await session.commit()

        update_kwargs = {"deck_file_id": file_id} if file_id else {}
        await _pipeline_update_status(pipeline_id, "attaching", "Deck attached", **update_kwargs)

        return str(file_id) if file_id else None
    finally:
        await session.close()


async def run_contact_pipeline(contact_id: str, user_id: str, admin: bool = False):
    """Main pipeline orchestrator: research → score → pitch → deck → attach.
    admin=True when started from the admin console: nothing on the rep's contact changes
    (no notes appended, rep score untouched, no links added, deck stored admin-only)."""
    print(f"[PIPELINE] Starting for contact={contact_id} user={user_id} admin={admin}")

    # Create pipeline record
    session = await get_db_session()
    try:
        pipeline = ContactPipelineDB(
            contact_id=uuid.UUID(contact_id),
            user_id=uuid.UUID(user_id),
            status="pending",
            current_step="Initializing...",
            started_at=datetime.now(timezone.utc),
            admin_run=admin,
        )
        session.add(pipeline)
        await session.commit()
        pipeline_id = pipeline.id
    finally:
        await session.close()
    if not admin:  # console runs announce themselves (POST /api/contact/{id}/pipeline/run)
        await _api_routes_live("pipeline_started", {"contactId": contact_id})

    # Get contact name for logging
    session = await get_db_session()
    try:
        result = await session.execute(select(ContactDB).where(ContactDB.id == uuid.UUID(contact_id)))
        contact = result.scalar_one_or_none()
        contact_name = contact.name if contact else "Unknown"
    finally:
        await session.close()

    try:
        # Step 1: Research
        print(f"[PIPELINE] Step 1/5: Researching {contact_name}...")
        research_data = await pipeline_step_research(pipeline_id, contact_id, user_id, admin=admin)

        # Step 2: Score
        print(f"[PIPELINE] Step 2/5: Scoring {contact_name}...")
        await pipeline_step_score(pipeline_id, contact_id, user_id, admin=admin)

        # Step 3: Pitch
        print(f"[PIPELINE] Step 3/5: Generating pitch for {contact_name}...")
        pitch_data = await pipeline_step_pitch(pipeline_id, contact_id, user_id, research_data, admin=admin)

        # Step 4: Deck via Gamma API
        slides = pitch_data.get("slides", [])
        pptx_bytes = None
        gamma_url = None
        no_deck_reason = "the pitch had no slides to build a deck from"
        if slides:
            print(f"[PIPELINE] Step 4/5: Generating Gamma deck for {contact_name}...")
            pptx_bytes, gamma_url, no_deck_reason = await pipeline_step_deck(pipeline_id, slides, contact_name)

        # Step 5: Attach deck to contact (PPTX file + Gamma URL link)
        deck_created = bool(pptx_bytes or gamma_url)
        if deck_created:
            print(f"[PIPELINE] Step 5/5: Attaching deck for {contact_name}...")
            await pipeline_step_attach(pipeline_id, contact_id, user_id, pptx_bytes, contact_name, gamma_url, admin=admin)

        # Mark complete — when there's no deck, say why (it used to be overwritten by "Pipeline complete")
        final_status = "complete" if deck_created else "complete_no_deck"
        await _pipeline_update_status(
            pipeline_id, final_status,
            "Pipeline complete" if deck_created else f"Complete, no deck: {no_deck_reason}"[:100],
            completed_at=datetime.now(timezone.utc),
        )
        print(f"[PIPELINE] Complete for {contact_name} (status={final_status})")
        await _api_routes_live("pipeline_finished", {"contactId": contact_id, "name": contact_name, "status": final_status})

    except Exception as e:
        print(f"[PIPELINE] Failed for {contact_name}: {e}")
        traceback.print_exc()
        await _pipeline_update_status(
            pipeline_id, "failed", f"Failed: {str(e)}",
            error_message=str(e) or e.__class__.__name__,
            completed_at=datetime.now(timezone.utc),
        )
        await _api_routes_live("pipeline_finished", {"contactId": contact_id, "name": contact_name, "status": "failed"})


# --- FASTAPI APP ---
# Interactive API docs only outside production (ENABLE_API_DOCS=true shows them in production too).
_API_DOCS = not IS_PRODUCTION or os.environ.get("ENABLE_API_DOCS", "false").lower() == "true"
app = FastAPI(title="Contact Assistant API - Multi-User (PostgreSQL)", version="2.1.0-admin-cmd",
              docs_url="/docs" if _API_DOCS else None, redoc_url="/redoc" if _API_DOCS else None,
              openapi_url="/openapi.json" if _API_DOCS else None)
# Every JSON body is sanitized before any route sees it (control / invisible characters stripped, script links in
# URL fields refused) — see sanitize.py. Added before CORS so CORS stays outermost and its 400s reach browsers.
from sanitize import SanitizeJSONMiddleware, safe_csv_writer  # noqa: E402
app.add_middleware(SanitizeJSONMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["X-Row-Count", "Content-Disposition"])

from api_routes import api_router, seed_initial_data_if_empty
app.include_router(api_router)
# Contact share links: owner routes under /api, public page /s/{token} (token only).
from share_routes import share_api_router, share_public_router
app.include_router(share_api_router)
app.include_router(share_public_router)
# My card: your own contact QR (vCard) + public card page /c/{token}.
from card_routes import card_api_router, card_public_router
app.include_router(card_api_router)
app.include_router(card_public_router)
# Admin Exports → Send to CRM (HubSpot / Salesforce / Zoho / Pipedrive; credentials from env, see crm.py).
from crm_routes import crm_router
app.include_router(crm_router)


@app.on_event("startup")
async def startup_event():
    print("=" * 50)
    print("[STARTUP] Event Scout Intelligence API v3.4 Starting...")
    
    # Validate database URL format
    db_url_status = "not configured"
    if ASYNC_DATABASE_URL:
        if ASYNC_DATABASE_URL.startswith("postgresql+asyncpg://"):
            # Mask credentials for logging
            try:
                url_parts = ASYNC_DATABASE_URL.replace("postgresql+asyncpg://", "").split("@")
                if len(url_parts) == 2:
                    db_url_status = f"configured (host: {url_parts[1].split('/')[0]})"
                else:
                    db_url_status = "configured (format OK)"
            except:
                db_url_status = "configured"
        else:
            db_url_status = f"invalid format (starts with: {ASYNC_DATABASE_URL[:30]}...)"
    print(f"[STARTUP] Database: {db_url_status}")
    
    print(f"[STARTUP] AI (OpenRouter) configured: {ai_configured} · models: {ai_client.status()}")
    print(f"[STARTUP] Available models: {', '.join(m['name'] for m in AVAILABLE_MODELS.values())}")
    print(f"[STARTUP] Webhook URL: {WEBHOOK_URL or 'not set'}")
    # whether "Forgot password?" can work (no secrets printed)
    print("[STARTUP] Password reset email: " + (
        f"on via {SMTP_HOST}:{SMTP_PORT}{' (TLS check off)' if not SMTP_TLS_VERIFY else ''}" if EMAIL_ENABLED
        else "dev log (codes printed here)" if PASSWORD_RESET_DEV_LOG
        else "OFF — set SMTP_HOST + SMTP_FROM in .env (Forgot password shows 'not configured')"))
    print("=" * 50)

    # Initialize database tables
    db_ok = await init_db()
    if not db_ok:
        print("[STARTUP] WARNING: Database initialization failed!")
        return

    # Seed demo data if database is fresh
    await seed_initial_data_if_empty()

    # The stored temperature is the score's level (score_levels: hot / warm / cool / cold). Older code
    # stored other cut-offs (hot ≥ 70, then a 3-level rule that folded cool into cold), so recompute it for
    # every scored contact and every admin score with today's rule.
    try:
        s = await get_db_session()
        try:
            res = await s.execute(
                update(ContactDB)
                .where(ContactDB.lead_score.isnot(None))
                # updated_at kept as-is: this housekeeping isn't an edit, and the column's onupdate
                # would otherwise stamp every scored contact "Last updated: now" on each restart
                .values(lead_temperature=score_levels.level_case(ContactDB.lead_score), updated_at=ContactDB.updated_at)
            )
            admin_res = await s.execute(
                update(AdminContactScoreDB)
                .where(AdminContactScoreDB.score.isnot(None))
                .values(temperature=score_levels.level_case(AdminContactScoreDB.score))
            )
            await s.commit()
            print(f"[STARTUP] Lead temperature re-derived from scores ({res.rowcount} contacts, {admin_res.rowcount} admin scores)")
        finally:
            await s.close()
    except Exception as e:
        print(f"[STARTUP] Could not re-derive lead temperatures: {e}")

    # Pipeline runs that were in progress when the server stopped can never finish (their task
    # died with the process). Left "running", they'd also block new runs for that contact.
    try:
        s = await get_db_session()
        try:
            res = await s.execute(
                update(ContactPipelineDB)
                .where(ContactPipelineDB.status.in_(_api_routes_module.PIPELINE_RUNNING))
                .values(status="failed", current_step="Interrupted (server restarted)",
                        error_message="The server restarted while this run was in progress. Run it again.",
                        # it died at its last progress, not at this restart (else every interrupted run
                        # shows the same "failed" time, maybe days late)
                        completed_at=func.coalesce(ContactPipelineDB.updated_at, ContactPipelineDB.started_at, datetime.now(timezone.utc)))
            )
            await s.commit()
            if res.rowcount:
                print(f"[STARTUP] Marked {res.rowcount} interrupted pipeline run(s) as failed")
        finally:
            await s.close()
    except Exception as e:
        print(f"[STARTUP] Could not clean up interrupted pipeline runs: {e}")

    # Rebuild FAISS indices from Postgres
    print("[STARTUP] Rebuilding FAISS indices from PostgreSQL...")
    session = await get_db_session()
    try:
        # Get all users
        result = await session.execute(select(UserDB))
        users = result.scalars().all()
        for user in users:
            user_id_str = str(user.id)
            contacts_result = await session.execute(
                select(ContactDB).where(ContactDB.user_id == user.id)
            )
            contacts = contacts_result.scalars().all()
            faiss_index.build_for_user(user_id_str, [contact_view.index_view(c) for c in contacts])

        print(f"[STARTUP] FAISS indices built for {len(users)} users")
    except Exception as e:
        print(f"[STARTUP] Error building FAISS indices: {e}")
        traceback.print_exc()
    finally:
        await session.close()

    # Auto-backup to secondary Postgres (non-blocking)
    if ASYNC_BACKUP_URL:
        # (asyncio is imported at module level; a local import here would shadow it for the
        # whole function and break the task scheduling below.)
        asyncio.create_task(_run_startup_backup())
        asyncio.create_task(_periodic_backup_loop())
        print(f"[STARTUP] Backup configured: {ASYNC_BACKUP_URL[:30]}...")
    else:
        print("[STARTUP] No BACKUP_DATABASE_URL set - backup disabled")

    # Recently deleted contacts are kept 30 days; purge expired ones now and every 6 hours.
    _purge_task = asyncio.create_task(_purge_deleted_contacts_loop())
    _background_tasks.add(_purge_task)
    _purge_task.add_done_callback(_background_tasks.discard)

    # Every company's industry, worked out automatically for the admin dashboard.
    _industry_task = asyncio.create_task(_classify_industries_loop())
    _background_tasks.add(_industry_task)
    _industry_task.add_done_callback(_background_tasks.discard)

    print("[STARTUP] Ready!")


BACKUP_INTERVAL_HOURS = int(os.environ.get("BACKUP_INTERVAL_HOURS", "6"))


async def _run_startup_backup():
    """Background task: auto-backup on startup (with small delay to let app fully start)."""
    await asyncio.sleep(5)  # Let startup finish first
    try:
        result = await run_backup()
        print(f"[BACKUP] Startup auto-backup complete: {result.get('summary', 'unknown')}")
    except Exception as e:
        print(f"[BACKUP] Startup auto-backup failed: {e}")


async def _purge_deleted_contacts_loop():
    """Background task: permanently remove contacts deleted more than 30 days ago."""
    from api_routes import purge_expired_deleted_contacts
    while True:
        try:
            await purge_expired_deleted_contacts()
        except Exception as e:
            print(f"[CONTACTS] Purge of expired deleted contacts failed: {e}")
        await asyncio.sleep(6 * 3600)


async def _classify_industries_loop():
    """Background task: work out the industry of every company the team meets (AI), so the admin
    dashboard never waits on reps running research. Shortly after startup, then every 30 minutes."""
    from api_routes import classify_company_industries
    await asyncio.sleep(20)  # let startup finish first
    while True:
        try:
            await classify_company_industries()
        except Exception as e:
            print(f"[INDUSTRY] Background classification failed: {e}")
        await asyncio.sleep(30 * 60)


async def _periodic_backup_loop():
    """Background task: run backup every N hours."""
    interval = BACKUP_INTERVAL_HOURS * 3600
    print(f"[BACKUP] Periodic backup scheduled every {BACKUP_INTERVAL_HOURS} hours")
    while True:
        await asyncio.sleep(interval)
        try:
            result = await run_backup()
            print(f"[BACKUP] Periodic backup complete: {result.get('summary', 'unknown')}")
        except Exception as e:
            print(f"[BACKUP] Periodic backup failed: {e}")


# --- BACKUP SYSTEM ---
_last_backup_result = {}


async def run_backup() -> dict:
    """Copy all data from primary DB to backup DB. Returns stats."""
    import time as _time
    start = _time.time()

    backup_factory = get_backup_session_factory()
    if not backup_factory:
        return {"status": "error", "message": "Backup database not configured"}

    primary_session = await get_db_session()
    backup_session = backup_factory()

    stats = {}

    # Every table the app defines, in FK-safe order (parents before children) — so a table added later
    # (lists, events, assignments, booth visits, share links, …) is never silently left out of the backup.
    model_by_table = {m.local_table.name: m.class_ for m in Base.registry.mappers}
    tables = [(t.name, model_by_table[t.name]) for t in Base.metadata.sorted_tables if t.name in model_by_table]

    try:
        for table_name, model in tables:
            try:
                # Read all rows from primary
                result = await primary_session.execute(select(model))
                rows = result.scalars().all()

                if not rows:
                    stats[table_name] = 0
                    continue

                # Get column names (excluding relationships)
                columns = [c.key for c in model.__table__.columns]

                # For each row, upsert into backup using merge
                for row in rows:
                    row_data = {col: getattr(row, col) for col in columns}
                    # Use merge for upsert behavior (insert or update by PK)
                    merged = await backup_session.merge(model(**row_data))

                await backup_session.commit()
                stats[table_name] = len(rows)

            except Exception as table_err:
                await backup_session.rollback()
                stats[table_name] = f"ERROR: {str(table_err)[:100]}"
                print(f"[BACKUP] Error backing up {table_name}: {table_err}")

        duration = round(_time.time() - start, 2)
        total_rows = sum(v for v in stats.values() if isinstance(v, int))
        failed = [name for name, v in stats.items() if not isinstance(v, int)]
        copied = len(stats) - len(failed)

        global _last_backup_result
        _last_backup_result = {
            # a table that failed makes the backup "partial", never "success"
            "status": "partial" if failed else "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": duration,
            "tables": stats,
            "total_rows": total_rows,
            "failed_tables": failed,
            "summary": f"{total_rows} rows across {copied} tables in {duration}s"
                       + (f" — {len(failed)} failed: {', '.join(failed)}" if failed else ""),
        }
        return _last_backup_result

    except Exception as e:
        traceback.print_exc()
        _last_backup_result = {
            "status": "error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": str(e),
            "tables": stats
        }
        return _last_backup_result
    finally:
        await primary_session.close()
        await backup_session.close()


@app.on_event("shutdown")
async def shutdown_event():
    print("[SHUTDOWN] Disposing database engines...")
    await dispose_engines()
    print("[SHUTDOWN] Complete.")


@app.get("/")
async def root():
    return {"message": "Event Scout Intelligence API", "version": "3.4.0"}


@app.get("/health/")
async def health_check(debug: bool = Query(False, description="Include debug information")):
    """Health check with graceful degradation. Returns partial status even if DB is unavailable."""
    health_status = {
        "status": "healthy",
        "database": "unknown",
        "database_connected": False,
        "total_users": 0,
        "ai_configured": ai_configured,
        "ai": ai_client.status(),
        "webhook_configured": bool(WEBHOOK_URL),
        "version": "3.4.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    if debug:
        # public endpoint: say whether things are set up, never where the database lives
        db_url = os.environ.get("DATABASE_URL", "")
        health_status["debug"] = {
            "database_url_configured": bool(db_url),
            "async_database_url_configured": bool(ASYNC_DATABASE_URL),
            "session_factory_exists": get_session_factory() is not None,
            "engine_exists": get_engine() is not None,
        }
    
    # Check database configuration + a quick, time-boxed ping (never blocks the health check)
    if ASYNC_DATABASE_URL:
        is_sqlite = ASYNC_DATABASE_URL.startswith("sqlite")
        health_status["database"] = "sqlite" if is_sqlite else "postgresql"
        factory = get_session_factory()
        if factory is not None:
            async def _ping():
                async with factory() as session:
                    await session.execute(select(1))
            try:
                await asyncio.wait_for(_ping(), timeout=2.0)
                health_status["database_connected"] = True
            except Exception:
                health_status["database_connected"] = False
                health_status["database_note"] = "Database did not answer within 2 s"
        if is_sqlite:
            health_status["database_network"] = "local"
        elif "railway.internal" in ASYNC_DATABASE_URL:
            health_status["database_network"] = "internal"
        elif "proxy.rlwy.net" in ASYNC_DATABASE_URL:
            health_status["database_network"] = "public_proxy"
        else:
            health_status["database_network"] = "external"
    else:
        health_status["status"] = "degraded"
        health_status["database"] = "not_configured"
        health_status["database_error"] = "DATABASE_URL not set"
    
    return health_status






# --- AUTH RATE LIMITS (per process; the app runs as one worker) ---
_AUTH_HITS: Dict[str, List[float]] = {}


def _client_ip(request: Request) -> str:
    # uvicorn --proxy-headers already put the real client (X-Forwarded-For) in request.client
    return (request.client.host if request.client else "") or "unknown"


def _rate_limit(key: str, limit: int, window_s: int) -> None:
    """429 once `key` was seen `limit` times in the last `window_s` seconds (password guessing,
    sign-up spam, reset-email floods). AUTH_RATE_LIMITS=off disables it (automated test runs)."""
    if os.environ.get("AUTH_RATE_LIMITS", "on").lower() == "off":
        return
    import time
    now = time.monotonic()
    hits = [t for t in _AUTH_HITS.get(key, []) if now - t < window_s]
    if len(hits) >= limit:
        raise HTTPException(status_code=429, detail="Too many attempts. Please wait a few minutes and try again.")
    hits.append(now)
    _AUTH_HITS[key] = hits
    if len(_AUTH_HITS) > 20000:  # forget stale keys so the table can't grow without bound
        for k in [k for k, v in _AUTH_HITS.items() if not v or now - v[-1] > 3600]:
            _AUTH_HITS.pop(k, None)


@app.post("/register/")
async def register_user(user: UserRegister, request: Request):
    _rate_limit(f"register:{_client_ip(request)}", 10, 3600)
    if len(user.password) < PASSWORD_MIN_LENGTH:
        raise HTTPException(status_code=400, detail=f"Password must be at least {PASSWORD_MIN_LENGTH} characters")
    if not user.name.strip():
        raise HTTPException(status_code=400, detail="Please enter your name")
    session = await get_db_session()
    try:
        # Check if email already exists (emails are case-insensitive; new accounts are stored lowercase)
        email = user.email.strip().lower()
        result = await session.execute(select(UserDB.id).where(func.lower(UserDB.email) == email).limit(1))
        if result.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Email already registered")

        user_id = uuid.uuid4()
        # Never admin at sign-up: the email isn't verified (see ADMIN_EMAILS / manage_admins.py).
        is_admin = False
        db_user = UserDB(
            id=user_id,
            name=user.name,
            email=email,
            password_hash=await asyncio.to_thread(hash_password, user.password),
            is_admin=is_admin,
        )
        session.add(db_user)
        await session.commit()

        # Initialize empty FAISS index for user
        faiss_index.build_for_user(str(user_id), [])

        print(f"[USER] Created new user: {email} with ID: {user_id}")
        await _api_routes_live("user_created", {"userId": str(user_id), "name": user.name})
        return {
            "status": "success",
            "message": "User registered successfully",
            "user_id": str(user_id),
            "name": user.name,
            "email": email,
            "is_admin": is_admin,
            **issue_auth_tokens(str(user_id), email, is_admin),
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.post("/login/")
async def login_user(credentials: UserLogin, request: Request):
    ip = _client_ip(request)
    _rate_limit(f"login:{ip}:{credentials.email.strip().lower()}", 10, 300)  # guessing one account's password
    _rate_limit(f"login:{ip}", 60, 300)                                      # sweeping many accounts
    session = await get_db_session()
    try:
        # Exact match first, then case-insensitive (the apps lowercase what people type; older
        # accounts may have been stored with capitals).
        email = credentials.email.strip()
        result = await session.execute(select(UserDB).where(UserDB.email == email))
        user = result.scalar_one_or_none()
        if not user:
            result = await session.execute(
                select(UserDB).where(func.lower(UserDB.email) == email.lower()).order_by(UserDB.created_at).limit(1)
            )
            user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        if not await asyncio.to_thread(verify_password, credentials.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        is_admin = bool(user.is_admin)

        return {
            "status": "success",
            "message": "Login successful",
            "user_id": str(user.id),
            "name": user.name,
            "email": user.email,
            "is_admin": is_admin,
            **issue_auth_tokens(str(user.id), user.email, is_admin, user.token_version),
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


# --- LINKEDIN OAUTH ENDPOINTS ---
@app.get("/auth/linkedin/url")
async def get_linkedin_auth_endpoint(state: Optional[str] = None, platform: Optional[str] = None):
    """Get LinkedIn OAuth authorization URL plus the redirect URI and state the client should use.
    platform=native: for the installed app (LinkedIn → /auth/linkedin/app-return → eventscout://callback)."""
    try:
        used_state = state or str(uuid.uuid4())
        if (platform or "").lower() == "native":
            url = await get_linkedin_auth_url(used_state, linkedin_redirect_for("native"))
            # the app watches for its own scheme; the API's return page forwards there
            return {"url": url, "redirectUri": LINKEDIN_APP_CALLBACK, "state": used_state}
        url = await get_linkedin_auth_url(used_state)
        return {"url": url, "redirectUri": LINKEDIN_REDIRECT_URI, "state": used_state}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/auth/linkedin/app-return", response_class=HTMLResponse)
async def linkedin_app_return(request: Request):
    """LinkedIn redirects here for the installed app (https is required); forward the result to the
    app's own scheme, where openAuthSessionAsync is waiting. Only code / state / error are passed on."""
    from urllib.parse import urlencode as _urlencode
    keep = {k: v for k, v in request.query_params.items() if k in ("code", "state", "error", "error_description")}
    target = f"{LINKEDIN_APP_CALLBACK}?{_urlencode(keep)}"
    safe = html_escape(target, quote=True)
    page = (f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>EventScout</title><meta http-equiv=refresh content='0;url={safe}'>"
            f"<p style='font-family:sans-serif;text-align:center;margin-top:30vh'>Returning to EventScout… "
            f"<a href='{safe}'>Open the app</a></p>")
    return HTMLResponse(page, status_code=200, headers={"Cache-Control": "no-store", "X-Robots-Tag": "noindex"})


@app.post("/auth/linkedin/callback")
async def linkedin_oauth_callback(auth: LinkedInAuth):
    """Exchange LinkedIn authorization code for user session."""
    session = await get_db_session()
    try:
        # Exchange code for LinkedIn profile
        linkedin_data = await exchange_linkedin_code(auth.code, linkedin_redirect_for(auth.platform))
        email = linkedin_data.get('email')
        linkedin_sub = (linkedin_data.get('profile') or {}).get('sub')

        if not email:
            raise HTTPException(status_code=400, detail="LinkedIn email not found")
        # Only trust the email (to join an existing account or to grant admin) when LinkedIn verified it.
        email_verified = (linkedin_data.get('profile') or {}).get('email_verified') is True

        # A LinkedIn identity connected to an account wins over an email match, so a
        # user who linked LinkedIn to their email/password account lands in that account.
        user = None
        if linkedin_sub:
            result = await session.execute(select(UserDB).where(UserDB.linkedin_sub == linkedin_sub))
            user = result.scalar_one_or_none()
        if not user:
            result = await session.execute(select(UserDB).where(func.lower(UserDB.email) == email.strip().lower()))
            user = result.scalar_one_or_none()
            if user and not email_verified:
                raise HTTPException(
                    status_code=409,
                    detail="An account with this email already exists. Sign in with your email and password, then connect LinkedIn from your profile.",
                )
            if user and linkedin_sub:
                if user.linkedin_sub and user.linkedin_sub != linkedin_sub:
                    raise HTTPException(
                        status_code=409,
                        detail="This account is connected to a different LinkedIn profile.",
                    )
                if not user.linkedin_sub and user.linkedin_disconnected_at:
                    # The user deliberately disconnected LinkedIn — don't re-link it behind their back.
                    raise HTTPException(
                        status_code=403,
                        detail="LinkedIn was disconnected from this account. Sign in with your email and password, then reconnect LinkedIn from your profile.",
                    )
                if not user.linkedin_sub:
                    # Same email as LinkedIn reports, so connect the identity on first LinkedIn sign-in.
                    user.linkedin_sub = linkedin_sub
                    user.linkedin_email = email
                    user.linkedin_linked_at = datetime.now(timezone.utc)
                    await session.commit()

        if user:
            # Existing user - login
            user_id = str(user.id)
            name = user.name
            email = user.email
            is_admin = user.is_admin
            message = "LinkedIn login successful"
        else:
            # New user - create account with LinkedIn data
            user_id = str(uuid.uuid4())
            name = linkedin_data.get('name', email.split('@')[0])
            is_admin = email_verified and email.strip().lower() in ADMIN_EMAILS
            
            db_user = UserDB(
                id=uuid.UUID(user_id),
                name=name,
                email=email,
                password_hash=hash_password(str(uuid.uuid4())),  # Random password for OAuth users
                is_admin=is_admin,
                linkedin_sub=linkedin_sub,
                linkedin_email=email,
                linkedin_linked_at=datetime.now(timezone.utc) if linkedin_sub else None,
                password_set=False,
            )
            session.add(db_user)
            await session.commit()
            
            # Initialize empty FAISS index for user
            faiss_index.build_for_user(user_id, [])
            
            message = "Account created via LinkedIn"
            await _api_routes_live("user_created", {"userId": user_id, "name": name})
            print(f"[LINKEDIN] Created new user: {email} with ID: {user_id} (admin={is_admin})")
        
        tokens = issue_auth_tokens(str(user_id), email, is_admin, user.token_version if user else 0)

        return {
            "status": "success",
            "message": message,
            **tokens,
            "user": {
                "id": user_id,
                "name": name,
                "email": email,
                "is_admin": is_admin,
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.get("/auth/linkedin/link")
async def get_linkedin_link_status(current_user_id: str = Depends(get_current_user_id)):
    """Whether the signed-in account has a LinkedIn identity connected."""
    session = await get_db_session()
    try:
        result = await session.execute(select(UserDB).where(UserDB.id == uuid.UUID(current_user_id)))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found. Please log in again.")
        return {
            "status": "success",
            "connected": bool(user.linkedin_sub),
            "linkedin_email": user.linkedin_email,
            "linked_at": user.linkedin_linked_at.isoformat() if user.linkedin_linked_at else None,
            "has_password": bool(user.password_set),
        }
    finally:
        await session.close()


@app.post("/auth/linkedin/link")
async def link_linkedin_account(auth: LinkedInAuth, current_user_id: str = Depends(get_current_user_id)):
    """Connect a LinkedIn identity to the signed-in account.

    Unlike /auth/linkedin/callback this never creates an account or switches the
    session: the caller stays signed in as themselves, and future LinkedIn sign-ins
    resolve to this account.
    """
    session = await get_db_session()
    try:
        result = await session.execute(select(UserDB).where(UserDB.id == uuid.UUID(current_user_id)))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found. Please log in again.")

        linkedin_data = await exchange_linkedin_code(auth.code, linkedin_redirect_for(auth.platform))
        linkedin_sub = (linkedin_data.get("profile") or {}).get("sub")
        linkedin_email = (linkedin_data.get("email") or "").strip()
        if not linkedin_sub:
            raise HTTPException(status_code=400, detail="LinkedIn did not return an account identifier")

        if user.linkedin_sub and user.linkedin_sub != linkedin_sub:
            raise HTTPException(status_code=409, detail="Your account is already connected to a different LinkedIn profile.")

        result = await session.execute(
            select(UserDB).where(UserDB.linkedin_sub == linkedin_sub, UserDB.id != user.id)
        )
        if result.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="This LinkedIn profile is already connected to another EventScout account.")

        # LinkedIn sign-in falls back to email matching, so a separate account owning the
        # LinkedIn email would make sign-in ambiguous — ask the user to use that account.
        if linkedin_email:
            result = await session.execute(
                select(UserDB).where(func.lower(UserDB.email) == linkedin_email.lower(), UserDB.id != user.id)
            )
            if result.scalar_one_or_none():
                raise HTTPException(
                    status_code=409,
                    detail=f"Another EventScout account already uses {linkedin_email}. Sign in to that account with LinkedIn instead.",
                )

        if not user.linkedin_sub:
            user.linkedin_sub = linkedin_sub
            user.linkedin_email = linkedin_email or None
            user.linkedin_linked_at = datetime.now(timezone.utc)
            user.linkedin_disconnected_at = None
            await session.commit()
            print(f"[LINKEDIN] Linked LinkedIn identity to user {user.id}")

        return {
            "status": "success",
            "message": "LinkedIn connected",
            "linkedin": {
                "name": linkedin_data.get("name") or "",
                "email": linkedin_email,
                "picture": linkedin_data.get("picture"),
            },
            "user": {"id": str(user.id), "name": user.name, "email": user.email},
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


PASSWORD_MIN_LENGTH = 8


class SetPasswordRequest(BaseModel):
    new_password: str


class LinkedInUnlinkRequest(BaseModel):
    password: str


@app.post("/auth/password/set")
async def set_account_password(body: SetPasswordRequest, current_user_id: str = Depends(get_current_user_id)):
    """Give a LinkedIn-created account its first password (so it can sign in without LinkedIn).

    Only allowed while the account has no password of its own; changing an existing
    password must go through a flow that verifies the current one.
    """
    if len(body.new_password) < PASSWORD_MIN_LENGTH:
        raise HTTPException(status_code=400, detail=f"Password must be at least {PASSWORD_MIN_LENGTH} characters")

    session = await get_db_session()
    try:
        result = await session.execute(select(UserDB).where(UserDB.id == uuid.UUID(current_user_id)))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found. Please log in again.")
        if user.password_set:
            raise HTTPException(status_code=409, detail="This account already has a password.")

        user.password_hash = await asyncio.to_thread(hash_password, body.new_password)
        user.password_set = True
        await session.commit()
        print(f"[AUTH] Password set for LinkedIn-created user {user.id}")
        return {"status": "success", "message": "Password set", "email": user.email}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.post("/auth/linkedin/unlink")
async def unlink_linkedin_account(body: LinkedInUnlinkRequest, current_user_id: str = Depends(get_current_user_id)):
    """Disconnect LinkedIn from the signed-in account.

    Requires the account password, which also guarantees the user can still sign in
    afterwards (LinkedIn-created accounts must call /auth/password/set first).
    """
    session = await get_db_session()
    try:
        result = await session.execute(select(UserDB).where(UserDB.id == uuid.UUID(current_user_id)))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found. Please log in again.")
        if not user.linkedin_sub:
            raise HTTPException(status_code=409, detail="LinkedIn is not connected to this account.")
        if not user.password_set:
            raise HTTPException(status_code=409, detail="Set a password before disconnecting LinkedIn, so you can still sign in.")
        if not await asyncio.to_thread(verify_password, body.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Incorrect password")

        user.linkedin_sub = None
        user.linkedin_email = None
        user.linkedin_linked_at = None
        user.linkedin_disconnected_at = datetime.now(timezone.utc)
        await session.commit()
        print(f"[LINKEDIN] Disconnected LinkedIn from user {user.id}")
        return {"status": "success", "message": "LinkedIn disconnected", "email": user.email}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


# --- PASSWORD RESET ---
RESET_CODE_TTL = timedelta(minutes=15)
RESET_RESEND_COOLDOWN = timedelta(seconds=60)
RESET_MAX_ATTEMPTS = 5
RESET_GENERIC_MESSAGE = "If an account exists for that email, we've sent a 6-digit reset code."


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    email: EmailStr
    code: str
    new_password: str


def _as_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """SQLite hands back naive datetimes even for timezone=True columns."""
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _hash_reset_code(user_id, code: str) -> str:
    # Keyed hash so a leaked database row can't be brute-forced offline without JWT_SECRET.
    return hmac.new(JWT_SECRET.encode("utf-8"), f"{user_id}:{code}".encode("utf-8"), hashlib.sha256).hexdigest()


def _smtp_tls_context() -> ssl.SSLContext:
    """TLS for the mail server: verified as usual, against SMTP_CA_FILE when set (a company server's
    own certificate), or unverified when SMTP_TLS_VERIFY=false."""
    context = ssl.create_default_context(cafile=SMTP_CA_FILE or None)
    if not SMTP_TLS_VERIFY:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    return context


def _send_email_sync(to_address: str, subject: str, text_body: str, html_body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM
    msg["To"] = to_address
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    context = _smtp_tls_context()
    if SMTP_USE_SSL:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context, timeout=20) as server:
            if SMTP_USERNAME:
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
    else:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.starttls(context=context)
            if SMTP_USERNAME:
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)


async def _deliver_reset_code(email: str, name: str, code: str) -> None:
    minutes = int(RESET_CODE_TTL.total_seconds() // 60)
    if not EMAIL_ENABLED:
        # Only reachable with PASSWORD_RESET_DEV_LOG=true (local development).
        print(f"[AUTH][DEV] Password reset code for {email}: {code} (expires in {minutes} min)")
        return
    greeting = f"Hi {name}," if name else "Hi,"
    text_body = (
        f"{greeting}\n\nYour EventScout password reset code is: {code}\n\n"
        f"It expires in {minutes} minutes. If you didn't ask to reset your password, you can ignore this email.\n"
    )
    html_body = (
        f"<p>{greeting}</p><p>Your EventScout password reset code is:</p>"
        f"<p style=\"font-size:28px;font-weight:700;letter-spacing:6px;font-family:monospace\">{code}</p>"
        f"<p>It expires in {minutes} minutes. If you didn't ask to reset your password, you can ignore this email.</p>"
    )
    await asyncio.to_thread(_send_email_sync, email, "Your EventScout password reset code", text_body, html_body)


@app.post("/auth/password/forgot")
async def request_password_reset(body: ForgotPasswordRequest, request: Request):
    _rate_limit(f"forgot:{_client_ip(request)}", 10, 900)
    """Email a one-time 6-digit code for resetting the password.

    Always answers with the same message whether or not the email has an account, so
    the endpoint can't be used to discover who is registered.
    """
    if not EMAIL_ENABLED and not PASSWORD_RESET_DEV_LOG:
        raise HTTPException(status_code=501, detail="Password reset by email is not configured on the server.")

    generic = {
        "status": "success",
        "message": RESET_GENERIC_MESSAGE,
        "expires_in": int(RESET_CODE_TTL.total_seconds()),
        "resend_after": int(RESET_RESEND_COOLDOWN.total_seconds()),
    }
    email = body.email.strip().lower()
    session = await get_db_session()
    try:
        result = await session.execute(select(UserDB).where(func.lower(UserDB.email) == email))
        user = result.scalar_one_or_none()
        if not user:
            return generic

        now = datetime.now(timezone.utc)
        result = await session.execute(
            select(PasswordResetDB)
            .where(PasswordResetDB.user_id == user.id)
            .order_by(PasswordResetDB.created_at.desc())
            .limit(1)
        )
        latest = result.scalar_one_or_none()
        if latest and latest.used_at is None and now - _as_utc(latest.created_at) < RESET_RESEND_COOLDOWN:
            # Throttle: a code was just sent; don't mail another yet.
            return generic

        # Only the newest code is ever valid.
        await session.execute(
            update(PasswordResetDB)
            .where(PasswordResetDB.user_id == user.id, PasswordResetDB.used_at.is_(None))
            .values(used_at=now)
        )
        code = f"{secrets.randbelow(10 ** 6):06d}"
        session.add(PasswordResetDB(
            user_id=user.id,
            code_hash=_hash_reset_code(user.id, code),
            expires_at=now + RESET_CODE_TTL,
            created_at=now,
        ))
        await session.commit()

        try:
            await _deliver_reset_code(user.email, user.name, code)
        except Exception as send_err:
            # Keep the response generic (no account enumeration); the failure is in the log.
            print(f"[AUTH] Failed to send password reset email to user {user.id}: {send_err}")
        return generic
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.post("/auth/password/reset")
async def reset_password_with_code(body: ResetPasswordRequest):
    """Set a new password using an emailed code, then sign the user in."""
    if len(body.new_password) < PASSWORD_MIN_LENGTH:
        raise HTTPException(status_code=400, detail=f"Password must be at least {PASSWORD_MIN_LENGTH} characters")
    code = body.code.strip()
    invalid = HTTPException(status_code=400, detail="This code is invalid or has expired. Request a new one.")
    if not (len(code) == 6 and code.isdigit()):
        raise invalid

    session = await get_db_session()
    try:
        result = await session.execute(select(UserDB).where(func.lower(UserDB.email) == body.email.strip().lower()))
        user = result.scalar_one_or_none()
        if not user:
            raise invalid

        now = datetime.now(timezone.utc)
        result = await session.execute(
            select(PasswordResetDB)
            .where(PasswordResetDB.user_id == user.id, PasswordResetDB.used_at.is_(None))
            .order_by(PasswordResetDB.created_at.desc())
            .limit(1)
        )
        reset = result.scalar_one_or_none()
        if not reset or _as_utc(reset.expires_at) < now:
            raise invalid

        if not hmac.compare_digest(reset.code_hash, _hash_reset_code(user.id, code)):
            reset.attempts = (reset.attempts or 0) + 1
            remaining = RESET_MAX_ATTEMPTS - reset.attempts
            if remaining <= 0:
                reset.used_at = now  # burn the code
            await session.commit()
            if remaining <= 0:
                raise HTTPException(status_code=429, detail="Too many incorrect attempts. Request a new code.")
            raise HTTPException(
                status_code=400,
                detail=f"Incorrect code. {remaining} attempt{'s' if remaining != 1 else ''} left.",
            )

        user.password_hash = await asyncio.to_thread(hash_password, body.new_password)
        user.password_set = True
        # Sign out every other device: tokens issued before this carry an older version.
        user.token_version = int(user.token_version or 0) + 1
        reset.used_at = now
        await session.commit()
        print(f"[AUTH] Password reset for user {user.id}; other sessions signed out")

        return {
            "status": "success",
            "message": "Password reset",
            "user_id": str(user.id),
            "name": user.name,
            "email": user.email,
            "is_admin": user.is_admin,
            "other_sessions_signed_out": True,
            # Fresh tokens at the new version keep this device signed in.
            **issue_auth_tokens(str(user.id), user.email, user.is_admin, user.token_version),
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.get("/auth/session")
async def check_session(authorization: Optional[str] = Header(None)):
    """Lightweight check the app runs on launch/resume.

    Answers 401 with SESSION_REVOKED_DETAIL if this device was signed out (e.g. by a
    password reset elsewhere). Expiry isn't enforced, matching the API-key routes.
    """
    uid = await identity.signed_in_user(authorization, verify_exp=False)
    # name / email / role as they are NOW, so an admin's rename reaches the app without a new sign-in
    session = await get_db_session()
    try:
        user = (await session.execute(select(UserDB).where(UserDB.id == uid))).scalar_one_or_none()
    finally:
        await session.close()
    return {
        "status": "success", "valid": True, "user_id": str(uid),
        "name": user.name if user else None,
        "email": user.email if user else None,
        "is_admin": bool(user.is_admin) if user else False,
    }


class RefreshRequest(BaseModel):
    refreshToken: str


@app.post("/auth/refresh")
async def refresh_auth_token(body: RefreshRequest):
    """Validate a refresh token and issue a fresh access + refresh token pair."""
    payload = decode_jwt_token(body.refreshToken)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Provided token is not a refresh token")

    user_id = payload.get("sub") or payload.get("userId")
    if not user_id:
        raise HTTPException(status_code=401, detail="Refresh token missing subject")

    session = await get_db_session()
    try:
        result = await session.execute(select(UserDB).where(UserDB.id == uuid.UUID(user_id)))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=401, detail=SESSION_REVOKED_DETAIL)  # account deleted
        if int(payload.get("tv") or 0) < int(user.token_version or 0):
            raise HTTPException(status_code=401, detail=SESSION_REVOKED_DETAIL)

        return {
            "status": "success",
            **issue_auth_tokens(str(user.id), user.email, user.is_admin, user.token_version),
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.post("/add_contact/")
async def add_contact_route(request: AddContactRequest, api_key: str = Depends(verify_api_key),
                            authorization: Optional[str] = Header(None)):
    request.user_id = await _chat_identity(request.user_id, authorization)
    try:
        result = await add_contact_logic(request.contact, request.user_id)
        return {
            "status": "success",
            "message": "Contact added",
            "contact_id": result["contact_id"],
            "qr_base64": result["qr_base64"],
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add_contact_from_image/")
async def add_contact_image_route(
    file: UploadFile = File(...),
    user_id: str = Query(..., description="User ID"),
    client_id: Optional[str] = Query(None, description="Phone-made UUID: retries return the saved contact instead of a second copy"),
    event_name: Optional[str] = Query(None, description="Event the scan was taken at (offline scans upload later)"),
    captured_at: Optional[str] = Query(None, description="When the scan was taken, ISO 8601"),
    api_key: str = Depends(verify_api_key),
    authorization: Optional[str] = Header(None),
):
    user_id = await _chat_identity(user_id, authorization)  # a token can only add to its own contacts
    try:
        return await add_contact_from_image(file, user_id, client_id, event_name, captured_at)
    except HTTPException:
        raise
    except TimeoutError as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=504,
            detail="AI processing timed out. The image might be too complex or the AI service is slow. Please try again."
        )
    except Exception as e:
        traceback.print_exc()
        error_msg = str(e)
        if "timeout" in error_msg.lower():
            raise HTTPException(
                status_code=504,
                detail="AI processing timed out. Please try again."
            )
        raise HTTPException(status_code=500, detail=f"Failed to process card: {error_msg}")


@app.post("/scan_qr/")
async def scan_qr_route(
    file: UploadFile = File(...),
    user_id: str = Query(..., description="User ID"),
    client_id: Optional[str] = Query(None, description="Phone-made UUID: retries return the saved contact instead of a second copy"),
    event_name: Optional[str] = Query(None, description="Event the scan was taken at (offline scans upload later)"),
    captured_at: Optional[str] = Query(None, description="When the scan was taken, ISO 8601"),
    api_key: str = Depends(verify_api_key),
    authorization: Optional[str] = Header(None),
):
    user_id = await _chat_identity(user_id, authorization)
    return await add_contact_from_qr(file, user_id, client_id, event_name, captured_at)




@app.post("/search/")
async def search_route(query: SearchQuery, api_key: str = Depends(verify_api_key),
                       authorization: Optional[str] = Header(None)):
    query.user_id = await _chat_identity(query.user_id, authorization)
    try:
        results = faiss_index.search(query.user_id, query.query, k=4)
        return {"status": "success", "results": [{"text": r[0], "meta": r[1]} for r in results]}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))










async def _chat_identity(query_user_id: str, authorization: Optional[str]) -> str:
    """The user a per-user legacy route (`?user_id=` + API key) acts for (identity.legacy_user: the token's
    own account only — 401 without a usable token, 403 for anyone else's id, admins included)."""
    await identity.legacy_user(authorization, query_user_id)
    return str(query_user_id)


CHAT_ATTACHMENT_LIMIT = 3
CHAT_ATTACHMENT_MAX_BYTES = 10 * 1024 * 1024
_CHAT_BINARY_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/heic", "image/heif", "application/pdf",
}
_CHAT_TEXT_TYPES = {
    "text/plain", "text/csv", "text/markdown", "application/json", "text/tab-separated-values",
}
_CHAT_EXT_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
    ".heic": "image/heic", ".heif": "image/heif", ".pdf": "application/pdf",
    ".txt": "text/plain", ".csv": "text/csv", ".md": "text/markdown", ".json": "application/json",
    ".tsv": "text/tab-separated-values", ".vcf": "text/plain",
}


async def read_chat_attachments(files: List[UploadFile]) -> list:
    """Validate uploaded chat files and turn them into parts for the AI call."""
    if len(files) > CHAT_ATTACHMENT_LIMIT:
        raise HTTPException(status_code=400, detail=f"Attach up to {CHAT_ATTACHMENT_LIMIT} files per question")
    attachments = []
    for f in files:
        name = os.path.basename(f.filename or "file")[:120]
        ext = os.path.splitext(name)[1].lower()
        mime = (f.content_type or "").split(";")[0].strip().lower()
        if mime not in _CHAT_BINARY_TYPES and mime not in _CHAT_TEXT_TYPES:
            mime = _CHAT_EXT_TYPES.get(ext, mime)  # pickers often send application/octet-stream
        if mime == "image/jpg":
            mime = "image/jpeg"
        if mime not in _CHAT_BINARY_TYPES and mime not in _CHAT_TEXT_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"\"{name}\" isn't a supported file. Attach an image, PDF, or text/CSV file.",
            )
        data = await f.read(CHAT_ATTACHMENT_MAX_BYTES + 1)
        if len(data) > CHAT_ATTACHMENT_MAX_BYTES:
            raise HTTPException(status_code=413, detail=f"\"{name}\" is larger than 10 MB")
        if not data:
            raise HTTPException(status_code=400, detail=f"\"{name}\" is empty")
        if mime in _CHAT_TEXT_TYPES:
            text = data.decode("utf-8", errors="replace")
            attachments.append({"name": name, "mime_type": mime, "text": text[:60000]})
        else:
            attachments.append({"name": name, "mime_type": mime, "data": data})
    return attachments


async def prepare_converse(query: ConverseQuery, attachments: Optional[list] = None, workspace: bool = False) -> dict:
    """Everything the assistant needs before the AI answers: FAISS retrieval over the user's
    contacts + profile/event context. Used by run_converse and the streaming routes."""
    # Admin mode (cross-user search) only for real admins
    # workspace=True: the admin console's assistant (all users' contacts, the admin's own sales profile)
    is_admin_mode = workspace
    if query.admin_mode and not workspace:
        session_check = await get_db_session()
        try:
            admin_result = await session_check.execute(
                select(UserDB).where(UserDB.id == uuid.UUID(query.user_id))
            )
            admin_user = admin_result.scalar_one_or_none()
            is_admin_mode = bool(admin_user and admin_user.is_admin)
        finally:
            await session_check.close()

    # FAISS search — cross-user for admin mode
    if is_admin_mode:
        # best matches across every rep (ranked by similarity, not by which rep's index came first)
        top_k = query.top_k or 8
        all_results = []
        q_vec = get_embedding(query.query).astype("float32").reshape(1, -1)
        for uid in list(faiss_index.indices.keys()):
            try:
                all_results.extend(faiss_index.search_scored(uid, query.query, k=top_k, q_vec=q_vec))
            except Exception:
                pass
        all_results.sort(key=lambda r: r[0], reverse=True)
        retrieved_contacts = [(text, meta) for _, text, meta in all_results[:top_k]]
    else:
        try:
            retrieved_contacts = faiss_index.search(query.user_id, query.query, k=query.top_k or 4)
        except Exception as search_err:
            print(f"[CHAT] FAISS search failed for {query.user_id}: {search_err}")
            retrieved_contacts = []

    session = await get_db_session()
    try:
        user_profile = {}
        if workspace:
            # no rep's identity: only the admin console's own profile
            user_profile = await admin_ai_profile(session)
        else:
            result = await session.execute(
                select(UserProfileDB).where(UserProfileDB.user_id == uuid.UUID(query.user_id))
            )
            profile_row = result.scalar_one_or_none()
            user_profile = event_identity.read_profile(profile_row)

        if is_admin_mode:
            all_contacts_result = await session.execute(select(ContactDB))
        else:
            all_contacts_result = await session.execute(
                select(ContactDB).where(ContactDB.user_id == uuid.UUID(query.user_id))
            )
        # The console's assistant ranks by the admin's own score, never the rep's (admin/rep isolation).
        admin_scores = {}
        if workspace:
            rows = await session.execute(
                select(AdminContactScoreDB.contact_id, AdminContactScoreDB.score, AdminContactScoreDB.temperature)
            )
            admin_scores = {str(cid).replace("-", ""): (s, t) for cid, s, t in rows.all()}
        all_contacts = []
        for c in all_contacts_result.scalars().all():
            score, temp = admin_scores.get(str(c.id).replace("-", ""), (None, None)) if workspace else (c.lead_score, c.lead_temperature)
            all_contacts.append(contact_view.chat_summary(c, score, temp))

        # Exhibitors of the user's active event (none if they haven't picked one — no default show)
        exhibitors = []
        event_name = (user_profile.get("current_event_name") or "").strip()
        if event_name:
            try:
                ex_result = await session.execute(
                    select(ExhibitorDB).where(await event_identity.event_filter(session, ExhibitorDB.event_name, event_name)).limit(50)
                )
                exhibitors = [
                    {
                        "name": e.name, "booth": e.booth or "", "hall": e.hall or "",
                        "category": e.category or "", "country": e.country or "",
                        "description": e.description or "",
                    }
                    for e in ex_result.scalars().all()
                ]
            except Exception as ex_err:
                print(f"[CHAT] Error loading exhibitors: {ex_err}")
    finally:
        await session.close()

    intent = detect_intent(query.query)
    augmented_query = query.query
    if intent == 'research':
        augmented_query = query.query + "\n\n" + RESEARCH_PROMPT_SUPPLEMENT
    elif intent == 'pitch':
        augmented_query = query.query + "\n\n" + PITCH_PROMPT_SUPPLEMENT
    if attachments:
        names = ", ".join(a["name"] for a in attachments)
        augmented_query += f"\n\n(The user attached: {names}. Use their contents to answer.)"

    history = [
        {"role": m.get("role", "user"), "content": (m.get("content") or "")[:4000]}
        for m in (query.conversation_history or [])[-12:]
        if m.get("content")
    ]

    if workspace:
        # the rep's score / reasoning stays out of the admin assistant's context and contact cards
        swapped = []
        for text, meta in retrieved_contacts:
            score, temp = admin_scores.get(str(meta.get("id", "")).replace("-", ""), (None, None))
            swapped.append((text, {**meta, "lead_score": score, "lead_temperature": temp,
                                   "lead_score_reasoning": "", "lead_recommended_actions": []}))
        retrieved_contacts = swapped

    enriched_contacts = [contact_view.chat_card(meta) for _, meta in retrieved_contacts]

    return {
        "query": query, "attachments": attachments, "intent": intent,
        "enriched_contacts": enriched_contacts, "all_contacts": all_contacts,
        "exhibitors": exhibitors, "user_profile": user_profile,
        # intelligence_engine.generate_response / stream_response arguments
        "engine_args": dict(
            query=augmented_query,
            retrieved_contacts=retrieved_contacts,
            conversation_history=history,
            user_profile=user_profile,
            all_contacts=all_contacts,
            exhibitors=exhibitors,
            attachments=attachments,
            ui_format=query.ui_format,
        ),
    }


def _converse_format(query: ConverseQuery) -> str:
    # "openui": `response` may contain ```openui-lang blocks between plain text
    return "openui" if (query.ui_format == "openui" and OPENUI_PROMPT) else "text"


def finish_converse(ctx: dict, response_text: str, model_label: str) -> dict:
    """The /converse/ response for an answer: the text plus contacts and UI components."""
    query, intent, enriched_contacts = ctx["query"], ctx["intent"], ctx["enriched_contacts"]
    ui_components = generate_ui_components(
        query=query.query, response_text=response_text,
        retrieved_contacts=enriched_contacts,
        all_contacts=ctx["all_contacts"],
        exhibitors=ctx["exhibitors"], user_profile=ctx["user_profile"],
    )
    if intent == 'research' and enriched_contacts:
        ui_components.insert(0, {
            "type": "research_card",
            "data": {
                "contact_id": enriched_contacts[0].get("id", ""),
                "contact_name": enriched_contacts[0].get("name", ""),
                "company": enriched_contacts[0].get("company_name", ""),
                "saved": False,
            }
        })
    elif intent == 'pitch' and enriched_contacts:
        ui_components.insert(0, {
            "type": "pitch_preview",
            "data": {
                "contact_id": enriched_contacts[0].get("id", ""),
                "contact_name": enriched_contacts[0].get("name", ""),
                "company": enriched_contacts[0].get("company_name", ""),
            }
        })

    return {
        "status": "success",
        "response": response_text,
        "format": _converse_format(query),
        "model": model_label,  # the model that actually answered
        "retrieved_contacts": enriched_contacts,
        "ui_components": ui_components,
        "intent": intent,
        "query": query.query,
        "attachments": [{"name": a["name"], "mime_type": a["mime_type"]} for a in ctx["attachments"] or []],
    }


async def run_converse(query: ConverseQuery, attachments: Optional[list] = None, workspace: bool = False) -> dict:
    """The assistant: FAISS retrieval over the user's contacts + profile/event context → AI answer."""
    ctx = await prepare_converse(query, attachments, workspace)
    try:
        response_text, label = await intelligence_engine.generate_response(**ctx["engine_args"])
    except AIUnavailable as e:
        print(f"[CHAT] AI unavailable ({e.reason}): {e}")
        raise HTTPException(status_code=503, detail=AI_UNAVAILABLE_MESSAGES.get(e.reason, AI_UNAVAILABLE_MESSAGES["unavailable"]))
    return finish_converse(ctx, response_text, label)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def converse_stream_response(ctx: dict):
    """The answer as Server-Sent Events:
      meta  {format, retrieved_contacts}  before the AI starts (contact cards can link right away)
      model {model}                       the model that's answering
      delta {text}                        the next piece of the answer
      done  <the /converse/ response>     the full answer, contacts, UI components
      error {detail}                      no answer (or the answer broke off) — user-facing message
    """
    from fastapi.responses import StreamingResponse

    async def events():
        # Padding comment: some proxies / iOS hold back the first bytes of a response.
        yield ":" + " " * 2048 + "\n\n"
        yield _sse("meta", {
            "format": _converse_format(ctx["query"]),
            "retrieved_contacts": [{k: v for k, v in c.items() if k != "photo_base64"} for c in ctx["enriched_contacts"]],
        })
        parts: List[str] = []
        label = ""
        try:
            async for kind, value in intelligence_engine.stream_response(**ctx["engine_args"]):
                if kind == "model":
                    label = value
                    yield _sse("model", {"model": value})
                else:
                    parts.append(value)
                    yield _sse("delta", {"text": value})
            yield _sse("done", finish_converse(ctx, "".join(parts).strip(), label))
        except AIUnavailable as e:
            print(f"[CHAT] AI unavailable while streaming ({e.reason}): {e}")
            yield _sse("error", {"detail": AI_UNAVAILABLE_MESSAGES.get(e.reason, AI_UNAVAILABLE_MESSAGES["unavailable"])})
        except Exception as e:
            traceback.print_exc()
            yield _sse("error", {"detail": f"Conversation error: {str(e)}"})

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Content-Type-Options": "nosniff",
    })


@app.post("/converse/")
async def converse_route(
    query: ConverseQuery,
    api_key: str = Depends(verify_api_key),
    authorization: Optional[str] = Header(None),
):
    if not query.query.strip():
        raise HTTPException(status_code=400, detail="Ask a question first")
    query.user_id = await _chat_identity(query.user_id, authorization)
    try:
        return await run_converse(query)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Conversation error: {str(e)}")


@app.post("/converse/stream/")
async def converse_stream_route(
    query: ConverseQuery,
    api_key: str = Depends(verify_api_key),
    authorization: Optional[str] = Header(None),
):
    """/converse/, streamed as Server-Sent Events so the app can show the answer as it's written."""
    if not query.query.strip():
        raise HTTPException(status_code=400, detail="Ask a question first")
    query.user_id = await _chat_identity(query.user_id, authorization)
    try:
        ctx = await prepare_converse(query)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Conversation error: {str(e)}")
    return converse_stream_response(ctx)


async def _files_converse_query(query: str, user_id: str, conversation_history: Optional[str],
                                ui_format: Optional[str], files: List[UploadFile]) -> Tuple[ConverseQuery, list]:
    """The multipart /converse/with_files/ form → (ConverseQuery, attachments). user_id must be checked already."""
    history = []
    if conversation_history:
        try:
            parsed = json.loads(conversation_history)
            if isinstance(parsed, list):
                history = [
                    {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
                    for m in parsed if isinstance(m, dict)
                ]
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="conversation_history must be a JSON list")
    attachments = await read_chat_attachments(files)
    question = query.strip() or "Summarize the attached file(s) and tell me anything useful for my event."
    return ConverseQuery(query=question, user_id=user_id, conversation_history=history, ui_format=ui_format), attachments


@app.post("/converse/with_files/")
async def converse_with_files_route(
    query: str = Form(""),
    user_id: str = Form(...),
    conversation_history: Optional[str] = Form(None),
    ui_format: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    api_key: str = Depends(verify_api_key),
    authorization: Optional[str] = Header(None),
):
    """Ask the assistant about attached files (images, PDFs, text/CSV) plus your contacts."""
    user_id = await _chat_identity(user_id, authorization)
    cq, attachments = await _files_converse_query(query, user_id, conversation_history, ui_format, files)
    try:
        return await run_converse(cq, attachments=attachments)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Conversation error: {str(e)}")


@app.post("/converse/with_files/stream/")
async def converse_with_files_stream_route(
    query: str = Form(""),
    user_id: str = Form(...),
    conversation_history: Optional[str] = Form(None),
    ui_format: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    api_key: str = Depends(verify_api_key),
    authorization: Optional[str] = Header(None),
):
    """/converse/with_files/, streamed as Server-Sent Events (see converse_stream_response)."""
    user_id = await _chat_identity(user_id, authorization)
    cq, attachments = await _files_converse_query(query, user_id, conversation_history, ui_format, files)
    try:
        ctx = await prepare_converse(cq, attachments=attachments)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Conversation error: {str(e)}")
    return converse_stream_response(ctx)


@app.get("/export_contacts/")
async def export_contacts_route(
    user_id: str = Query(..., description="User ID"),
    format: str = Query("csv", description="Export format: csv or json"),
    api_key: str = Depends(verify_api_key),
    authorization: Optional[str] = Header(None),
):
    user_id = await _chat_identity(user_id, authorization)  # a token can only export its own contacts
    import csv
    from io import StringIO
    from fastapi.responses import StreamingResponse

    format = (format or "csv").strip().lower()
    if format not in ("csv", "json"):
        raise HTTPException(status_code=400, detail="Export format must be csv or json")

    session = await get_db_session()
    try:
        result = await session.execute(
            select(ContactDB).where(ContactDB.user_id == uuid.UUID(user_id)).order_by(ContactDB.created_at.desc())
        )
        contacts = result.scalars().all()

        # Get pipeline data for research/pitch columns
        contact_ids = [c.id for c in contacts]
        pipeline_map = {}
        if contact_ids:
            pip_result = await session.execute(
                select(ContactPipelineDB)
                .where(ContactPipelineDB.contact_id.in_(contact_ids),
                       ContactPipelineDB.admin_run == False)  # noqa: E712 — admin console runs stay admin-only
                .order_by(ContactPipelineDB.created_at.desc())
            )
            for p in pip_result.scalars().all():
                cid = str(p.contact_id)
                if cid not in pipeline_map:
                    pipeline_map[cid] = p

        if format == "json":
            contact_list = [contact_view.export_view(c, pipeline_map.get(str(c.id))) for c in contacts]
            return {"status": "success", "format": "eventscout.contacts.v1", "exported_at": datetime.now(timezone.utc).isoformat(),
                    "total": len(contact_list), "contacts": contact_list}

        # CSV export
        output = StringIO()
        writer = safe_csv_writer(output)  # formula-injection safe (sanitize.py)
        writer.writerow(contact_view.EXPORT_CSV_HEADER)
        for c in contacts:
            writer.writerow(contact_view.export_csv_row(c, pipeline_map.get(str(c.id))))

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=event_scout_contacts.csv"},
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


async def verify_enrich_webhook(x_webhook_secret: Optional[str] = Header(None)) -> None:
    """The n8n enrichment callback writes into any user's contact, so it needs its own secret
    (ENRICH_WEBHOOK_SECRET, sent as X-Webhook-Secret) — not the API key that ships in the app."""
    expected = os.environ.get("ENRICH_WEBHOOK_SECRET", "")
    if not expected:
        raise HTTPException(status_code=503, detail="The enrichment webhook isn't configured (set ENRICH_WEBHOOK_SECRET).")
    if not x_webhook_secret or not hmac.compare_digest(x_webhook_secret, expected):
        raise HTTPException(status_code=401, detail="Invalid webhook secret")


@app.post("/contact/{contact_id}/enrich")
async def enrich_contact_route(
    contact_id: str,
    enrich_data: EnrichRequest,
    user_id: str = Query(None, description="User ID (optional, searches all users)"),
    _webhook: None = Depends(verify_enrich_webhook),
):
    session = await get_db_session()
    try:
        if user_id:
            result = await session.execute(
                select(ContactDB).where(ContactDB.id == uuid.UUID(contact_id), ContactDB.user_id == uuid.UUID(user_id))
            )
        else:
            result = await session.execute(
                select(ContactDB).where(ContactDB.id == uuid.UUID(contact_id))
            )
        c = result.scalar_one_or_none()
        if not c:
            raise HTTPException(status_code=404, detail="Contact not found")

        # Appended notes and links go through the contact store (tolerant: notes over the limit are
        # shortened, never refused — the workflow can't fix them).
        fields = {}
        if enrich_data.notes_append:
            existing_notes = c.notes or ""
            fields["notes"] = f"{existing_notes}\n\n---\n{enrich_data.notes_append}" if existing_notes else enrich_data.notes_append
        if enrich_data.links:
            fields["links"] = list(c.links or []) + [{**link, "added_by": link.get("added_by", "n8n")} for link in enrich_data.links]
        await contact_store.update_contact(session, c, fields, strict=False)

        # Update shared_contacts too
        shared_result = await session.execute(
            select(SharedContactDB).where(SharedContactDB.original_contact_id == uuid.UUID(contact_id))
        )
        shared = shared_result.scalar_one_or_none()
        if shared:
            shared.notes = c.notes
            shared.links = c.links
            shared.enriched = True
            shared.enriched_at = datetime.now(timezone.utc)
            await session.commit()

        return {
            "status": "success",
            "message": "Contact enriched successfully",
            "contact_id": contact_id,
            "notes": c.notes or "",
            "links": c.links or [],
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


# --- USER PROFILE ENDPOINTS ---

@app.get("/user/profile/")
async def get_user_profile_route(
    user_id: str = Query(..., description="User ID"),
    api_key: str = Depends(verify_api_key),
    authorization: Optional[str] = Header(None),
):
    user_id = await _chat_identity(user_id, authorization)  # a token can only read its own profile
    session = await get_db_session()
    try:
        result = await session.execute(
            select(UserProfileDB).where(UserProfileDB.user_id == uuid.UUID(user_id))
        )
        profile = result.scalar_one_or_none()
        return {"status": "success", "profile": event_identity.read_profile(profile)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.put("/user/profile/")
async def update_user_profile_route(
    profile_update: UserProfileUpdate,
    user_id: str = Query(..., description="User ID"),
    api_key: str = Depends(verify_api_key),
    authorization: Optional[str] = Header(None),
):
    user_id = await _chat_identity(user_id, authorization)  # a token can only edit its own profile
    session = await get_db_session()
    try:
        # Validate user exists first
        user_result = await session.execute(
            select(UserDB).where(UserDB.id == uuid.UUID(user_id))
        )
        user = user_result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail=f"User not found. Please log in again.")

        result = await session.execute(
            select(UserProfileDB).where(UserProfileDB.user_id == uuid.UUID(user_id))
        )
        profile = result.scalar_one_or_none()

        update_data = profile_update.model_dump(exclude_unset=True)
        if not update_data:
            raise HTTPException(status_code=400, detail="No fields provided for update")
        # Profile photo: validated + shrunk to a ≤512px JPEG (same as contact photos); "" removes it.
        remove_avatar = False
        if "avatar" in update_data:
            raw = update_data.pop("avatar")
            if raw:
                update_data["avatar"] = "data:image/jpeg;base64," + contact_store.process_contact_photo(raw)
            else:
                remove_avatar = True
        existing = event_identity.read_profile(profile)
        event_identity.normalize_profile_events(update_data, existing)
        normalize_profile_sales(update_data)
        merged = dict(existing)
        for key, value in update_data.items():
            if value is not None:
                merged[key] = value
        if remove_avatar:
            merged.pop("avatar", None)
        event_identity.sync_event_context(merged, update_data, existing)
        profile = event_identity.write_profile(session, profile, uuid.UUID(user_id), merged)
        profile.updated_at = datetime.now(timezone.utc)
        # One name everywhere: the profile's full name is the account name the console shows.
        new_name = (update_data.get("full_name") or "").strip()
        if new_name and new_name != user.name:
            user.name = new_name[:255]
        await session.commit()
        await _api_routes_live("user_updated", {"userId": user_id, "name": user.name})

        return {"status": "success", "profile": profile.profile_data}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.get("/dashboard")
async def get_dashboard(
    user_id: str = Query(..., description="User ID"),
    # The phone's Date.getTimezoneOffset() (minutes, UTC − local; India = -330). "Met today" is
    # the user's local day — it used to be the UTC day, so early-morning contacts counted as yesterday.
    tz_offset: int = Query(0, ge=-900, le=900),
    api_key: str = Depends(verify_api_key),
    authorization: Optional[str] = Header(None),
):
    user_id = await _chat_identity(user_id, authorization)  # a token can only read its own stats
    session = await get_db_session()
    try:
        # Get all contacts for this user
        result = await session.execute(
            select(ContactDB)
            .where(ContactDB.user_id == uuid.UUID(user_id))
        )
        contacts = result.scalars().all()

        total = len(contacts)
        scored = 0
        unscored = 0
        by_temperature = {"hot": 0, "warm": 0, "cool": 0, "cold": 0}  # score_levels (cool used to count as cold)
        by_source: Dict[str, int] = {}
        by_event: Dict[str, int] = {}
        now = datetime.now(timezone.utc)
        added_today = 0
        added_week = 0

        top_contacts = []

        for c in contacts:
            if c.lead_score is not None:  # a real 0 is still a score
                scored += 1
                by_temperature[score_levels.level_for(c.lead_score)] += 1
            else:
                unscored += 1
            src = c.source or "manual"
            by_source[src] = by_source.get(src, 0) + 1
            if c.event_name:
                by_event[c.event_name] = by_event.get(c.event_name, 0) + 1
            created = c.created_at if (c.created_at and c.created_at.tzinfo) else (c.created_at.replace(tzinfo=timezone.utc) if c.created_at else None)
            if created:
                if (created - timedelta(minutes=tz_offset)).date() == (now - timedelta(minutes=tz_offset)).date():
                    added_today += 1
                if now - created <= timedelta(days=7):
                    added_week += 1

        # Sort by lead_score to get top contacts
        sorted_contacts = sorted([c for c in contacts if c.lead_score is not None], key=lambda x: x.lead_score, reverse=True)
        for tc in sorted_contacts[:5]:
            top_contacts.append(contact_view.dashboard_top(tc))

        # Check if profile exists
        profile_result = await session.execute(
            select(UserProfileDB).where(UserProfileDB.user_id == uuid.UUID(user_id))
        )
        profile_row = profile_result.scalar_one_or_none()
        profile_data = event_identity.read_profile(profile_row)

        return {
            "status": "success",
            "total_contacts": total,
            "scored_contacts": scored,
            "unscored_contacts": unscored,
            "by_temperature": by_temperature,
            "by_source": by_source,
            "by_event": by_event,
            "added_today": added_today,
            "added_this_week": added_week,
            "top_contacts": top_contacts,
            "has_profile": profile_row is not None,
            # The event the user picked (was hardcoded "World Health Expo 2026")
            "current_event": profile_data.get("current_event_name") or None,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


# --- LEAD SCORING ---
# scoring.py scores (rep and admin); this server's AI call is its scorer.
scoring.ai_scorer = score_contact_with_ai
import api_routes as _api_routes_module  # noqa: E402


def _admin_converse_query(query: str, history: Optional[list], ui_format: Optional[str]) -> "ConverseQuery":
    # the whole workspace (every rep's contacts); no one user's identity
    return ConverseQuery(query=query, user_id=str(uuid.UUID(int=0)), conversation_history=history or [], top_k=8,
                         ui_format="openui" if ui_format == "openui" else None)


async def _admin_console_converse(query: str, history: Optional[list] = None, ui_format: Optional[str] = None) -> dict:
    return await run_converse(_admin_converse_query(query, history, ui_format), workspace=True)


async def _admin_console_converse_stream(query: str, history: Optional[list] = None, ui_format: Optional[str] = None,
                                         files: Optional[List[UploadFile]] = None):
    """The admin copilot as SSE (same events as /converse/stream/). `files`: attached images / PDFs / text
    (same rules as the app's /converse/with_files/: 3 files, 10 MB each)."""
    attachments = await read_chat_attachments(files) if files else None
    if attachments and not query.strip():
        query = "Summarize the attached file(s) and tell me anything useful for our events and leads."
    try:
        ctx = await prepare_converse(_admin_converse_query(query, history, ui_format), attachments, True)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Conversation error: {str(e)}")
    return converse_stream_response(ctx)


_api_routes_module.converse_impl = _admin_console_converse
_api_routes_module.converse_stream_impl = _admin_console_converse_stream
# admin voice questions (same model and rules as the app); looked up at call time — defined further down
_api_routes_module.transcribe_impl = lambda file: transcribe_upload(file)

# Admin console: the real pipeline, backup, password hashing, model list and webhook target.
_api_routes_module.pipeline_impl = run_contact_pipeline
_api_routes_module.pipeline_settings_impl = lambda: {
    "aiConfigured": bool(ai_client.ai_configured),
    "gammaConfigured": bool(GAMMA_API_KEY),
    "autoPipelineEnabled": AUTO_PIPELINE_ENABLED,
    # where the switch comes from on the next restart (the toggle itself is in memory)
    "autoPipelineFromEnv": os.environ.get("AUTO_PIPELINE_ENABLED", "false").lower() == "true",
}


def _set_auto_pipeline(enabled: bool) -> None:
    """Admin console switch: run the pipeline automatically after every card / QR scan (the rep's
    own pipeline, so it writes to the rep's contact). In memory until the server restarts."""
    global AUTO_PIPELINE_ENABLED
    AUTO_PIPELINE_ENABLED = bool(enabled)
    print(f"[PIPELINE] Auto-pipeline {'enabled' if AUTO_PIPELINE_ENABLED else 'disabled'} from the admin console")


_api_routes_module.pipeline_settings_update_impl = _set_auto_pipeline
_api_routes_module.backup_impl = run_backup
_api_routes_module.backup_configured_impl = lambda: bool(ASYNC_BACKUP_URL)
# the last backup to the backup database, whoever ran it (startup, the periodic loop, or the console)
_api_routes_module.last_backup_impl = lambda: _last_backup_result
# what the System page shows about this server (no secrets: flags, version, schedule)
_api_routes_module.system_info_impl = lambda: {
    "version": "3.4.0",
    "aiConfigured": bool(ai_client.ai_configured),
    "webhookConfigured": bool((WEBHOOK_URL or "").strip()),
    "backupIntervalHours": BACKUP_INTERVAL_HOURS if ASYNC_BACKUP_URL else None,
}
_api_routes_module.hash_password_impl = hash_password
# the admin's own account page (/api/admin/me): same password check, tokens and profile rules as the app
_api_routes_module.verify_password_impl = verify_password
_api_routes_module.normalize_profile_impl = normalize_profile_sales
_api_routes_module.webhook_url_impl = lambda: WEBHOOK_URL
_api_routes_module.delete_user_index_impl = lambda user_id: faiss_index.delete_user(user_id)


def _configured_models() -> list:
    """The OpenRouter models this server really calls, and for what."""
    roles = [("chat", ai_client.CHAT_MODEL, "Ask AI and the admin assistant"),
             ("default", ai_client.DEFAULT_MODEL, "Lead scoring, research and pitches"),
             ("vision", ai_client.VISION_MODEL, "Reading business cards"),
             ("audio", ai_client.AUDIO_MODEL, "Voice notes")]
    roles += [("fallback", m, "Used if the main model fails") for m in ai_client.FALLBACK_MODELS]
    out, seen = [], {}
    for role, mid, use in roles:
        if mid in seen:
            seen[mid]["uses"].append(use)
            continue
        provider = mid.split("/", 1)[0].replace("-", " ").title() if "/" in mid else "OpenRouter"
        seen[mid] = {"id": mid, "name": model_label(mid), "provider": provider, "role": role, "uses": [use],
                     "default": role == "chat", "configured": bool(ai_client.ai_configured)}
        out.append(seen[mid])
    return out


_api_routes_module.models_impl = _configured_models
def _index_contact_upsert(user_id: str, contact_dict: dict) -> None:
    # update_contact is a no-op for contacts not yet indexed (e.g. created via /api/contacts).
    if not faiss_index.update_contact(user_id, contact_dict["id"], contact_dict):
        faiss_index.add_contact(user_id, contact_dict)


# every contact write (contact_store) keeps the search index in step
contact_store.effects.index_update = _index_contact_upsert
contact_store.effects.index_delete = lambda user_id, contact_id: faiss_index.delete_contact(user_id, contact_id)




async def is_auto_score_enabled(user_id: str) -> bool:
    """True only if the user switched on "Auto-score on scan" (profile.auto_score_on_add).

    An unset preference means off, so scans never spend AI calls the user didn't opt into.
    Scoring also needs OpenRouter to be configured on the server.
    """
    if not ai_configured:
        return False
    session = await get_db_session()
    try:
        result = await session.execute(select(UserProfileDB).where(UserProfileDB.user_id == uuid.UUID(user_id)))
        profile = result.scalar_one_or_none()
        return event_identity.read_profile(profile).get("auto_score_on_add") is True
    except Exception as e:
        print(f"[AUTO-SCORE] Could not read preference for user {user_id}: {e}")
        return False
    finally:
        await session.close()


# asyncio only keeps weak references to tasks, so an un-referenced background task can be
# garbage-collected before it finishes. Hold them here until they're done.
_background_tasks: set = set()


def start_auto_score(contact_id: str, user_id: str) -> None:
    task = asyncio.create_task(auto_score_contact(contact_id, user_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def auto_score_contact(contact_id: str, user_id: str) -> None:
    """Background task: score a freshly scanned contact. Failures are logged, never raised."""
    try:
        result = await scoring.score(contact_id, "rep", owner=uuid.UUID(user_id))
        print(f"[AUTO-SCORE] Contact {contact_id} scored {result['score']} ({result['temperature']})")
        await _api_routes_live("score_updated", {
            "contactId": contact_id, "name": result.get("name"),
            "score": result["score"], "scoreBand": score_levels.band_for(result["score"]),
        })
    except HTTPException as e:
        print(f"[AUTO-SCORE] Contact {contact_id} not scored: {e.detail}")
    except Exception as e:
        print(f"[AUTO-SCORE] Contact {contact_id} not scored: {e}")



# --- VOICE QUESTIONS (transcription) ---

TRANSCRIBE_MAX_BYTES = 8 * 1024 * 1024
_AUDIO_EXT_TYPES = {
    ".m4a": "audio/mp4", ".mp4": "audio/mp4", ".aac": "audio/aac", ".3gp": "audio/3gpp",
    ".caf": "audio/x-caf", ".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg",
    ".webm": "audio/webm", ".flac": "audio/flac",
}
TRANSCRIBE_UNAVAILABLE_MESSAGES = {
    "bad_key": "Voice input is unavailable: the server's OpenRouter API key isn't working. Ask your admin to check it.",
    "no_credits": "Voice input is paused: the server's OpenRouter account is out of credits. Type your question instead.",
    "rate_limited": "Voice input is busy right now (AI rate limit). Please try again in a minute, or type your question.",
    "timeout": "Transcribing took too long. Please try again, or type your question.",
    "unavailable": "Couldn't transcribe that recording. Please try again, or type your question.",
}


@app.post("/transcribe_audio/")
async def transcribe_audio(
    file: UploadFile = File(...),
    user_id: str = Query(...),
    api_key: str = Depends(verify_api_key),
    authorization: Optional[str] = Header(None),
):
    """Transcribe a voice recording (OpenRouter audio input) — the mobile app's voice questions."""
    await _chat_identity(user_id, authorization)
    return {"status": "success", "transcript": await transcribe_upload(file)}


async def transcribe_upload(file: UploadFile) -> str:
    """A voice recording → the words spoken (OpenRouter audio input). Raises HTTPException with a
    message the person can act on. Used by /transcribe_audio/ (app) and /api/transcribe (admin)."""
    if not ai_configured:
        raise HTTPException(status_code=503, detail="Voice input isn't set up on the server (no OpenRouter API key).")

    audio_data = await file.read(TRANSCRIBE_MAX_BYTES + 1)
    if len(audio_data) > TRANSCRIBE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="That recording is too long. Keep voice questions under about 2 minutes.")
    if len(audio_data) < 1000:
        raise HTTPException(status_code=422, detail="That recording was too short — hold the mic and speak, then tap to send.")

    # Phones record .m4a (AAC in MP4) / .3gp / .caf; browsers webm. Pickers often say octet-stream.
    ext = os.path.splitext(file.filename or "")[1].lower()
    mime = (file.content_type or "").split(";")[0].strip().lower()
    if not mime.startswith("audio/") or mime in ("audio/x-m4a", "audio/m4a"):
        mime = _AUDIO_EXT_TYPES.get(ext, "audio/mp4" if mime in ("audio/x-m4a", "audio/m4a") else "audio/webm")

    prompt = (
        "Transcribe this voice recording of someone asking an event/sales assistant a question. "
        "Return ONLY the words spoken, in the language spoken, with normal punctuation. "
        "If there is no intelligible speech, return exactly: [no speech]"
    )
    # OpenRouter names audio formats by extension; phone recordings are AAC in an MP4 (.m4a)
    # container, which some providers call "aac" or "mp4" — try those in turn.
    fmt = (ext.lstrip(".") or mime.split("/")[-1]).replace("mpeg", "mp3").replace("x-caf", "caf")
    formats = {"m4a": ["m4a", "aac", "mp4"], "mp4": ["mp4", "m4a", "aac"], "3gp": ["3gp", "aac"], "caf": ["caf", "aac"]}.get(fmt, [fmt])
    transcript: Optional[str] = None
    last: Optional[AIError] = None
    for f in formats:
        try:
            transcript, used = await ai_client.chat(
                [{"role": "user", "content": [ai_client.text_part(prompt), ai_client.audio_part(audio_data, f)]}],
                model=ai_client.AUDIO_MODEL, max_tokens=800, temperature=0, timeout=40, title="Event Scout Voice",
            )
            print(f"[TRANSCRIBE] {used} ({f}) ok")
            break
        except AIError as e:
            last = e
            print(f"[TRANSCRIBE] format {f} failed ({e.reason}): {e}")
            if e.reason in ("not_configured", "bad_key", "no_credits", "rate_limited"):
                break

    if transcript is None:
        reason = last.reason if last else "unavailable"
        raise HTTPException(status_code=503, detail=TRANSCRIBE_UNAVAILABLE_MESSAGES.get(reason, TRANSCRIBE_UNAVAILABLE_MESSAGES["unavailable"]))
    if not transcript or transcript.strip("[] .").lower() == "no speech":
        raise HTTPException(status_code=422, detail="Didn't catch that — no speech was detected. Try again closer to the mic.")
    return transcript.strip()


# The old single-file-app routes (list_contacts, PUT/DELETE /contact/{id}, /admin/... and friends) were
# deleted: the mobile app and the admin console use the /api routes, which carry proper per-user / admin auth.


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8003))
    uvicorn.run(app, host="0.0.0.0", port=port)
