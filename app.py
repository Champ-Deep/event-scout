import os
import re
import json
import uuid
import asyncio
import qrcode
try:
    from pyzbar.pyzbar import decode as decode_qr
except ImportError:
    print("[WARNING] pyzbar not found. QR scanning will be disabled.")
    def decode_qr(image):
        return []
from PIL import Image
import base64
import traceback
import numpy as np
import faiss
from io import BytesIO
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

from sentence_transformers import SentenceTransformer

import bcrypt

import google.generativeai as genai

from sqlalchemy import select, update, delete, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from database import (
    UserDB, ContactDB, SharedContactDB, UserProfileDB, ConversationDB, ExhibitorDB, UserCardDB, EventFileDB,
    ContactFileDB, ContactPipelineDB, AdminBroadcastDB, Base,
    get_engine, get_session_factory, get_backup_session_factory, init_db, dispose_engines,
    ASYNC_DATABASE_URL, ASYNC_BACKUP_URL
)

import httpx

# --- CONFIG ---
APP_API_KEY = os.environ.get("APP_API_KEY", "1234")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")  # n8n webhook endpoint
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
ADMIN_EMAILS = [e.strip().lower() for e in os.environ.get("ADMIN_EMAILS", "deep@lakeb2b.com").split(",") if e.strip()]

# Pipeline config
PRESENTON_API_URL = os.environ.get("PRESENTON_API_URL", "")  # Self-hosted Presenton instance
PRESENTON_API_KEY = os.environ.get("PRESENTON_API_KEY", "")
AUTO_PIPELINE_ENABLED = os.environ.get("AUTO_PIPELINE_ENABLED", "false").lower() == "true"

# Available AI models for chat (all via OpenRouter)
AVAILABLE_MODELS = {
    "claude-opus": {"id": "anthropic/claude-opus-4", "name": "Claude Opus 4.6", "provider": "Anthropic"},
    "gpt-5": {"id": "openai/gpt-4.1", "name": "GPT-5.2", "provider": "OpenAI"},
    "gemini-pro": {"id": "google/gemini-2.5-pro-preview-06-05", "name": "Gemini 2.5 Pro", "provider": "Google"},
}

BASE_DIR = os.getcwd()
TEMP_DIR = os.path.join(BASE_DIR, "temp_images")
QR_DIR = os.path.join(BASE_DIR, "saved_qr")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(QR_DIR, exist_ok=True)

# --- Configure Gemini ---
gemini_configured = False
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_configured = True
        print("[INIT] Gemini API configured successfully")
    except Exception as e:
        print(f"[INIT] Failed to configure Gemini: {e}")
else:
    print("[INIT] WARNING: GEMINI_API_KEY not set")


# --- Pydantic models ---
class Contact(BaseModel):
    name: str
    email: str
    phone: str
    linkedin: str
    company_name: str = "N/A"
    notes: str = ""
    links: List[Dict[str, str]] = []


class SearchQuery(BaseModel):
    query: str
    user_id: str


class ConverseQuery(BaseModel):
    query: str
    user_id: str
    conversation_history: Optional[List[Dict[str, str]]] = None
    top_k: Optional[int] = 4
    admin_mode: Optional[bool] = False  # Cross-user search for admins


class ContactUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    company_name: Optional[str] = None
    notes: Optional[str] = None
    links: Optional[List[Dict[str, str]]] = None


class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class AddContactRequest(BaseModel):
    contact: Contact
    user_id: str


class UserProfileUpdate(BaseModel):
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
    preferred_ai_model: Optional[str] = None  # claude-opus, gpt-5, gemini-pro


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


class AudioNoteRequest(BaseModel):
    contact_id: str
    audio_base64: Optional[str] = None
    transcript: Optional[str] = None


class UserCardUpdate(BaseModel):
    """Update user's digital business card for QR/NFC sharing."""
    full_name: Optional[str] = None
    job_title: Optional[str] = None
    company_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin_url: Optional[str] = None
    zoom_number: Optional[str] = None
    website: Optional[str] = None
    photo_url: Optional[str] = None
    photo_base64: Optional[str] = None  # Base64 data URI for profile photo
    bio: Optional[str] = None
    custom_fields: Optional[Dict[str, str]] = None  # For any additional fields


class ContactAcceptedWebhook(BaseModel):
    """Payload for card acceptance webhook to trigger n8n email workflow."""
    contact_email: str
    contact_name: str
    user_name: str
    user_email: str
    user_company: Optional[str] = None
    timestamp: str


# --- AUTH ---
def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key


async def verify_admin(user_id: str = Query(...), api_key: str = Depends(verify_api_key)):
    """Verify that the requesting user is an admin."""
    session = await get_db_session()
    try:
        result = await session.execute(select(UserDB).where(UserDB.id == uuid.UUID(user_id)))
        user = result.scalar_one_or_none()
        if not user or not user.is_admin:
            raise HTTPException(status_code=403, detail="Admin access required")
        return user_id
    finally:
        await session.close()


# --- PASSWORD HASHING ---
def hash_password(password: str) -> str:
    password_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    password_bytes = plain_password.encode('utf-8')[:72]
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_bytes)


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
class FAISSIndex:
    """In-memory FAISS index for semantic search. Rebuilt from Postgres on startup."""

    def __init__(self):
        self.indices: Dict[str, faiss.IndexFlatIP] = {}  # user_id -> FAISS index
        self.texts: Dict[str, List[str]] = {}  # user_id -> text summaries
        self.metadata: Dict[str, List[dict]] = {}  # user_id -> contact metadata
        print("[FAISS] In-memory index manager initialized")

    def build_for_user(self, user_id: str, contacts: List[dict]):
        """Build FAISS index for a user from a list of contact dicts."""
        if not contacts:
            self.indices[user_id] = None
            self.texts[user_id] = []
            self.metadata[user_id] = []
            return

        texts = []
        metas = []
        for c in contacts:
            summary = f"{c['name']}, {c['email']}, {c['phone']}, {c['linkedin']}, {c['company_name']}"
            if c.get('notes'):
                summary += f", {c['notes'][:200]}"
            texts.append(summary)
            metas.append(c)

        vecs = np.array([get_embedding(t) for t in texts]).astype("float32")
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)

        self.indices[user_id] = index
        self.texts[user_id] = texts
        self.metadata[user_id] = metas
        print(f"[FAISS] Built index for user {user_id[:8]}... with {len(contacts)} contacts")

    def add_contact(self, user_id: str, contact_dict: dict):
        """Add a single contact to user's index."""
        summary = f"{contact_dict['name']}, {contact_dict['email']}, {contact_dict['phone']}, {contact_dict['linkedin']}, {contact_dict['company_name']}"
        if contact_dict.get('notes'):
            summary += f", {contact_dict['notes'][:200]}"

        if user_id not in self.texts:
            self.texts[user_id] = []
            self.metadata[user_id] = []

        self.texts[user_id].append(summary)
        self.metadata[user_id].append(contact_dict)

        # Rebuild index
        vecs = np.array([get_embedding(t) for t in self.texts[user_id]]).astype("float32")
        self.indices[user_id] = faiss.IndexFlatIP(vecs.shape[1])
        self.indices[user_id].add(vecs)

    def update_contact(self, user_id: str, contact_id: str, contact_dict: dict):
        """Update a contact in user's index."""
        if user_id not in self.metadata:
            return False
        for i, m in enumerate(self.metadata[user_id]):
            if m.get("id") == contact_id:
                summary = f"{contact_dict['name']}, {contact_dict['email']}, {contact_dict['phone']}, {contact_dict['linkedin']}, {contact_dict['company_name']}"
                if contact_dict.get('notes'):
                    summary += f", {contact_dict['notes'][:200]}"
                self.texts[user_id][i] = summary
                self.metadata[user_id][i] = contact_dict
                # Rebuild
                if self.texts[user_id]:
                    vecs = np.array([get_embedding(t) for t in self.texts[user_id]]).astype("float32")
                    self.indices[user_id] = faiss.IndexFlatIP(vecs.shape[1])
                    self.indices[user_id].add(vecs)
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
                # Rebuild
                if self.texts[user_id]:
                    vecs = np.array([get_embedding(t) for t in self.texts[user_id]]).astype("float32")
                    self.indices[user_id] = faiss.IndexFlatIP(vecs.shape[1])
                    self.indices[user_id].add(vecs)
                else:
                    self.indices[user_id] = None
                return True
        return False

    def delete_user(self, user_id: str):
        """Remove all FAISS data for a user."""
        self.indices.pop(user_id, None)
        self.texts.pop(user_id, None)
        self.metadata.pop(user_id, None)

    def search(self, user_id: str, query: str, k: int = 4) -> List[tuple]:
        """Search user's contacts semantically."""
        index = self.indices.get(user_id)
        if index is None or index.ntotal == 0:
            return []
        texts = self.texts.get(user_id, [])
        metas = self.metadata.get(user_id, [])

        q_vec = get_embedding(query).astype("float32").reshape(1, -1)
        D, I = index.search(q_vec, min(k, len(texts)))
        results = []
        for idx in I[0]:
            if 0 <= idx < len(texts):
                results.append((texts[idx], metas[idx]))
        return results


faiss_index = FAISSIndex()


# --- INTELLIGENCE ENGINE (Multi-Provider AI Chat) ---
class IntelligenceEngine:
    """Multi-provider AI engine. Routes chat through OpenRouter (Claude/GPT/Gemini).
    Falls back to direct Gemini API if OpenRouter is not configured.
    OCR and lead scoring always use Gemini directly (unchanged)."""

    def __init__(self):
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self.gemini_model = "gemini-2.5-flash"

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

    async def _call_openrouter(self, system_prompt: str, contact_context: str,
                                query: str, history: list = None, model_id: str = None) -> str:
        """Call OpenRouter API with the specified model."""
        if not model_id:
            model_id = AVAILABLE_MODELS["claude-opus"]["id"]

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for msg in history:
                role = msg.get("role", "user")
                if role not in ("user", "assistant"):
                    role = "user"
                messages.append({"role": role, "content": msg.get("content", "")})

        user_message = query
        if contact_context:
            user_message = f"{contact_context}\n\n---\nUser Query: {query}"
        messages.append({"role": "user", "content": user_message})

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    self.openrouter_url,
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://event-scout-delta.vercel.app",
                        "X-Title": "Event Scout",
                    },
                    json={
                        "model": model_id,
                        "messages": messages,
                        "max_tokens": 2000,
                        "temperature": 0.7,
                    },
                )
                data = resp.json()
                if "choices" in data and data["choices"]:
                    return data["choices"][0]["message"]["content"]
                elif "error" in data:
                    error_msg = data["error"].get("message", str(data["error"]))
                    print(f"[OPENROUTER] API error: {error_msg}")
                    raise Exception(f"OpenRouter error: {error_msg}")
                else:
                    raise Exception(f"Unexpected OpenRouter response: {data}")
        except httpx.TimeoutException:
            print("[OPENROUTER] Request timed out")
            raise Exception("AI response timed out. Please try again.")

    def _call_gemini_direct(self, system_prompt: str, contact_context: str,
                             query: str, history: list = None) -> str:
        """Fallback: call Gemini directly when OpenRouter is not available."""
        if not GEMINI_API_KEY or not gemini_configured:
            return "AI not configured. Please set up OpenRouter API key in settings."

        try:
            model = genai.GenerativeModel(
                model_name=self.gemini_model,
                system_instruction=system_prompt
            )
            full_prompt = f"{contact_context}\n\nUser Query: {query}"

            if history:
                gemini_history = []
                for msg in history:
                    role = "user" if msg.get("role") == "user" else "model"
                    gemini_history.append({"role": role, "parts": [msg.get("content", "")]})
                chat = model.start_chat(history=gemini_history)
                response = chat.send_message(full_prompt)
            else:
                response = model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            print(f"[GEMINI] Direct API error: {e}")
            traceback.print_exc()
            return f"AI temporarily unavailable. Error: {str(e)}"

    def _generate_fallback_response(self, query: str, retrieved_contacts: List[tuple]) -> str:
        if not retrieved_contacts:
            return f"No contacts found matching your query: '{query}'. Try adding some contacts first."
        response_parts = [f"Found {len(retrieved_contacts)} contact(s):\n"]
        for i, (text, meta) in enumerate(retrieved_contacts, 1):
            response_parts.append(f"**{i}. {meta.get('name', 'N/A')}** — {meta.get('company_name', 'N/A')}\n"
                                  f"- Email: {meta.get('email', 'N/A')} | Phone: {meta.get('phone', 'N/A')}")
        return "\n".join(response_parts)

    async def generate_response(self, query: str, retrieved_contacts: List[tuple],
                                 conversation_history: list = None, user_profile: dict = None,
                                 all_contacts: list = None, exhibitors: list = None) -> str:
        """Generate AI response using OpenRouter (primary) or Gemini (fallback)."""
        system_prompt = self._build_system_prompt(user_profile, exhibitors)
        contact_context = self._build_context_from_contacts(retrieved_contacts, all_contacts)

        # Get user's preferred model
        model_key = (user_profile or {}).get("preferred_ai_model", "claude-opus")
        model_info = AVAILABLE_MODELS.get(model_key, AVAILABLE_MODELS["claude-opus"])
        model_id = model_info["id"]

        if OPENROUTER_API_KEY:
            try:
                result = await self._call_openrouter(
                    system_prompt, contact_context, query,
                    conversation_history, model_id
                )
                return result
            except Exception as e:
                print(f"[AI] OpenRouter failed ({model_info['name']}): {e}")
                # Try Gemini as fallback
                print("[AI] Falling back to direct Gemini...")
                return self._call_gemini_direct(system_prompt, contact_context, query, conversation_history)
        else:
            return self._call_gemini_direct(system_prompt, contact_context, query, conversation_history)


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
    img = qr.make_image(fill_color="black", back_color="white")
    qr_path = os.path.join(QR_DIR, f"qr_{contact_id}.png")
    img.save(qr_path)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    qr_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {"qr_path": qr_path, "qr_base64": qr_base64, "contact_id": contact_id}


def get_qr_base64(contact_id: str) -> str:
    qr_path = os.path.join(QR_DIR, f"qr_{contact_id}.png")
    if os.path.exists(qr_path):
        with open(qr_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None


# --- QR SCANNER ---
def parse_vcard(vcard_text: str) -> dict:
    fields = {"name": "N/A", "email": "N/A", "phone": "N/A", "linkedin": "N/A", "company_name": "N/A"}
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
    return fields


def scan_qr_image(image_path: str) -> str:
    img = Image.open(image_path)
    decoded_objects = decode_qr(img)
    if not decoded_objects:
        return None
    return decoded_objects[0].data.decode("utf-8")


# --- IMAGE TEXT EXTRACTION WITH GEMINI ---
def extract_contact_from_image_with_gemini(image_path: str) -> dict:
    fields = {"name": "N/A", "email": "N/A", "phone": "N/A", "linkedin": "N/A", "company_name": "N/A"}
    if not GEMINI_API_KEY or not gemini_configured:
        print("[GEMINI] API key not configured")
        return fields

    # Use stable model for better reliability
    model_names = ['gemini-2.0-flash', 'gemini-1.5-flash']
    img = Image.open(image_path)
    print(f"[GEMINI] Image opened: {img.size}, mode={img.mode}")

    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    # Compress large images for faster OCR (business cards don't need 4K)
    max_dim = 1600
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        print(f"[GEMINI] Image resized to: {img.size}")

    prompt = """You are an expert at reading business cards. Look at this image carefully and extract ALL contact information you can find.

This is a business card or contact information image. Extract the following fields:
- name: The person's full name (first and last name)
- email: Their email address (look for @ symbol)
- phone: Their phone number (look for digits, +, -, parentheses)
- linkedin: Their LinkedIn URL or profile (look for linkedin.com or "LinkedIn:")
- company_name: Their company or organization name

Return ONLY a valid JSON object with exactly these keys. Use "N/A" for any field you cannot find:
{"name": "...", "email": "...", "phone": "...", "linkedin": "...", "company_name": "..."}

CRITICAL: Return ONLY the raw JSON object. No markdown, no backticks, no explanation."""

    last_error = None
    model_errors = []
    for model_name in model_names:
        try:
            print(f"[GEMINI] Trying model: {model_name}")
            model = genai.GenerativeModel(model_name)

            # 20s timeout — card OCR is simple, but network latency can be high
            import google.generativeai.types as genai_types
            request_options = genai_types.RequestOptions(timeout=20)
            response = model.generate_content([prompt, img], request_options=request_options)
            if not response or not response.text:
                err_msg = f"{model_name}: empty response"
                print(f"[GEMINI] {err_msg}")
                model_errors.append(err_msg)
                continue

            response_text = response.text.strip()
            print(f"[GEMINI] Raw response from {model_name}: {response_text[:500]}")

            # ROBUST JSON PARSING
            try:
                # 1. Try direct parse
                extracted = json.loads(response_text)
            except json.JSONDecodeError:
                # 2. Try to find JSON block block
                match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if match:
                    try:
                        extracted = json.loads(match.group(0))
                    except:
                        # 3. Clean common markdown garbage
                        clean_text = response_text.replace("```json", "").replace("```", "").strip()
                        extracted = json.loads(clean_text)
                else:
                    raise Exception("No JSON object found in response")

            for key in fields:
                if key in extracted and extracted[key] and str(extracted[key]).strip() and extracted[key] != "N/A":
                    fields[key] = str(extracted[key]).strip()

            print(f"[GEMINI] Extracted with {model_name}: {fields}")
            fields["_source"] = f"gemini:{model_name}"
            return fields

        except json.JSONDecodeError as e:
            err_msg = f"{model_name}: JSON parse error - {str(e)[:80]}"
            print(f"[GEMINI] {err_msg}")
            model_errors.append(err_msg)
            last_error = e
            continue
        except Exception as e:
            err_msg = f"{model_name}: {str(e)[:100]}"
            print(f"[GEMINI] {err_msg}")
            model_errors.append(err_msg)
            last_error = e
            # Fast-fail on rate limit — skip remaining Gemini models, go to OpenRouter
            err_str = str(e).lower()
            if "429" in str(e) or "ResourceExhausted" in type(e).__name__ or "quota" in err_str or "rate" in err_str:
                print(f"[GEMINI] Rate limited — skipping remaining models")
                break
            traceback.print_exc()
            continue

    print(f"[GEMINI] All models failed. Errors: {model_errors}")
    fields["_errors"] = model_errors
    return fields


# --- OPENROUTER VISION FALLBACK FOR OCR ---
async def extract_contact_with_openrouter(image_path: str) -> dict:
    """Fallback OCR using OpenRouter vision model when Gemini fails entirely."""
    fields = {"name": "N/A", "email": "N/A", "phone": "N/A", "linkedin": "N/A", "company_name": "N/A"}
    if not OPENROUTER_API_KEY:
        print("[OPENROUTER-OCR] No API key configured")
        return fields

    import base64
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    prompt = """You are an expert at reading business cards. Extract ALL contact information from this image.

Return ONLY a valid JSON object with exactly these keys. Use "N/A" for any field you cannot find:
{"name": "...", "email": "...", "phone": "...", "linkedin": "...", "company_name": "..."}

CRITICAL: Return ONLY the raw JSON object. No markdown, no backticks, no explanation."""

    # Try multiple models via OpenRouter
    or_models = [
        "google/gemini-2.5-flash",      # Gemini via OpenRouter (different quota pool)
        "anthropic/claude-sonnet-4-5",   # Claude Sonnet as fallback
    ]

    for or_model in or_models:
        try:
            print(f"[OPENROUTER-OCR] Trying model: {or_model}")
            async with httpx.AsyncClient(timeout=25.0) as client:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://event-scout-delta.vercel.app",
                        "X-Title": "Event Scout OCR",
                    },
                    json={
                        "model": or_model,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                            ]
                        }],
                        "max_tokens": 500,
                        "temperature": 0.1,
                    },
                )

            data = resp.json()
            if "error" in data:
                err_msg = data["error"].get("message", str(data["error"]))
                print(f"[OPENROUTER-OCR] API error with {or_model}: {err_msg}")
                continue

            if "choices" not in data or not data["choices"]:
                print(f"[OPENROUTER-OCR] No choices in response from {or_model}")
                continue

            response_text = data["choices"][0]["message"]["content"].strip()
            print(f"[OPENROUTER-OCR] Raw response from {or_model}: {response_text[:500]}")

            # Parse JSON from response (handle markdown wrapping)
            if response_text.startswith("```"):
                parts = response_text.split("```")
                response_text = parts[1] if len(parts) > 1 else response_text
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            if not response_text.startswith("{"):
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    response_text = response_text[json_start:json_end]

            extracted = json.loads(response_text)
            for key in fields:
                if key in extracted and extracted[key] and str(extracted[key]).strip() and extracted[key] != "N/A":
                    fields[key] = str(extracted[key]).strip()

            print(f"[OPENROUTER-OCR] Extracted with {or_model}: {fields}")
            fields["_source"] = f"openrouter:{or_model}"
            return fields

        except Exception as e:
            print(f"[OPENROUTER-OCR] Error with {or_model}: {e}")
            traceback.print_exc()
            continue

    print("[OPENROUTER-OCR] All models failed")
    return fields


# --- LINKEDIN AUTO-LOOKUP ---
def lookup_linkedin_with_gemini(name: str, company: str) -> Optional[str]:
    """Try to find LinkedIn URL using Gemini when OCR didn't find one."""
    if not GEMINI_API_KEY or not gemini_configured:
        return None
    if not name or name == "N/A":
        return None

    prompt = f"""Given this person: {name} at {company if company != 'N/A' else 'unknown company'}.

What is their most likely LinkedIn profile URL?

Rules:
- Return ONLY the URL (e.g. https://linkedin.com/in/username)
- If you cannot determine it with reasonable confidence, return exactly: N/A
- Do NOT make up URLs. Only return a URL if you're reasonably confident.
- Return ONLY the URL or N/A, nothing else."""

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        import google.generativeai.types as genai_types
        request_options = genai_types.RequestOptions(timeout=20)
        response = model.generate_content(prompt, request_options=request_options)
        if response and response.text:
            url = response.text.strip()
            if "linkedin.com" in url.lower() and url.startswith("http"):
                return url
    except Exception as e:
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
        "callback_url": f"https://event-scout-production.up.railway.app/contact/{contact_id}/enrich",
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
            conditions.append(ContactDB.email == email)
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
                "matched_on": "email" if (email and email != "N/A" and existing.email == email) else "phone"
            }
        return None
    finally:
        await session.close()


# --- CONTACT LOGIC (now using Postgres) ---
async def add_contact_logic(contact: Contact, user_id: str, source: str = "manual", photo_base64: str = None) -> dict:
    """Add contact to PostgreSQL and FAISS index."""
    session = await get_db_session()
    try:
        contact_id = str(uuid.uuid4())

        # Save to Postgres
        db_contact = ContactDB(
            id=uuid.UUID(contact_id),
            user_id=uuid.UUID(user_id),
            name=contact.name,
            email=contact.email,
            phone=contact.phone,
            linkedin=contact.linkedin,
            company_name=contact.company_name,
            notes=contact.notes,
            links=contact.links or [],
            source=source,
            photo_base64=photo_base64,
        )
        session.add(db_contact)

        # Also save to shared_contacts pool
        shared = SharedContactDB(
            original_contact_id=uuid.UUID(contact_id),
            user_id=uuid.UUID(user_id),
            name=contact.name,
            email=contact.email,
            phone=contact.phone,
            linkedin=contact.linkedin,
            company_name=contact.company_name,
            notes=contact.notes,
            links=contact.links or [],
            source=source,
        )
        session.add(shared)
        await session.commit()

        # Generate QR
        qr_result = create_qr(contact, contact_id)

        # Update FAISS index
        contact_dict = {
            "id": contact_id,
            "name": contact.name,
            "email": contact.email,
            "phone": contact.phone,
            "linkedin": contact.linkedin,
            "company_name": contact.company_name,
            "notes": contact.notes,
            "links": contact.links or [],
            "source": source,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        faiss_index.add_contact(user_id, contact_dict)

        # Fire webhook in background (don't block)
        if WEBHOOK_URL:
            import asyncio  # noqa: local import for task creation
            # Get user data for webhook
            user_result = await session.execute(select(UserDB).where(UserDB.id == uuid.UUID(user_id)))
            user_row = user_result.scalar_one_or_none()
            user_data = {"user_id": user_id, "name": user_row.name if user_row else "N/A", "email": user_row.email if user_row else "N/A"}
            asyncio.create_task(fire_webhook(contact_dict, user_data, contact_id))

        return {"contact_id": contact_id, "qr_base64": qr_result["qr_base64"]}
    except Exception as e:
        await session.rollback()
        raise e
    finally:
        await session.close()


async def add_contact_from_qr(file: UploadFile, user_id: str):
    temp_filename = os.path.join(TEMP_DIR, file.filename or f"{uuid.uuid4()}.png")
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
        contact_obj = Contact(**fields)
        result = await add_contact_logic(contact_obj, user_id, source="qr")
        return {
            "status": "success",
            "message": "Contact added from QR code",
            "extracted_fields": fields,
            "contact_id": result["contact_id"],
            "qr_base64": result["qr_base64"]
        }
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


async def add_contact_from_image(file: UploadFile, user_id: str):
    temp_filename = os.path.join(TEMP_DIR, file.filename or f"{uuid.uuid4()}.png")
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
        # Run synchronous Gemini OCR in thread pool with hard timeout
        # Gemini SDK timeout is unreliable — asyncio.wait_for enforces a real deadline
        try:
            fields = await asyncio.wait_for(
                asyncio.to_thread(extract_contact_from_image_with_gemini, temp_filename),
                timeout=20
            )
        except asyncio.TimeoutError:
            print("[SCAN] Gemini OCR hard timeout (20s) — skipping to OpenRouter")
            fields = {"name": "N/A", "email": "N/A", "phone": "N/A", "linkedin": "N/A", "company_name": "N/A"}
            fields["_errors"] = ["Gemini timeout (20s hard limit)"]
        has_info = any(v != "N/A" for k, v in fields.items() if k not in ("linkedin", "_errors", "_source"))
        gemini_errors = fields.pop("_errors", [])
        gemini_source = fields.pop("_source", None)

        # If Gemini failed, try OpenRouter as fallback
        if not has_info:
            print(f"[SCAN] Gemini returned no info. Errors: {gemini_errors}. Trying OpenRouter fallback...")
            fields = await extract_contact_with_openrouter(temp_filename)
            has_info = any(v != "N/A" for k, v in fields.items() if k not in ("linkedin", "_errors", "_source"))
            or_source = fields.pop("_source", None)
            if has_info:
                print(f"[SCAN] OpenRouter fallback succeeded via {or_source}")
            else:
                # Both Gemini and OpenRouter failed — report all errors
                all_errors = gemini_errors.copy()
                all_errors.append("OpenRouter fallback also returned no contact info")
                error_detail = f"No contact information found in image. Models tried: {', '.join(all_errors)}"
                print(f"[SCAN] All OCR models failed. {error_detail}")
                raise HTTPException(status_code=400, detail=error_detail)

        # DISABLED: LinkedIn auto-lookup generates hallucinated/fake URLs
        # Only use LinkedIn if actually found on the business card via OCR
        # if fields.get("linkedin", "N/A") == "N/A":
        #     linkedin_url = lookup_linkedin_with_gemini(fields["name"], fields.get("company_name", "N/A"))
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
        result = await add_contact_logic(contact_obj, user_id, source="scan", photo_base64=photo_base64)

        # If LinkedIn was AI-detected, update the source in DB
        if fields.get("linkedin_source") == "ai_detected":
            session = await get_db_session()
            try:
                stmt = (
                    update(ContactDB)
                    .where(ContactDB.id == uuid.UUID(result["contact_id"]))
                    .values(linkedin_source="ai_detected")
                )
                await session.execute(stmt)
                await session.commit()
            finally:
                await session.close()

        # Trigger AI pipeline in background (research → pitch → deck)
        if AUTO_PIPELINE_ENABLED and OPENROUTER_API_KEY:
            asyncio.create_task(run_contact_pipeline(result["contact_id"], user_id))
            print(f"[PIPELINE] Auto-triggered for contact {result['contact_id']}")

        return {
            "status": "success",
            "message": "Contact added from image",
            "extracted_fields": fields,
            "contact_id": result["contact_id"],
            "qr_base64": result["qr_base64"],
            "photo_base64": photo_base64,
            "pipeline_started": AUTO_PIPELINE_ENABLED and bool(OPENROUTER_API_KEY),
        }
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


# --- LEAD SCORING ---
def score_contact_with_gemini(contact_meta: dict, user_profile: dict) -> dict:
    if not GEMINI_API_KEY or not gemini_configured:
        return {
            "score": 50, "temperature": "warm",
            "reasoning": "Gemini not configured.",
            "recommended_actions": ["Configure Gemini API for AI-powered scoring"],
            "breakdown": {"profile_fit": 10, "role_relevance": 10, "company_fit": 10, "engagement": 10, "timing": 10}
        }

    prompt = f"""You are a lead scoring expert. Score this contact against the user's profile and event context.

USER PROFILE:
- Company: {user_profile.get('company_name', 'N/A')}
- Job Title: {user_profile.get('job_title', 'N/A')}
- Products/Services: {json.dumps(user_profile.get('products', []))}
- Target Industries: {json.dumps(user_profile.get('target_industries', []))}
- Target Roles: {json.dumps(user_profile.get('target_roles', []))}
- Target Company Sizes: {json.dumps(user_profile.get('target_company_sizes', []))}
- Target Geographies: {json.dumps(user_profile.get('target_geographies', []))}
- Value Propositions: {json.dumps(user_profile.get('value_propositions', []))}
- Event: {user_profile.get('current_event_name', 'N/A')}
- Event Goals: {json.dumps(user_profile.get('event_goals', []))}

CONTACT:
- Name: {contact_meta.get('name', 'N/A')}
- Company: {contact_meta.get('company_name', 'N/A')}
- Email: {contact_meta.get('email', 'N/A')}
- Phone: {contact_meta.get('phone', 'N/A')}
- LinkedIn: {contact_meta.get('linkedin', 'N/A')}
- Notes: {contact_meta.get('notes', '')}

Score this contact on 5 dimensions (each 0-20, total 0-100):
1. profile_fit 2. role_relevance 3. company_fit 4. engagement 5. timing

Return ONLY a valid JSON object:
{{"score": <0-100>, "temperature": "<hot|warm|cold>", "reasoning": "<2-3 sentences>", "recommended_actions": ["action1", "action2", "action3"], "breakdown": {{"profile_fit": <0-20>, "role_relevance": <0-20>, "company_fit": <0-20>, "engagement": <0-20>, "timing": <0-20>}}}}

Rules: hot >= 70, warm = 40-69, cold < 40. Return ONLY raw JSON."""

    model_names = ['gemini-2.5-flash', 'gemini-2.0-flash']
    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            import google.generativeai.types as genai_types
            request_options = genai_types.RequestOptions(timeout=30)
            response = model.generate_content(prompt, request_options=request_options)
            if not response or not response.text:
                continue
            response_text = response.text.strip()
            if response_text.startswith("```"):
                parts = response_text.split("```")
                response_text = parts[1] if len(parts) > 1 else response_text
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            if not response_text.startswith("{"):
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    response_text = response_text[json_start:json_end]
            result = json.loads(response_text)
            score = result.get("score", 50)
            result["temperature"] = "hot" if score >= 70 else ("warm" if score >= 40 else "cold")
            return result
        except Exception as e:
            print(f"[SCORING] Error with {model_name}: {e}")
            continue

    return {
        "score": 50, "temperature": "warm",
        "reasoning": "Scoring temporarily unavailable.",
        "recommended_actions": ["Retry scoring later"],
        "breakdown": {"profile_fit": 10, "role_relevance": 10, "company_fit": 10, "engagement": 10, "timing": 10}
    }


# --- INTELLIGENCE PIPELINE ---

RESEARCH_PROMPT_TEMPLATE = """You are a B2B sales intelligence analyst for Lake B2B, a leading data-driven growth solutions provider. Research this contact for an upcoming sales engagement at WHX Dubai 2026.

CONTACT:
- Name: {name}
- Company: {company_name}
- Email: {email}
- LinkedIn: {linkedin}

RESEARCH REQUIREMENTS:
1. PERSON PROFILE: Likely role, seniority level, decision-making authority, key responsibilities
2. COMPANY ANALYSIS: Industry vertical, estimated size, core products/services, growth signals, tech stack indicators
3. PAIN POINTS: Based on their industry and likely role, what challenges do they face with data quality, lead generation, customer acquisition, or marketing ROI?
4. OPPORTUNITIES: How could Lake B2B's offerings (intent data, ICP technology, multi-channel outreach, data enrichment, growth advisory) address their specific challenges?
5. CONVERSATION STARTERS: 3 specific, natural talking points the sales team can use at the event
6. COMPETITIVE LANDSCAPE: Who else in the data/martech space might be pitching them?

Return ONLY valid JSON (no markdown fences):
{{"person": {{"name": "", "likely_title": "", "seniority": "", "authority": ""}}, "company": {{"name": "", "industry": "", "size_estimate": "", "products": "", "growth_signals": ""}}, "industry": "", "pain_points": [], "opportunities": [], "talking_points": [], "competitive_notes": "", "confidence_level": "high|medium|low"}}"""

PITCH_PROMPT_TEMPLATE = """You are a senior pitch strategist for Lake B2B. Generate a personalized pitch for this contact.

LAKE B2B BRAND VOICE:
- Tone: Professional, confident, data-driven, results-oriented
- Key metrics: 3.25X higher lead-to-opportunity conversion, 50% shorter sales cycles (47 vs 94 days), 3X better campaign ROI (12:1 vs 4:1)
- Language: Action-oriented, metric-backed claims, problem-to-solution narrative
- Terminology: Intent signals, buyer intent, firmographic/technographic data, ICP alignment, multi-channel outreach, full-funnel generation
- CTA style: "Let's explore how..." — consultative, not pushy

CONTACT RESEARCH:
{research_json}

SALES REP PROFILE:
{user_profile_json}

Generate TWO outputs:

OUTPUT 1 — PITCH DECK (8 slides):
Each slide as: {{"title": "...", "content": "2-3 bullet points or short paragraphs", "speaker_notes": "what to say"}}
Structure:
1. Opening Hook — Personalized to their company/industry challenge
2. The Challenge — Their specific pain points from research
3. Market Reality — Industry urgency with data
4. Cost of Inaction — What they risk by not acting
5. Lake B2B's Approach — Data + intent signals methodology
6. The Solution — Product mapping to their needs
7. Proof Points — Our metrics (3.25X, 50% shorter, 12:1 ROI)
8. Next Steps — Clear CTA with meeting/demo suggestion

OUTPUT 2 — EMAIL PITCH:
- Subject (under 50 chars, personalized)
- Body (3-4 paragraphs: hook, value prop, proof, CTA)
- Reference the attached presentation
- Professional sign-off

Return ONLY valid JSON (no markdown fences):
{{"slides": [...], "email_subject": "...", "email_body": "..."}}"""


async def _pipeline_call_openrouter(prompt: str, model_id: str = None, max_tokens: int = 3000) -> str:
    """Call OpenRouter for pipeline tasks (research/pitch). Returns raw text."""
    if not OPENROUTER_API_KEY:
        raise Exception("OpenRouter API key not configured")
    if not model_id:
        model_id = AVAILABLE_MODELS["claude-opus"]["id"]

    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://event-scout-delta.vercel.app",
                "X-Title": "Event Scout Pipeline",
            },
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.4,
            },
        )
        data = resp.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"]
        elif "error" in data:
            raise Exception(f"OpenRouter error: {data['error'].get('message', str(data['error']))}")
        else:
            raise Exception(f"Unexpected OpenRouter response: {json.dumps(data)[:200]}")


async def _pipeline_update_status(pipeline_id, status: str, step: str = "", **kwargs):
    """Update pipeline status in DB."""
    session = await get_db_session()
    try:
        result = await session.execute(select(ContactPipelineDB).where(ContactPipelineDB.id == pipeline_id))
        pipeline = result.scalar_one_or_none()
        if not pipeline:
            return
        pipeline.status = status
        pipeline.current_step = step
        pipeline.updated_at = datetime.now(timezone.utc)
        for k, v in kwargs.items():
            if hasattr(pipeline, k):
                setattr(pipeline, k, v)
        await session.commit()
    finally:
        await session.close()


async def pipeline_step_research(pipeline_id, contact_id: str, user_id: str):
    """Step 1: Research the contact using Claude via OpenRouter."""
    await _pipeline_update_status(pipeline_id, "researching", "Researching contact...")

    session = await get_db_session()
    try:
        result = await session.execute(select(ContactDB).where(ContactDB.id == uuid.UUID(contact_id)))
        contact = result.scalar_one_or_none()
        if not contact:
            raise Exception(f"Contact {contact_id} not found")

        prompt = RESEARCH_PROMPT_TEMPLATE.format(
            name=contact.name or "Unknown",
            company_name=contact.company_name or "Unknown",
            email=contact.email or "N/A",
            linkedin=contact.linkedin or "N/A",
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

        # Also append research to contact notes
        existing_notes = contact.notes or ""
        research_block = f"\n\n--- AI Research ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}) ---\n{research_summary}"
        contact.notes = existing_notes + research_block
        contact.updated_at = datetime.now(timezone.utc)
        await session.commit()

        return research_data
    finally:
        await session.close()


async def pipeline_step_score(pipeline_id, contact_id: str, user_id: str):
    """Step 2: Score the lead using existing Gemini scoring."""
    await _pipeline_update_status(pipeline_id, "scoring", "Scoring lead...")

    session = await get_db_session()
    try:
        result = await session.execute(select(ContactDB).where(ContactDB.id == uuid.UUID(contact_id)))
        contact = result.scalar_one_or_none()
        if not contact:
            raise Exception(f"Contact {contact_id} not found")

        # Get user profile for scoring context
        prof_result = await session.execute(
            select(UserProfileDB).where(UserProfileDB.user_id == uuid.UUID(user_id))
        )
        profile = prof_result.scalar_one_or_none()
        user_profile = profile.profile_data if profile else {}

        contact_meta = {
            "name": contact.name, "email": contact.email, "phone": contact.phone,
            "company_name": contact.company_name, "linkedin": contact.linkedin,
            "notes": contact.notes or "", "source": contact.source or "scan",
        }

        try:
            score_result = await asyncio.wait_for(
                asyncio.to_thread(score_contact_with_gemini, contact_meta, user_profile),
                timeout=30
            )
        except asyncio.TimeoutError:
            print("[PIPELINE] Scoring hard timeout (30s) — using default score")
            score_result = {"score": 50, "temperature": "warm", "reasoning": "Scoring timed out - manual review needed", "breakdown": {}, "recommended_actions": ["Manual review needed"]}

        # Save scores to contact
        contact.lead_score = score_result.get("score")
        contact.lead_temperature = score_result.get("temperature")
        contact.lead_score_reasoning = score_result.get("reasoning", "")
        contact.lead_score_breakdown = score_result.get("breakdown", {})
        contact.lead_recommended_actions = score_result.get("recommended_actions", [])
        contact.updated_at = datetime.now(timezone.utc)
        await session.commit()

        # Update FAISS
        contact_dict = {
            "id": contact_id, "name": contact.name, "email": contact.email,
            "phone": contact.phone, "company_name": contact.company_name,
            "lead_score": contact.lead_score, "lead_temperature": contact.lead_temperature,
        }
        faiss_index.update_contact(user_id, contact_id, contact_dict)

        await _pipeline_update_status(pipeline_id, "scoring", "Scoring complete", score_completed=True)
        return score_result
    finally:
        await session.close()


async def pipeline_step_pitch(pipeline_id, contact_id: str, user_id: str, research_data: dict):
    """Step 3: Generate personalized pitch + email using Claude."""
    await _pipeline_update_status(pipeline_id, "pitching", "Generating pitch...")

    session = await get_db_session()
    try:
        # Get user profile
        prof_result = await session.execute(
            select(UserProfileDB).where(UserProfileDB.user_id == uuid.UUID(user_id))
        )
        profile = prof_result.scalar_one_or_none()
        user_profile = profile.profile_data if profile else {}

        prompt = PITCH_PROMPT_TEMPLATE.format(
            research_json=json.dumps(research_data, indent=2),
            user_profile_json=json.dumps(user_profile, indent=2),
        )

        raw_response = await _pipeline_call_openrouter(prompt, max_tokens=4000)

        # Parse JSON
        clean = raw_response.strip()
        if clean.startswith("```"):
            clean = re.sub(r'^```(?:json)?\s*', '', clean)
            clean = re.sub(r'\s*```$', '', clean)

        try:
            pitch_data = json.loads(clean)
        except json.JSONDecodeError:
            pitch_data = {"slides": [], "email_subject": "Meeting Follow-up", "email_body": raw_response}

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


async def pipeline_step_deck(pipeline_id, slides_content: list, contact_name: str):
    """Step 4: Generate PPTX via Presenton API."""
    if not PRESENTON_API_URL:
        print("[PIPELINE] Presenton not configured — skipping deck generation")
        await _pipeline_update_status(pipeline_id, "generating_deck", "Skipped (Presenton not configured)")
        return None

    await _pipeline_update_status(pipeline_id, "generating_deck", "Creating presentation...")

    try:
        # Build markdown from slides for Presenton
        slides_markdown = []
        full_narrative = []
        for slide in slides_content:
            title = slide.get("title", "Slide")
            content = slide.get("content", "")
            slides_markdown.append(f"# {title}\n\n{content}")
            full_narrative.append(f"{title}: {content}")

        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {
                "content": "\n\n".join(full_narrative),
                "tone": "sales_pitch",
                "verbosity": "standard",
                "n_slides": len(slides_content),
                "template": "modern",
                "export_as": "pptx",
                "web_search": False,
                "include_title_slide": True,
            }
            headers = {"Content-Type": "application/json"}
            if PRESENTON_API_KEY:
                headers["Authorization"] = f"Bearer {PRESENTON_API_KEY}"

            resp = await client.post(
                f"{PRESENTON_API_URL.rstrip('/')}/api/v1/ppt/presentation/generate",
                json=payload,
                headers=headers,
            )

            if resp.status_code != 200:
                raise Exception(f"Presenton API error: {resp.status_code} - {resp.text[:200]}")

            result = resp.json()
            presentation_id = result.get("presentation_id", "")
            download_path = result.get("path", "")

            if not download_path:
                raise Exception("No download path in Presenton response")

            # Download the PPTX
            pptx_resp = await client.get(download_path)
            if pptx_resp.status_code != 200:
                raise Exception(f"Failed to download PPTX: {pptx_resp.status_code}")

            await _pipeline_update_status(
                pipeline_id, "generating_deck", "Deck created",
                presenton_presentation_id=presentation_id,
            )

            return pptx_resp.content  # Raw PPTX bytes

    except Exception as e:
        print(f"[PIPELINE] Deck generation failed: {e}")
        await _pipeline_update_status(pipeline_id, "generating_deck", f"Deck failed: {str(e)[:100]}")
        return None


async def pipeline_step_attach(pipeline_id, contact_id: str, user_id: str, pptx_bytes: bytes, contact_name: str):
    """Step 5: Store the generated PPTX as a ContactFileDB record."""
    await _pipeline_update_status(pipeline_id, "attaching", "Attaching deck to contact...")

    session = await get_db_session()
    try:
        safe_name = re.sub(r'[^a-zA-Z0-9_\- ]', '', contact_name or "contact").strip().replace(' ', '_')
        filename = f"pitch_deck_{safe_name}.pptx"

        db_file = ContactFileDB(
            contact_id=uuid.UUID(contact_id),
            filename=filename,
            original_filename=filename,
            file_type="pptx",
            mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            file_size=len(pptx_bytes),
            file_data=pptx_bytes,
            description="Auto-generated pitch deck by AI Pipeline",
            category="pitch",
            uploaded_by=uuid.UUID(user_id),
        )
        session.add(db_file)
        await session.commit()

        await _pipeline_update_status(
            pipeline_id, "attaching", "Deck attached",
            deck_file_id=db_file.id,
        )

        return str(db_file.id)
    finally:
        await session.close()


async def run_contact_pipeline(contact_id: str, user_id: str):
    """Main pipeline orchestrator: research → score → pitch → deck → attach."""
    print(f"[PIPELINE] Starting for contact={contact_id} user={user_id}")

    # Create pipeline record
    session = await get_db_session()
    try:
        pipeline = ContactPipelineDB(
            contact_id=uuid.UUID(contact_id),
            user_id=uuid.UUID(user_id),
            status="pending",
            current_step="Initializing...",
            started_at=datetime.now(timezone.utc),
        )
        session.add(pipeline)
        await session.commit()
        pipeline_id = pipeline.id
    finally:
        await session.close()

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
        research_data = await pipeline_step_research(pipeline_id, contact_id, user_id)

        # Step 2: Score
        print(f"[PIPELINE] Step 2/5: Scoring {contact_name}...")
        await pipeline_step_score(pipeline_id, contact_id, user_id)

        # Step 3: Pitch
        print(f"[PIPELINE] Step 3/5: Generating pitch for {contact_name}...")
        pitch_data = await pipeline_step_pitch(pipeline_id, contact_id, user_id, research_data)

        # Step 4: Deck (optional — depends on Presenton being configured)
        slides = pitch_data.get("slides", [])
        pptx_bytes = None
        if slides:
            print(f"[PIPELINE] Step 4/5: Generating deck for {contact_name}...")
            pptx_bytes = await pipeline_step_deck(pipeline_id, slides, contact_name)

        # Step 5: Attach deck (if generated)
        if pptx_bytes:
            print(f"[PIPELINE] Step 5/5: Attaching deck for {contact_name}...")
            await pipeline_step_attach(pipeline_id, contact_id, user_id, pptx_bytes, contact_name)

        # Mark complete
        final_status = "complete" if pptx_bytes else "complete_no_deck"
        await _pipeline_update_status(
            pipeline_id, final_status, "Pipeline complete",
            completed_at=datetime.now(timezone.utc),
        )
        print(f"[PIPELINE] Complete for {contact_name} (status={final_status})")

    except Exception as e:
        print(f"[PIPELINE] Failed for {contact_name}: {e}")
        traceback.print_exc()
        await _pipeline_update_status(
            pipeline_id, "failed", f"Failed: {str(e)[:200]}",
            error_message=str(e),
        )


# --- FASTAPI APP ---
app = FastAPI(title="Contact Assistant API - Multi-User (PostgreSQL)", version="2.1.0-admin-cmd")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


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
    
    print(f"[STARTUP] Gemini configured: {gemini_configured} (OCR/scoring)")
    print(f"[STARTUP] OpenRouter configured: {bool(OPENROUTER_API_KEY)} (AI chat)")
    print(f"[STARTUP] Available models: {', '.join(m['name'] for m in AVAILABLE_MODELS.values())}")
    print(f"[STARTUP] Webhook URL: {WEBHOOK_URL or 'not set'}")
    print("=" * 50)

    # Initialize database tables
    db_ok = await init_db()
    if not db_ok:
        print("[STARTUP] WARNING: Database initialization failed!")
        return

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
            contact_dicts = []
            for c in contacts:
                contact_dicts.append({
                    "id": str(c.id),
                    "name": c.name,
                    "email": c.email or "N/A",
                    "phone": c.phone or "N/A",
                    "linkedin": c.linkedin or "N/A",
                    "company_name": c.company_name or "N/A",
                    "notes": c.notes or "",
                    "links": c.links or [],
                    "source": c.source or "manual",
                    "linkedin_source": c.linkedin_source or "card",
                    "lead_score": c.lead_score,
                    "lead_temperature": c.lead_temperature,
                    "lead_score_reasoning": c.lead_score_reasoning or "",
                    "lead_score_breakdown": c.lead_score_breakdown or {},
                    "lead_recommended_actions": c.lead_recommended_actions or [],
                    "created_at": c.created_at.isoformat() if c.created_at else "",
                })
            faiss_index.build_for_user(user_id_str, contact_dicts)

        print(f"[STARTUP] FAISS indices built for {len(users)} users")
    except Exception as e:
        print(f"[STARTUP] Error building FAISS indices: {e}")
        traceback.print_exc()
    finally:
        await session.close()

    # Auto-backup to secondary Postgres (non-blocking)
    if ASYNC_BACKUP_URL:
        import asyncio
        asyncio.create_task(_run_startup_backup())
        asyncio.create_task(_periodic_backup_loop())
        print(f"[STARTUP] Backup configured: {ASYNC_BACKUP_URL[:30]}...")
    else:
        print("[STARTUP] No BACKUP_DATABASE_URL set - backup disabled")

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

    # Tables in FK-safe order (parents before children)
    tables = [
        ("users", UserDB),
        ("user_profiles", UserProfileDB),
        ("contacts", ContactDB),
        ("shared_contacts", SharedContactDB),
        ("exhibitors", ExhibitorDB),
        ("conversations", ConversationDB),
        ("user_cards", UserCardDB),
        ("event_files", EventFileDB),
        ("contact_files", ContactFileDB),
        ("contact_pipelines", ContactPipelineDB),
        ("admin_broadcasts", AdminBroadcastDB),
    ]

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

        global _last_backup_result
        _last_backup_result = {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": duration,
            "tables": stats,
            "total_rows": total_rows,
            "summary": f"{total_rows} rows across {len([v for v in stats.values() if isinstance(v, int)])} tables in {duration}s"
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
        "gemini_configured": gemini_configured,
        "openrouter_configured": bool(OPENROUTER_API_KEY),
        "webhook_configured": bool(WEBHOOK_URL),
        "version": "3.4.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    if debug:
        import os
        db_url = os.environ.get("DATABASE_URL", "")
        db_url_masked = "not_set"
        if db_url:
            if "@" in db_url:
                parts = db_url.split("@")
                if len(parts) == 2:
                    db_url_masked = f"postgres://***@{parts[1]}"
                else:
                    db_url_masked = "postgres://***@***"
            else:
                db_url_masked = db_url[:20] + "..." if len(db_url) > 20 else db_url
        
        health_status["debug"] = {
            "database_url_configured": bool(db_url),
            "database_url_masked": db_url_masked,
            "async_database_url_configured": bool(ASYNC_DATABASE_URL),
            "session_factory_exists": get_session_factory() is not None,
            "engine_exists": get_engine() is not None,
            "environment_keys": list(os.environ.keys()) if os.environ else [],
        }
    
    # Check database availability with timeout
    try:
        factory = get_session_factory()
        if factory is None:
            health_status["status"] = "degraded"
            health_status["database"] = "not_configured"
            health_status["database_error"] = "DATABASE_URL not set or invalid"
            return health_status
        
        # Database check with timeout to prevent hanging
        async def check_db():
            async with factory() as session:
                result = await session.execute(select(func.count(UserDB.id)))
                total_users = result.scalar() or 0
                health_status["database"] = "postgresql"
                health_status["database_connected"] = True
                health_status["total_users"] = total_users
        
        try:
            await asyncio.wait_for(check_db(), timeout=3.0)
        except asyncio.TimeoutError:
            health_status["status"] = "degraded"
            health_status["database"] = "timeout"
            health_status["database_error"] = "Database connection timed out after 3 seconds"
        except Exception as db_err:
            health_status["status"] = "degraded"
            health_status["database"] = "error"
            health_status["database_error"] = str(db_err)
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["database"] = "error"
        health_status["database_error"] = str(e)
    
    return health_status


@app.get("/debug")
async def debug_info():
    """Debug endpoint to check configuration (no secrets)."""
    import os
    db_url = os.environ.get("DATABASE_URL", "")
    db_url_masked = "not_set"
    if db_url:
        # Mask credentials
        try:
            if "@" in db_url:
                # postgres://user:pass@host:port/dbname
                parts = db_url.split("@")
                if len(parts) == 2:
                    db_url_masked = f"postgres://***@{parts[1]}"
                else:
                    db_url_masked = "postgres://***@***"
            else:
                db_url_masked = db_url[:20] + "..." if len(db_url) > 20 else db_url
        except:
            db_url_masked = "masked"
    
    factory = get_session_factory()
    engine = get_engine()
    
    return {
        "database_url_configured": bool(db_url),
        "database_url_masked": db_url_masked,
        "async_database_url_configured": bool(ASYNC_DATABASE_URL),
        "session_factory_exists": factory is not None,
        "engine_exists": engine is not None,
        "gemini_configured": gemini_configured,
        "openrouter_configured": bool(OPENROUTER_API_KEY),
        "webhook_configured": bool(WEBHOOK_URL),
        "environment_keys": list(os.environ.keys()) if os.environ else [],
    }


@app.get("/user/validate")
async def validate_user(user_id: str = Query(..., description="User ID to validate")):
    """Validate that a user_id exists in the database. Used to detect expired sessions."""
    session = await get_db_session()
    try:
        result = await session.execute(
            select(UserDB).where(UserDB.id == uuid.UUID(user_id))
        )
        user = result.scalar_one_or_none()

        if not user:
            return {
                "status": "invalid",
                "message": "User not found. Please log in again.",
                "valid": False,
            }

        return {
            "status": "valid",
            "valid": True,
            "user_id": str(user.id),
            "name": user.name,
            "email": user.email,
            "is_admin": user.is_admin,
        }
    except ValueError:
        return {
            "status": "invalid",
            "message": "Invalid user ID format",
            "valid": False,
        }
    except Exception as e:
        print(f"[VALIDATE] Error: {e}")
        return {
            "status": "error",
            "message": str(e),
            "valid": False,
        }
    finally:
        await session.close()


@app.post("/register/")
async def register_user(user: UserRegister):
    session = await get_db_session()
    try:
        # Check if email already exists
        result = await session.execute(select(UserDB).where(UserDB.email == user.email))
        existing = result.scalar_one_or_none()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")

        user_id = uuid.uuid4()
        is_admin = user.email.strip().lower() in ADMIN_EMAILS
        db_user = UserDB(
            id=user_id,
            name=user.name,
            email=user.email,
            password_hash=hash_password(user.password),
            is_admin=is_admin,
        )
        session.add(db_user)
        await session.commit()

        # Initialize empty FAISS index for user
        faiss_index.build_for_user(str(user_id), [])

        print(f"[USER] Created new user: {user.email} with ID: {user_id} (admin={is_admin})")
        return {
            "status": "success",
            "message": "User registered successfully",
            "user_id": str(user_id),
            "name": user.name,
            "email": user.email,
            "is_admin": is_admin,
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
async def login_user(credentials: UserLogin):
    session = await get_db_session()
    try:
        result = await session.execute(select(UserDB).where(UserDB.email == credentials.email))
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        if not verify_password(credentials.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Auto-promote to admin if email is in ADMIN_EMAILS but not yet flagged
        is_admin = user.is_admin
        if not is_admin and user.email.strip().lower() in ADMIN_EMAILS:
            user.is_admin = True
            is_admin = True
            await session.commit()
            print(f"[ADMIN] Auto-promoted {user.email} to admin")

        return {
            "status": "success",
            "message": "Login successful",
            "user_id": str(user.id),
            "name": user.name,
            "email": user.email,
            "is_admin": is_admin,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.post("/add_contact/")
async def add_contact_route(request: AddContactRequest, api_key: str = Depends(verify_api_key)):
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
    api_key: str = Depends(verify_api_key),
):
    try:
        return await add_contact_from_image(file, user_id)
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
    api_key: str = Depends(verify_api_key),
):
    return await add_contact_from_qr(file, user_id)


@app.post("/generate_qr/")
async def generate_qr_route(contact: Contact, api_key: str = Depends(verify_api_key)):
    try:
        result = create_qr(contact)
        return {"status": "success", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/")
async def search_route(query: SearchQuery, api_key: str = Depends(verify_api_key)):
    try:
        results = faiss_index.search(query.user_id, query.query, k=4)
        return {"status": "success", "results": [{"text": r[0], "meta": r[1]} for r in results]}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list_contacts/")
async def list_contacts_route(
    user_id: str = Query(..., description="User ID"),
    api_key: str = Depends(verify_api_key),
):
    session = await get_db_session()
    try:
        result = await session.execute(
            select(ContactDB).where(ContactDB.user_id == uuid.UUID(user_id)).order_by(ContactDB.created_at.desc())
        )
        contacts = result.scalars().all()

        contact_list = []
        for c in contacts:
            contact_data = {
                "id": str(c.id),
                "name": c.name,
                "email": c.email or "N/A",
                "phone": c.phone or "N/A",
                "linkedin": c.linkedin or "N/A",
                "linkedin_source": c.linkedin_source or "card",
                "company_name": c.company_name or "N/A",
                "notes": c.notes or "",
                "links": c.links or [],
                "source": c.source or "manual",
                "created_at": c.created_at.isoformat() if c.created_at else "",
                "lead_score": c.lead_score,
                "lead_temperature": c.lead_temperature,
                "lead_score_reasoning": c.lead_score_reasoning or "",
                "admin_notes": getattr(c, 'admin_notes', '') or "",
                "photo_base64": getattr(c, 'photo_base64', None),
                "qr_base64": get_qr_base64(str(c.id)),
            }
            contact_list.append(contact_data)

        return {"status": "success", "total_contacts": len(contact_list), "contacts": contact_list}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.get("/contact/{contact_id}")
async def get_contact_route(
    contact_id: str,
    user_id: str = Query(..., description="User ID"),
    api_key: str = Depends(verify_api_key),
):
    session = await get_db_session()
    try:
        result = await session.execute(
            select(ContactDB).where(
                ContactDB.id == uuid.UUID(contact_id),
                ContactDB.user_id == uuid.UUID(user_id),
            )
        )
        c = result.scalar_one_or_none()
        if not c:
            raise HTTPException(status_code=404, detail="Contact not found")

        contact_data = {
            "id": str(c.id),
            "name": c.name,
            "email": c.email or "N/A",
            "phone": c.phone or "N/A",
            "linkedin": c.linkedin or "N/A",
            "linkedin_source": c.linkedin_source or "card",
            "company_name": c.company_name or "N/A",
            "notes": c.notes or "",
            "links": c.links or [],
            "source": c.source or "manual",
            "created_at": c.created_at.isoformat() if c.created_at else "",
            "lead_score": c.lead_score,
            "lead_temperature": c.lead_temperature,
            "lead_score_reasoning": c.lead_score_reasoning or "",
            "lead_score_breakdown": c.lead_score_breakdown or {},
            "lead_recommended_actions": c.lead_recommended_actions or [],
            "admin_notes": getattr(c, 'admin_notes', '') or "",
            "photo_base64": getattr(c, 'photo_base64', None),
            "qr_base64": get_qr_base64(contact_id),
        }
        return {"status": "success", "contact": contact_data}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.delete("/contact/{contact_id}")
async def delete_contact_route(
    contact_id: str,
    user_id: str = Query(..., description="User ID"),
    api_key: str = Depends(verify_api_key),
):
    session = await get_db_session()
    try:
        result = await session.execute(
            select(ContactDB).where(
                ContactDB.id == uuid.UUID(contact_id),
                ContactDB.user_id == uuid.UUID(user_id),
            )
        )
        c = result.scalar_one_or_none()
        if not c:
            raise HTTPException(status_code=404, detail="Contact not found")

        contact_name = c.name
        await session.delete(c)
        await session.commit()

        # Remove QR
        qr_path = os.path.join(QR_DIR, f"qr_{contact_id}.png")
        if os.path.exists(qr_path):
            os.remove(qr_path)

        # Remove from FAISS
        faiss_index.delete_contact(user_id, contact_id)

        return {
            "status": "success",
            "message": f"Contact '{contact_name}' deleted successfully",
            "deleted_contact_id": contact_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.put("/contact/{contact_id}")
async def update_contact_route(
    contact_id: str,
    contact_update: ContactUpdate,
    user_id: str = Query(..., description="User ID"),
    api_key: str = Depends(verify_api_key),
):
    session = await get_db_session()
    try:
        result = await session.execute(
            select(ContactDB).where(
                ContactDB.id == uuid.UUID(contact_id),
                ContactDB.user_id == uuid.UUID(user_id),
            )
        )
        c = result.scalar_one_or_none()
        if not c:
            raise HTTPException(status_code=404, detail="Contact not found")

        update_data = contact_update.model_dump(exclude_unset=True)
        if not update_data:
            raise HTTPException(status_code=400, detail="No fields provided for update")

        # Apply updates
        for field, value in update_data.items():
            if value is not None:
                setattr(c, field, value)
        c.updated_at = datetime.now(timezone.utc)

        await session.commit()
        await session.refresh(c)

        # Regenerate QR
        updated_contact = Contact(
            name=c.name, email=c.email or "N/A", phone=c.phone or "N/A",
            linkedin=c.linkedin or "N/A", company_name=c.company_name or "N/A",
            notes=c.notes or "", links=c.links or [],
        )
        old_qr_path = os.path.join(QR_DIR, f"qr_{contact_id}.png")
        if os.path.exists(old_qr_path):
            os.remove(old_qr_path)
        qr_result = create_qr(updated_contact, contact_id)

        # Update FAISS index
        contact_dict = {
            "id": str(c.id),
            "name": c.name, "email": c.email or "N/A",
            "phone": c.phone or "N/A", "linkedin": c.linkedin or "N/A",
            "company_name": c.company_name or "N/A",
            "notes": c.notes or "", "links": c.links or [],
            "source": c.source or "manual",
            "lead_score": c.lead_score, "lead_temperature": c.lead_temperature,
            "lead_score_reasoning": c.lead_score_reasoning or "",
            "created_at": c.created_at.isoformat() if c.created_at else "",
        }
        faiss_index.update_contact(user_id, contact_id, contact_dict)

        return {
            "status": "success",
            "message": "Contact updated successfully",
            "contact": {
                "id": contact_id,
                "name": c.name, "email": c.email or "N/A",
                "phone": c.phone or "N/A", "linkedin": c.linkedin or "N/A",
                "company_name": c.company_name or "N/A",
                "notes": c.notes or "", "links": c.links or [],
                "qr_base64": qr_result["qr_base64"],
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.post("/converse/")
async def converse_route(query: ConverseQuery, api_key: str = Depends(verify_api_key)):
    try:
        # Check if admin mode requested — verify admin status
        is_admin_mode = False
        if query.admin_mode:
            session_check = await get_db_session()
            try:
                admin_result = await session_check.execute(
                    select(UserDB).where(UserDB.id == uuid.UUID(query.user_id))
                )
                admin_user = admin_result.scalar_one_or_none()
                is_admin_mode = admin_user and admin_user.is_admin
            finally:
                await session_check.close()

        # FAISS search — cross-user for admin mode
        if is_admin_mode:
            # Search across ALL users' FAISS indices
            all_results = []
            for uid in list(faiss_index.indices.keys()):
                try:
                    user_results = faiss_index.search(uid, query.query, k=2)
                    all_results.extend(user_results)
                except Exception:
                    pass
            # Take top results (already sorted by FAISS relevance within each user)
            retrieved_contacts = all_results[:query.top_k or 8]
        else:
            retrieved_contacts = faiss_index.search(query.user_id, query.query, k=query.top_k or 4)

        session = await get_db_session()
        try:
            # Load user profile
            result = await session.execute(
                select(UserProfileDB).where(UserProfileDB.user_id == uuid.UUID(query.user_id))
            )
            profile_row = result.scalar_one_or_none()
            user_profile = profile_row.profile_data if profile_row else {}

            # Load contacts summary — all users for admin mode, own contacts otherwise
            if is_admin_mode:
                all_contacts_result = await session.execute(select(ContactDB))
            else:
                all_contacts_result = await session.execute(
                    select(ContactDB).where(ContactDB.user_id == uuid.UUID(query.user_id))
                )
            all_contacts_rows = all_contacts_result.scalars().all()
            all_contacts = [
                {
                    "name": c.name, "company_name": c.company_name or "N/A",
                    "lead_score": c.lead_score, "lead_temperature": c.lead_temperature,
                }
                for c in all_contacts_rows
            ]

            # Always include exhibitor context for event intelligence
            exhibitors = []
            try:
                event_name = user_profile.get("current_event_name", "WHX Dubai 2026")
                ex_result = await session.execute(
                    select(ExhibitorDB).where(ExhibitorDB.event_name == event_name).limit(50)
                )
                ex_rows = ex_result.scalars().all()
                exhibitors = [
                    {
                        "name": e.name, "booth": e.booth or "", "hall": e.hall or "",
                        "category": e.category or "", "country": e.country or "",
                        "description": e.description or "",
                    }
                    for e in ex_rows
                ]
            except Exception as ex_err:
                print(f"[CHAT] Error loading exhibitors: {ex_err}")
        finally:
            await session.close()

        # Get model info for response metadata
        model_key = user_profile.get("preferred_ai_model", "claude-opus")
        model_info = AVAILABLE_MODELS.get(model_key, AVAILABLE_MODELS["claude-opus"])

        # Detect intent for research/pitch augmented prompting
        intent = detect_intent(query.query)
        augmented_query = query.query
        if intent == 'research':
            augmented_query = query.query + "\n\n" + RESEARCH_PROMPT_SUPPLEMENT
        elif intent == 'pitch':
            augmented_query = query.query + "\n\n" + PITCH_PROMPT_SUPPLEMENT

        response_text = await intelligence_engine.generate_response(
            query=augmented_query,
            retrieved_contacts=retrieved_contacts,
            conversation_history=query.conversation_history,
            user_profile=user_profile,
            all_contacts=all_contacts,
            exhibitors=exhibitors,
        )

        # Enrich retrieved contacts with full data for UI components
        enriched_contacts = []
        for text, meta in retrieved_contacts:
            enriched_contacts.append({
                "id": meta.get("id", ""),
                "name": meta.get("name", "N/A"),
                "email": meta.get("email", "N/A"),
                "phone": meta.get("phone", "N/A"),
                "linkedin": meta.get("linkedin", "N/A"),
                "company_name": meta.get("company_name", "N/A"),
                "lead_score": meta.get("lead_score"),
                "lead_temperature": meta.get("lead_temperature"),
                "lead_score_reasoning": meta.get("lead_score_reasoning", ""),
                "lead_recommended_actions": meta.get("lead_recommended_actions", []),
                "photo_base64": meta.get("photo_base64", ""),
                "source": meta.get("source", "manual"),
                "notes": (meta.get("notes", "") or "")[:200],
            })

        # Generate UI components based on context
        ui_components = generate_ui_components(
            query=query.query, response_text=response_text,
            retrieved_contacts=enriched_contacts,
            all_contacts=all_contacts,
            exhibitors=exhibitors, user_profile=user_profile,
        )

        # Add research/pitch-specific components
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
            "model": model_info["name"],
            "retrieved_contacts": enriched_contacts,
            "ui_components": ui_components,
            "intent": intent,
            "query": query.query,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Conversation error: {str(e)}")


@app.get("/export_contacts/")
async def export_contacts_route(
    user_id: str = Query(..., description="User ID"),
    format: str = Query("csv", description="Export format: csv or json"),
    api_key: str = Depends(verify_api_key),
):
    import csv
    from io import StringIO
    from fastapi.responses import StreamingResponse

    session = await get_db_session()
    try:
        result = await session.execute(
            select(ContactDB).where(ContactDB.user_id == uuid.UUID(user_id))
        )
        contacts = result.scalars().all()

        if format == "json":
            contact_list = []
            for c in contacts:
                contact_list.append({
                    "name": c.name, "email": c.email or "N/A",
                    "phone": c.phone or "N/A", "linkedin": c.linkedin or "N/A",
                    "company_name": c.company_name or "N/A",
                    "notes": c.notes or "", "links": c.links or [],
                    "source": c.source or "manual",
                })
            return {"status": "success", "total": len(contact_list), "contacts": contact_list}

        # CSV export
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["Name", "Email", "Phone", "LinkedIn", "Company", "Notes", "Source"])
        for c in contacts:
            writer.writerow([
                c.name, c.email or "N/A", c.phone or "N/A",
                c.linkedin or "N/A", c.company_name or "N/A",
                c.notes or "", c.source or "manual",
            ])

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


@app.post("/contact/{contact_id}/enrich")
async def enrich_contact_route(
    contact_id: str,
    enrich_data: EnrichRequest,
    user_id: str = Query(None, description="User ID (optional, searches all users)"),
    api_key: str = Depends(verify_api_key),
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

        owner_user_id = str(c.user_id)

        # Append notes
        if enrich_data.notes_append:
            existing_notes = c.notes or ""
            if existing_notes:
                c.notes = f"{existing_notes}\n\n---\n{enrich_data.notes_append}"
            else:
                c.notes = enrich_data.notes_append

        # Append links
        if enrich_data.links:
            existing_links = c.links or []
            for link in enrich_data.links:
                link["added_by"] = link.get("added_by", "n8n")
                existing_links.append(link)
            c.links = existing_links

        c.updated_at = datetime.now(timezone.utc)
        await session.commit()

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

        # Update FAISS
        contact_dict = {
            "id": str(c.id), "name": c.name,
            "email": c.email or "N/A", "phone": c.phone or "N/A",
            "linkedin": c.linkedin or "N/A", "company_name": c.company_name or "N/A",
            "notes": c.notes or "", "links": c.links or [],
            "source": c.source or "manual",
            "created_at": c.created_at.isoformat() if c.created_at else "",
        }
        faiss_index.update_contact(owner_user_id, contact_id, contact_dict)

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
):
    session = await get_db_session()
    try:
        result = await session.execute(
            select(UserProfileDB).where(UserProfileDB.user_id == uuid.UUID(user_id))
        )
        profile = result.scalar_one_or_none()
        return {"status": "success", "profile": profile.profile_data if profile else {}}
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
):
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

        if profile:
            existing = profile.profile_data or {}
            for key, value in update_data.items():
                if value is not None:
                    existing[key] = value
            profile.profile_data = existing
            profile.updated_at = datetime.now(timezone.utc)
        else:
            profile = UserProfileDB(
                user_id=uuid.UUID(user_id),
                profile_data={k: v for k, v in update_data.items() if v is not None},
            )
            session.add(profile)

        await session.commit()
        return {"status": "success", "profile": profile.profile_data}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


# --- LEAD SCORING ---

@app.post("/contact/{contact_id}/score")
async def score_contact_route(
    contact_id: str,
    user_id: str = Query(..., description="User ID"),
    api_key: str = Depends(verify_api_key),
):
    session = await get_db_session()
    try:
        result = await session.execute(
            select(ContactDB).where(ContactDB.id == uuid.UUID(contact_id), ContactDB.user_id == uuid.UUID(user_id))
        )
        c = result.scalar_one_or_none()
        if not c:
            raise HTTPException(status_code=404, detail="Contact not found")

        # Get user profile
        profile_result = await session.execute(
            select(UserProfileDB).where(UserProfileDB.user_id == uuid.UUID(user_id))
        )
        profile = profile_result.scalar_one_or_none()
        user_profile = profile.profile_data if profile else {}

        contact_meta = {
            "name": c.name, "company_name": c.company_name or "N/A",
            "email": c.email or "N/A", "phone": c.phone or "N/A",
            "linkedin": c.linkedin or "N/A", "notes": c.notes or "",
        }
        try:
            score_result = await asyncio.wait_for(
                asyncio.to_thread(score_contact_with_gemini, contact_meta, user_profile),
                timeout=30
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Scoring timed out — Gemini API unresponsive")

        # Save score to DB
        c.lead_score = score_result["score"]
        c.lead_temperature = score_result["temperature"]
        c.lead_score_reasoning = score_result["reasoning"]
        c.lead_score_breakdown = score_result.get("breakdown", {})
        c.lead_recommended_actions = score_result.get("recommended_actions", [])
        c.updated_at = datetime.now(timezone.utc)
        await session.commit()

        # Update FAISS
        contact_dict = {
            "id": str(c.id), "name": c.name,
            "email": c.email or "N/A", "phone": c.phone or "N/A",
            "linkedin": c.linkedin or "N/A", "company_name": c.company_name or "N/A",
            "notes": c.notes or "", "links": c.links or [],
            "lead_score": c.lead_score, "lead_temperature": c.lead_temperature,
            "lead_score_reasoning": c.lead_score_reasoning or "",
        }
        faiss_index.update_contact(user_id, contact_id, contact_dict)

        return {
            "status": "success",
            "contact_id": contact_id,
            "score": score_result["score"],
            "temperature": score_result["temperature"],
            "reasoning": score_result["reasoning"],
            "recommended_actions": score_result.get("recommended_actions", []),
            "breakdown": score_result.get("breakdown", {}),
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.post("/contacts/score_all")
async def score_all_contacts_route(
    user_id: str = Query(..., description="User ID"),
    api_key: str = Depends(verify_api_key),
):
    session = await get_db_session()
    try:
        result = await session.execute(
            select(ContactDB).where(
                ContactDB.user_id == uuid.UUID(user_id),
                ContactDB.lead_score.is_(None),
            )
        )
        unscored = result.scalars().all()

        profile_result = await session.execute(
            select(UserProfileDB).where(UserProfileDB.user_id == uuid.UUID(user_id))
        )
        profile = profile_result.scalar_one_or_none()
        user_profile = profile.profile_data if profile else {}

        scored = 0
        errors = 0
        for c in unscored:
            try:
                contact_meta = {
                    "name": c.name, "company_name": c.company_name or "N/A",
                    "email": c.email or "N/A", "phone": c.phone or "N/A",
                    "linkedin": c.linkedin or "N/A", "notes": c.notes or "",
                }
                score_result = await asyncio.wait_for(
                    asyncio.to_thread(score_contact_with_gemini, contact_meta, user_profile),
                    timeout=30
                )
                c.lead_score = score_result["score"]
                c.lead_temperature = score_result["temperature"]
                c.lead_score_reasoning = score_result["reasoning"]
                c.lead_score_breakdown = score_result.get("breakdown", {})
                c.lead_recommended_actions = score_result.get("recommended_actions", [])
                scored += 1
            except (asyncio.TimeoutError, Exception) as e:
                print(f"[SCORING] Error scoring {c.id}: {e}")
                errors += 1

        await session.commit()

        # Count total
        total_result = await session.execute(
            select(func.count(ContactDB.id)).where(ContactDB.user_id == uuid.UUID(user_id))
        )
        total = total_result.scalar() or 0

        return {"status": "success", "scored": scored, "skipped": total - scored - errors, "errors": errors, "total": total}
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.get("/dashboard")
async def dashboard_route(
    user_id: str = Query(..., description="User ID"),
    api_key: str = Depends(verify_api_key),
):
    session = await get_db_session()
    try:
        result = await session.execute(
            select(ContactDB).where(ContactDB.user_id == uuid.UUID(user_id))
        )
        contacts = result.scalars().all()

        total = len(contacts)
        scored = sum(1 for c in contacts if c.lead_score is not None)
        hot = sum(1 for c in contacts if c.lead_temperature == "hot")
        warm = sum(1 for c in contacts if c.lead_temperature == "warm")
        cold = sum(1 for c in contacts if c.lead_temperature == "cold")

        scored_contacts = sorted(
            [c for c in contacts if c.lead_score is not None],
            key=lambda x: x.lead_score or 0,
            reverse=True,
        )
        top_contacts = [
            {
                "id": str(c.id), "name": c.name,
                "company_name": c.company_name or "N/A",
                "lead_score": c.lead_score, "lead_temperature": c.lead_temperature,
            }
            for c in scored_contacts[:5]
        ]

        profile_result = await session.execute(
            select(UserProfileDB).where(UserProfileDB.user_id == uuid.UUID(user_id))
        )
        profile = profile_result.scalar_one_or_none()
        user_profile = profile.profile_data if profile else {}

        return {
            "status": "success",
            "total_contacts": total,
            "scored_contacts": scored,
            "unscored_contacts": total - scored,
            "by_temperature": {"hot": hot, "warm": warm, "cold": cold},
            "top_contacts": top_contacts,
            "has_profile": bool(user_profile),
            "current_event": user_profile.get("current_event_name", ""),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


# --- SHARED CONTACTS (admin/backend access) ---

@app.get("/shared_contacts/")
async def list_shared_contacts(
    api_key: str = Depends(verify_api_key),
    limit: int = Query(100, description="Max contacts to return"),
):
    """List all scanned contacts across all users (shared pool)."""
    session = await get_db_session()
    try:
        result = await session.execute(
            select(SharedContactDB).order_by(SharedContactDB.created_at.desc()).limit(limit)
        )
        contacts = result.scalars().all()
        return {
            "status": "success",
            "total": len(contacts),
            "contacts": [
                {
                    "id": str(c.id),
                    "original_contact_id": str(c.original_contact_id) if c.original_contact_id else None,
                    "user_id": str(c.user_id) if c.user_id else None,
                    "name": c.name, "email": c.email or "N/A",
                    "phone": c.phone or "N/A", "linkedin": c.linkedin or "N/A",
                    "company_name": c.company_name or "N/A",
                    "notes": c.notes or "", "links": c.links or [],
                    "source": c.source or "manual",
                    "webhook_sent": c.webhook_sent,
                    "enriched": c.enriched,
                    "created_at": c.created_at.isoformat() if c.created_at else "",
                }
                for c in contacts
            ],
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


# --- AI MODEL ENDPOINTS ---

@app.get("/models/")
async def list_models():
    """Return available AI models for chat."""
    return {
        "status": "success",
        "models": [
            {"key": key, **info}
            for key, info in AVAILABLE_MODELS.items()
        ],
        "openrouter_configured": bool(OPENROUTER_API_KEY),
    }


# --- EXHIBITOR ENDPOINTS ---

class ExhibitorImport(BaseModel):
    exhibitors: List[Dict[str, Any]]


@app.post("/admin/import_exhibitors/")
async def import_exhibitors_route(
    data: ExhibitorImport,
    api_key: str = Depends(verify_api_key),
):
    """Bulk import exhibitor data (admin endpoint)."""
    session = await get_db_session()
    try:
        imported = 0
        skipped = 0
        for ex in data.exhibitors:
            name = ex.get("name", "").strip()
            if not name:
                skipped += 1
                continue

            # Check if already exists
            result = await session.execute(
                select(ExhibitorDB).where(
                    ExhibitorDB.name == name,
                    ExhibitorDB.event_name == ex.get("event_name", "WHX Dubai 2026"),
                )
            )
            if result.scalar_one_or_none():
                skipped += 1
                continue

            db_ex = ExhibitorDB(
                event_name=ex.get("event_name", "WHX Dubai 2026"),
                name=name,
                booth=ex.get("booth", ""),
                hall=ex.get("hall", ""),
                category=ex.get("category", ""),
                subcategory=ex.get("subcategory", ""),
                country=ex.get("country", ""),
                website=ex.get("website", ""),
                description=ex.get("description", ""),
                products=ex.get("products", []),
                tags=ex.get("tags", []),
            )
            session.add(db_ex)
            imported += 1

        await session.commit()
        return {"status": "success", "imported": imported, "skipped": skipped}
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.get("/exhibitors/")
async def list_exhibitors_route(
    event: str = Query("WHX Dubai 2026", description="Event name (defaults to WHX Dubai 2026)"),
    category: str = Query("", description="Filter by category"),
    search: str = Query("", description="Search exhibitor names"),
    limit: int = Query(100, description="Max results"),
    api_key: str = Depends(verify_api_key),
):
    """List exhibitors with optional filtering by event, category, and search."""
    session = await get_db_session()
    try:
        query = select(ExhibitorDB).where(ExhibitorDB.event_name == event)
        if category:
            query = query.where(ExhibitorDB.category.ilike(f"%{category}%"))
        if search:
            query = query.where(ExhibitorDB.name.ilike(f"%{search}%"))
        query = query.order_by(ExhibitorDB.name).limit(limit)

        result = await session.execute(query)
        exhibitors = result.scalars().all()

        return {
            "status": "success",
            "total": len(exhibitors),
            "exhibitors": [
                {
                    "id": str(e.id),
                    "name": e.name,
                    "booth": e.booth or "",
                    "hall": e.hall or "",
                    "category": e.category or "",
                    "subcategory": e.subcategory or "",
                    "country": e.country or "",
                    "website": e.website or "",
                    "description": e.description or "",
                    "products": e.products or [],
                    "tags": e.tags or [],
                }
                for e in exhibitors
            ],
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.get("/exhibitors/categories/")
async def list_exhibitor_categories(
    event: str = Query("WHX Dubai 2026", description="Event name (defaults to WHX Dubai 2026)"),
    api_key: str = Depends(verify_api_key),
):
    """List unique exhibitor categories for filtering by event."""
    session = await get_db_session()
    try:
        result = await session.execute(
            select(ExhibitorDB.category).where(
                ExhibitorDB.event_name == event,
                ExhibitorDB.category != "",
            ).distinct()
        )
        categories = sorted([row[0] for row in result.all()])
        return {"status": "success", "categories": categories}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


# --- ADMIN ENDPOINTS ---

@app.get("/admin/users")
async def admin_list_users(
    admin_id: str = Depends(verify_admin),
    action: str = Query(None),
    target_user_id: str = Query(None),
):
    """List all users or delete a specific user (action=delete&target_user_id=xxx)."""
    session = await get_db_session()
    try:
        # Handle delete action
        if action == "delete" and target_user_id:
            if target_user_id == admin_id:
                raise HTTPException(status_code=400, detail="Cannot delete your own admin account")
            result = await session.execute(
                select(UserDB).where(UserDB.id == uuid.UUID(target_user_id))
            )
            user = result.scalar_one_or_none()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            user_name = user.name
            try:
                faiss_index.delete_user(target_user_id)
            except Exception as faiss_err:
                print(f"[FAISS] Error deleting user index: {faiss_err}")
            await session.execute(
                delete(ContactDB).where(ContactDB.user_id == uuid.UUID(target_user_id))
            )
            await session.execute(
                delete(ConversationDB).where(ConversationDB.user_id == uuid.UUID(target_user_id))
            )
            await session.execute(
                delete(UserProfileDB).where(UserProfileDB.user_id == uuid.UUID(target_user_id))
            )
            await session.execute(
                delete(UserCardDB).where(UserCardDB.user_id == uuid.UUID(target_user_id))
            )
            await session.delete(user)
            await session.commit()
            return {"status": "success", "message": f"User '{user_name}' and all their data deleted", "user_id": target_user_id}

        # Default: list all users
        result = await session.execute(select(UserDB))
        users = result.scalars().all()

        user_list = []
        for u in users:
            count_result = await session.execute(
                select(func.count(ContactDB.id)).where(ContactDB.user_id == u.id)
            )
            contact_count = count_result.scalar() or 0
            user_list.append({
                "user_id": str(u.id),
                "name": u.name,
                "email": u.email,
                "is_admin": u.is_admin,
                "contact_count": contact_count,
                "created_at": u.created_at.isoformat() if u.created_at else "",
            })

        return {"status": "success", "users": user_list, "total": len(user_list)}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()




@app.get("/admin/contacts")
async def admin_list_contacts(
    admin_id: str = Depends(verify_admin),
    filter_user: Optional[str] = Query(None, description="Filter by user_id"),
    limit: int = Query(200, description="Max contacts"),
):
    """List all contacts across all users, optionally filtered."""
    session = await get_db_session()
    try:
        query = select(ContactDB).order_by(ContactDB.created_at.desc()).limit(limit)
        if filter_user:
            query = query.where(ContactDB.user_id == uuid.UUID(filter_user))

        result = await session.execute(query)
        contacts = result.scalars().all()

        # Get user names for display
        user_ids = list(set(str(c.user_id) for c in contacts))
        user_map = {}
        if user_ids:
            users_result = await session.execute(select(UserDB).where(UserDB.id.in_([uuid.UUID(uid) for uid in user_ids])))
            for u in users_result.scalars().all():
                user_map[str(u.id)] = u.name

        return {
            "status": "success",
            "total": len(contacts),
            "contacts": [
                {
                    "id": str(c.id),
                    "user_id": str(c.user_id),
                    "user_name": user_map.get(str(c.user_id), "Unknown"),
                    "name": c.name, "email": c.email or "N/A",
                    "phone": c.phone or "N/A", "company_name": c.company_name or "N/A",
                    "linkedin": c.linkedin or "N/A",
                    "lead_score": c.lead_score, "lead_temperature": c.lead_temperature,
                    "lead_score_reasoning": c.lead_score_reasoning or "",
                    "lead_recommended_actions": c.lead_recommended_actions or [],
                    "source": c.source or "manual",
                    "notes": c.notes or "",
                    "admin_notes": getattr(c, 'admin_notes', '') or "",
                    "links": c.links or [],
                    "audio_notes_count": len(c.audio_notes) if c.audio_notes else 0,
                    "created_at": c.created_at.isoformat() if c.created_at else "",
                    "updated_at": c.updated_at.isoformat() if c.updated_at else "",
                }
                for c in contacts
            ],
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.get("/admin/activity")
async def admin_activity_feed(
    admin_id: str = Depends(verify_admin),
    limit: int = Query(50, description="Number of recent activities"),
):
    """Real-time activity feed for admin dashboard monitoring."""
    session = await get_db_session()
    try:
        # Get recent contacts with user info
        result = await session.execute(
            select(ContactDB, UserDB.name.label('user_name'))
            .join(UserDB, ContactDB.user_id == UserDB.id)
            .order_by(ContactDB.created_at.desc())
            .limit(limit)
        )

        activities = []
        for contact, user_name in result:
            activities.append({
                "type": "contact_added",
                "contact_id": str(contact.id),
                "contact_name": contact.name,
                "company": contact.company_name or "N/A",
                "user_id": str(contact.user_id),
                "user_name": user_name,
                "source": contact.source or "manual",
                "timestamp": contact.created_at.isoformat() if contact.created_at else "",
                "lead_temperature": contact.lead_temperature or "unscored",
                "lead_score": contact.lead_score,
            })

        return {"status": "success", "activities": activities}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.delete("/admin/contact/{contact_id}")
async def admin_delete_contact(
    contact_id: str,
    admin_id: str = Depends(verify_admin),
):
    """Admin endpoint to delete any contact across all users."""
    session = await get_db_session()
    try:
        result = await session.execute(
            select(ContactDB).where(ContactDB.id == uuid.UUID(contact_id))
        )
        contact = result.scalar_one_or_none()

        if not contact:
            raise HTTPException(status_code=404, detail="Contact not found")

        # Delete from FAISS
        try:
            faiss_index.delete_contact(str(contact.user_id), contact_id)
        except Exception as faiss_err:
            print(f"[FAISS] Failed to delete contact {contact_id}: {faiss_err}")

        # Delete from DB
        await session.delete(contact)
        await session.commit()

        return {"status": "success", "message": "Contact deleted", "contact_id": contact_id}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


# ==================== ADMIN MERGE & DEDUP ENDPOINTS ====================

class MergeUsersRequest(BaseModel):
    primary_user_id: str
    secondary_user_id: str

class MergeContactsRequest(BaseModel):
    primary_id: str
    secondary_id: str

class BatchDeleteRequest(BaseModel):
    contact_ids: List[str]


@app.post("/admin/users/merge")
async def admin_merge_users(req: MergeUsersRequest, admin_id: str = Depends(verify_admin)):
    """Merge two user accounts: move all contacts from secondary to primary, delete secondary."""
    if req.primary_user_id == req.secondary_user_id:
        raise HTTPException(status_code=400, detail="Cannot merge a user with themselves")

    session = await get_db_session()
    try:
        # Load both users
        p_result = await session.execute(select(UserDB).where(UserDB.id == uuid.UUID(req.primary_user_id)))
        primary = p_result.scalar_one_or_none()
        if not primary:
            raise HTTPException(status_code=404, detail="Primary user not found")

        s_result = await session.execute(select(UserDB).where(UserDB.id == uuid.UUID(req.secondary_user_id)))
        secondary = s_result.scalar_one_or_none()
        if not secondary:
            raise HTTPException(status_code=404, detail="Secondary user not found")

        primary_uuid = uuid.UUID(req.primary_user_id)
        secondary_uuid = uuid.UUID(req.secondary_user_id)

        # Count contacts being transferred
        count_result = await session.execute(
            select(func.count(ContactDB.id)).where(ContactDB.user_id == secondary_uuid)
        )
        contacts_transferred = count_result.scalar() or 0

        # Move all contacts from secondary to primary
        await session.execute(
            update(ContactDB).where(ContactDB.user_id == secondary_uuid).values(user_id=primary_uuid)
        )

        # Move conversations
        await session.execute(
            update(ConversationDB).where(ConversationDB.user_id == secondary_uuid).values(user_id=primary_uuid)
        )

        # Move contact pipelines (has user_id column)
        try:
            await session.execute(
                update(ContactPipelineDB).where(ContactPipelineDB.user_id == secondary_uuid).values(user_id=primary_uuid)
            )
        except Exception:
            pass  # Pipeline table may not have user_id FK

        # Delete secondary user's profile and card (non-critical)
        try:
            await session.execute(delete(UserProfileDB).where(UserProfileDB.user_id == secondary_uuid))
        except Exception:
            pass
        try:
            await session.execute(delete(UserCardDB).where(UserCardDB.user_id == secondary_uuid))
        except Exception:
            pass

        # Delete the secondary user
        await session.delete(secondary)
        await session.commit()

        # Rebuild FAISS for primary user with all their contacts (including transferred ones)
        try:
            result = await session.execute(
                select(ContactDB).where(ContactDB.user_id == primary_uuid)
            )
            all_contacts = result.scalars().all()
            contact_dicts = []
            for c in all_contacts:
                contact_dicts.append({
                    "id": str(c.id),
                    "name": c.name or "N/A",
                    "email": c.email or "N/A",
                    "phone": c.phone or "N/A",
                    "company_name": c.company_name or "N/A",
                    "notes": c.notes or "",
                })
            faiss_index.build_for_user(req.primary_user_id, contact_dicts)
            faiss_index.delete_user(req.secondary_user_id)
        except Exception as faiss_err:
            print(f"[FAISS] Error rebuilding after user merge: {faiss_err}")

        return {
            "status": "success",
            "message": f"User '{secondary.name}' merged into '{primary.name}'. {contacts_transferred} contacts transferred.",
            "primary_user_id": req.primary_user_id,
            "deleted_user_id": req.secondary_user_id,
            "contacts_transferred": contacts_transferred,
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.post("/admin/contacts/merge")
async def admin_merge_contacts(req: MergeContactsRequest, admin_id: str = Depends(verify_admin)):
    """Merge two duplicate contacts: absorb secondary into primary, delete secondary."""
    if req.primary_id == req.secondary_id:
        raise HTTPException(status_code=400, detail="Cannot merge a contact with itself")

    session = await get_db_session()
    try:
        p_result = await session.execute(select(ContactDB).where(ContactDB.id == uuid.UUID(req.primary_id)))
        primary = p_result.scalar_one_or_none()
        if not primary:
            raise HTTPException(status_code=404, detail="Primary contact not found")

        s_result = await session.execute(select(ContactDB).where(ContactDB.id == uuid.UUID(req.secondary_id)))
        secondary = s_result.scalar_one_or_none()
        if not secondary:
            raise HTTPException(status_code=404, detail="Secondary contact not found")

        # Merge fields: keep primary's value unless it's "N/A" or empty
        def pick(a, b):
            if a and a != "N/A" and a.strip():
                return a
            return b if b and b != "N/A" and b.strip() else a

        primary.name = pick(primary.name, secondary.name)
        primary.email = pick(primary.email, secondary.email)
        primary.phone = pick(primary.phone, secondary.phone)
        primary.linkedin = pick(primary.linkedin, secondary.linkedin)
        primary.company_name = pick(primary.company_name, secondary.company_name)

        # Concatenate notes
        if secondary.notes and secondary.notes.strip():
            existing = primary.notes or ""
            primary.notes = (existing + "\n---\n" + secondary.notes).strip() if existing.strip() else secondary.notes

        if secondary.admin_notes and secondary.admin_notes.strip():
            existing = primary.admin_notes or ""
            primary.admin_notes = (existing + "\n---\n" + secondary.admin_notes).strip() if existing.strip() else secondary.admin_notes

        # Combine JSON lists (copy to avoid mutation bug)
        merged_links = list(primary.links or [])
        existing_urls = {l.get("url") for l in merged_links if isinstance(l, dict)}
        for link in (secondary.links or []):
            if isinstance(link, dict) and link.get("url") not in existing_urls:
                merged_links.append(link)
        primary.links = merged_links

        merged_audio = list(primary.audio_notes or [])
        merged_audio.extend(secondary.audio_notes or [])
        primary.audio_notes = merged_audio

        merged_actions = list(primary.lead_recommended_actions or [])
        for action in (secondary.lead_recommended_actions or []):
            if action not in merged_actions:
                merged_actions.append(action)
        primary.lead_recommended_actions = merged_actions

        # Keep higher score
        p_score = primary.lead_score or 0
        s_score = secondary.lead_score or 0
        if s_score > p_score:
            primary.lead_score = secondary.lead_score
            primary.lead_temperature = secondary.lead_temperature
            primary.lead_score_reasoning = secondary.lead_score_reasoning
            primary.lead_score_breakdown = dict(secondary.lead_score_breakdown or {})

        # Keep photo
        if not primary.photo_base64 and secondary.photo_base64:
            primary.photo_base64 = secondary.photo_base64

        # Keep earlier created_at
        if secondary.created_at and (not primary.created_at or secondary.created_at < primary.created_at):
            primary.created_at = secondary.created_at

        # Reassign secondary's files and pipelines to primary
        try:
            await session.execute(
                update(ContactFileDB).where(ContactFileDB.contact_id == uuid.UUID(req.secondary_id)).values(contact_id=uuid.UUID(req.primary_id))
            )
        except Exception:
            pass
        try:
            await session.execute(
                update(ContactPipelineDB).where(ContactPipelineDB.contact_id == uuid.UUID(req.secondary_id)).values(contact_id=uuid.UUID(req.primary_id))
            )
        except Exception:
            pass

        # Delete secondary contact
        user_id_str = str(primary.user_id)
        await session.delete(secondary)
        await session.commit()

        # Update FAISS (async to prevent event loop blocking)
        try:
            await asyncio.to_thread(faiss_index.delete_contact, user_id_str, req.secondary_id)
        except Exception as faiss_err:
            print(f"[FAISS] Error deleting merged contact: {faiss_err}")

        return {
            "status": "success",
            "message": f"Contacts merged: '{primary.name}' kept, duplicate deleted",
            "merged_contact_id": req.primary_id,
            "deleted_contact_id": req.secondary_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


def normalize_phone(phone: str) -> str:
    """Strip non-digit chars for phone comparison."""
    if not phone or phone == "N/A":
        return ""
    return re.sub(r'[^\d+]', '', phone)


@app.get("/admin/contacts/duplicates")
async def admin_find_duplicates(
    target_user_id: str = Query(..., description="User ID to check for duplicates"),
    limit: int = Query(500, description="Max contacts to check (recent first)"),
    admin_id: str = Depends(verify_admin),
):
    """Find duplicate contacts within a user's contact list."""
    session = await get_db_session()
    try:
        result = await session.execute(
            select(ContactDB)
            .where(ContactDB.user_id == uuid.UUID(target_user_id))
            .order_by(ContactDB.created_at.desc())  # Most recent first
            .limit(limit)
        )
        contacts = result.scalars().all()

        if not contacts:
            return {"status": "success", "duplicate_groups": [], "total_groups": 0, "total_duplicate_contacts": 0}

        # Build contact info list
        contact_list = []
        for c in contacts:
            contact_list.append({
                "id": str(c.id),
                "name": c.name or "N/A",
                "email": (c.email or "N/A").strip().lower(),
                "phone": normalize_phone(c.phone or "N/A"),
                "company_name": (c.company_name or "N/A").strip().lower(),
                "lead_score": c.lead_score,
                "lead_temperature": c.lead_temperature,
                "source": c.source or "unknown",
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "photo_base64": bool(c.photo_base64),
            })

        # Find groups by matching criteria
        # Use union-find to merge overlapping groups
        parent = {c["id"]: c["id"] for c in contact_list}
        match_reasons = {}  # contact_id -> set of reasons

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b, reason):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
            match_reasons.setdefault(a, set()).add(reason)
            match_reasons.setdefault(b, set()).add(reason)

        # Group by email
        email_map = {}
        for c in contact_list:
            if c["email"] and c["email"] != "n/a":
                email_map.setdefault(c["email"], []).append(c["id"])
        for email, ids in email_map.items():
            for i in range(1, len(ids)):
                union(ids[0], ids[i], f"email: {email}")

        # Group by phone
        phone_map = {}
        for c in contact_list:
            if c["phone"] and len(c["phone"]) >= 7:
                phone_map.setdefault(c["phone"], []).append(c["id"])
        for phone, ids in phone_map.items():
            for i in range(1, len(ids)):
                union(ids[0], ids[i], f"phone: {phone}")

        # Group by name + company
        name_co_map = {}
        for c in contact_list:
            if c["name"] != "n/a" and c["company_name"] != "n/a":
                key = (c["name"].strip().lower(), c["company_name"].strip().lower())
                name_co_map.setdefault(key, []).append(c["id"])
        for (name, co), ids in name_co_map.items():
            for i in range(1, len(ids)):
                union(ids[0], ids[i], f"name+company: {name} @ {co}")

        # Collect groups
        groups_map = {}
        contact_by_id = {c["id"]: c for c in contact_list}
        for c in contact_list:
            root = find(c["id"])
            groups_map.setdefault(root, []).append(c["id"])

        # Filter to groups with 2+ members
        duplicate_groups = []
        for group_id, (root, member_ids) in enumerate(groups_map.items(), 1):
            if len(member_ids) < 2:
                continue
            all_reasons = set()
            for mid in member_ids:
                all_reasons.update(match_reasons.get(mid, set()))
            duplicate_groups.append({
                "group_id": group_id,
                "match_reasons": sorted(all_reasons),
                "contacts": [contact_by_id[mid] for mid in member_ids],
            })

        total_dupes = sum(len(g["contacts"]) for g in duplicate_groups)
        return {
            "status": "success",
            "duplicate_groups": duplicate_groups,
            "total_groups": len(duplicate_groups),
            "total_duplicate_contacts": total_dupes,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.post("/admin/contacts/batch_delete")
async def admin_batch_delete_contacts(req: BatchDeleteRequest, admin_id: str = Depends(verify_admin)):
    """Delete multiple contacts at once."""
    if not req.contact_ids:
        raise HTTPException(status_code=400, detail="No contact IDs provided")

    session = await get_db_session()
    try:
        uuids = [uuid.UUID(cid) for cid in req.contact_ids]

        # Load contacts to get user_ids for FAISS cleanup
        result = await session.execute(select(ContactDB).where(ContactDB.id.in_(uuids)))
        contacts = result.scalars().all()

        if not contacts:
            raise HTTPException(status_code=404, detail="No contacts found")

        # Delete related records first
        await session.execute(delete(ContactFileDB).where(ContactFileDB.contact_id.in_(uuids)))
        await session.execute(delete(ContactPipelineDB).where(ContactPipelineDB.contact_id.in_(uuids)))

        # Track user_ids for FAISS cleanup
        user_contact_map = {}
        for c in contacts:
            uid = str(c.user_id)
            user_contact_map.setdefault(uid, []).append(str(c.id))

        # Delete contacts
        await session.execute(delete(ContactDB).where(ContactDB.id.in_(uuids)))
        await session.commit()

        # Clean FAISS
        for uid, cids in user_contact_map.items():
            for cid in cids:
                try:
                    faiss_index.delete_contact(uid, cid)
                except Exception:
                    pass

        return {
            "status": "success",
            "deleted_count": len(contacts),
            "deleted_ids": [str(c.id) for c in contacts],
            "message": f"{len(contacts)} contacts deleted",
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.post("/admin/webhook/test")
async def test_webhook_connection(
    request: Request,
    admin_id: str = Depends(verify_admin),
    webhook_url: Optional[str] = Query(None, description="Override webhook URL for testing"),
    save: bool = Query(False, description="Save as new default webhook URL"),
):
    """Test n8n webhook connectivity from admin dashboard. Optionally override/save URL."""
    global WEBHOOK_URL

    target_url = webhook_url or WEBHOOK_URL

    if save and webhook_url:
        WEBHOOK_URL = webhook_url  # Update in-memory for this instance

    if not target_url:
        return {
            "status": "error",
            "message": "No webhook URL configured. Enter a URL above and test."
        }

    # Use custom payload if provided in request body, otherwise default test payload
    try:
        body = await request.json()
    except Exception:
        body = None

    payload = body or {
        "test": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": "Test webhook from Event Scout admin dashboard",
        "source": "admin_test"
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(target_url, json=payload)

            success = response.status_code == 200
            return {
                "status": "success" if success else "error",
                "status_code": response.status_code,
                "message": "Webhook reachable" if success else f"Webhook returned HTTP {response.status_code}",
                "webhook_url": target_url
            }
    except httpx.TimeoutException:
        return {
            "status": "error",
            "message": "Webhook timeout after 10 seconds",
            "webhook_url": target_url
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Webhook error: {str(e)}",
            "webhook_url": target_url
        }


@app.get("/admin/dashboard")
async def admin_dashboard(admin_id: str = Depends(verify_admin)):
    """Global admin dashboard with stats across all users."""
    session = await get_db_session()
    try:
        # Total users
        user_count = (await session.execute(select(func.count(UserDB.id)))).scalar() or 0

        # Total contacts
        contact_count = (await session.execute(select(func.count(ContactDB.id)))).scalar() or 0

        # Temperature breakdown
        hot = (await session.execute(
            select(func.count(ContactDB.id)).where(ContactDB.lead_temperature == "hot")
        )).scalar() or 0
        warm = (await session.execute(
            select(func.count(ContactDB.id)).where(ContactDB.lead_temperature == "warm")
        )).scalar() or 0
        cold = (await session.execute(
            select(func.count(ContactDB.id)).where(ContactDB.lead_temperature == "cold")
        )).scalar() or 0

        # Per-user summary
        users_result = await session.execute(select(UserDB))
        users = users_result.scalars().all()
        per_user = []
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        for u in users:
            cnt = (await session.execute(
                select(func.count(ContactDB.id)).where(ContactDB.user_id == u.id)
            )).scalar() or 0
            # Per-user temperature breakdown
            u_hot = (await session.execute(
                select(func.count(ContactDB.id)).where(ContactDB.user_id == u.id, ContactDB.lead_temperature == "hot")
            )).scalar() or 0
            u_warm = (await session.execute(
                select(func.count(ContactDB.id)).where(ContactDB.user_id == u.id, ContactDB.lead_temperature == "warm")
            )).scalar() or 0
            u_cold = (await session.execute(
                select(func.count(ContactDB.id)).where(ContactDB.user_id == u.id, ContactDB.lead_temperature == "cold")
            )).scalar() or 0
            # Today's contacts
            today_cnt = (await session.execute(
                select(func.count(ContactDB.id)).where(ContactDB.user_id == u.id, ContactDB.created_at >= today_start)
            )).scalar() or 0
            # Last scan time
            last_scan_result = await session.execute(
                select(ContactDB.created_at).where(ContactDB.user_id == u.id).order_by(ContactDB.created_at.desc()).limit(1)
            )
            last_scan_row = last_scan_result.scalar_one_or_none()
            per_user.append({
                "user_id": str(u.id), "name": u.name, "email": u.email,
                "contact_count": cnt, "is_admin": u.is_admin,
                "hot_count": u_hot, "warm_count": u_warm, "cold_count": u_cold,
                "today_count": today_cnt,
                "last_scan_at": last_scan_row.isoformat() if last_scan_row else None,
            })

        # Recent 10 contacts
        recent_result = await session.execute(
            select(ContactDB).order_by(ContactDB.created_at.desc()).limit(10)
        )
        recent = recent_result.scalars().all()

        # Map user names
        user_map = {str(u.id): u.name for u in users}

        return {
            "status": "success",
            "total_users": user_count,
            "total_contacts": contact_count,
            "by_temperature": {"hot": hot, "warm": warm, "cold": cold},
            "per_user": per_user,
            "recent_contacts": [
                {
                    "id": str(c.id),
                    "name": c.name, "company_name": c.company_name or "N/A",
                    "user_name": user_map.get(str(c.user_id), "Unknown"),
                    "lead_score": c.lead_score, "lead_temperature": c.lead_temperature,
                    "source": c.source or "manual",
                    "created_at": c.created_at.isoformat() if c.created_at else "",
                }
                for c in recent
            ],
            "webhook_configured": bool(WEBHOOK_URL),
            "webhook_url_preview": WEBHOOK_URL[:50] + "..." if len(WEBHOOK_URL) > 50 else WEBHOOK_URL if WEBHOOK_URL else None,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.post("/admin/test_webhook")
async def admin_test_webhook(admin_id: str = Depends(verify_admin)):
    """Send a test greeting message to the configured webhook endpoint."""
    if not WEBHOOK_URL:
        return {
            "status": "error",
            "webhook_sent": False,
            "webhook_message": "WEBHOOK_URL not configured. Set it in Railway environment variables.",
        }

    try:
        test_payload = {
            "type": "test",
            "message": "Hello! This is a test message from Event Scout Admin Dashboard.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "admin_id": admin_id,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                WEBHOOK_URL,
                json=test_payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code < 400:
                return {
                    "status": "success",
                    "webhook_sent": True,
                    "webhook_message": f"Test webhook sent successfully! Response status: {response.status_code}",
                    "response_status": response.status_code,
                }
            else:
                return {
                    "status": "error",
                    "webhook_sent": False,
                    "webhook_message": f"Webhook endpoint returned error: {response.status_code}",
                    "response_status": response.status_code,
                }
    except httpx.TimeoutException:
        return {
            "status": "error",
            "webhook_sent": False,
            "webhook_message": "Webhook request timed out (10s). Check n8n endpoint availability.",
        }
    except Exception as e:
        print(f"[WEBHOOK TEST ERROR] {str(e)}")
        traceback.print_exc()
        return {
            "status": "error",
            "webhook_sent": False,
            "webhook_message": f"Webhook test failed: {str(e)}",
        }


# --- EVENT FILES ENDPOINTS ---

@app.post("/admin/files/upload")
async def upload_event_file(
    file: UploadFile = File(...),
    description: str = Query(""),
    event_name: str = Query("WHX Dubai 2026"),
    category: str = Query("general"),
    user_id: str = Query(...),
    api_key: str = Depends(verify_api_key),
):
    """Upload event file (PDF/PPT). Admin only. Max 20MB."""
    await verify_admin(user_id, api_key)

    allowed_types = {
        'application/pdf': 'pdf',
        'application/vnd.ms-powerpoint': 'ppt',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
    }

    content_type = file.content_type or ''
    if content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="File type not allowed. Supported: PDF, PPT, PPTX")

    content = await file.read()
    file_size = len(content)

    if file_size == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if file_size > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File too large. Max 20MB, got {file_size / 1024 / 1024:.1f}MB")

    original_filename = file.filename or "untitled"
    safe_filename = re.sub(r'[^\w\s\-\.]', '_', original_filename)

    session = await get_db_session()
    try:
        db_file = EventFileDB(
            filename=safe_filename,
            original_filename=original_filename,
            file_type=allowed_types[content_type],
            mime_type=content_type,
            file_size=file_size,
            file_data=content,
            description=description,
            event_name=event_name,
            category=category,
            uploaded_by=uuid.UUID(user_id),
        )
        session.add(db_file)
        await session.commit()

        return {
            "status": "success",
            "file_id": str(db_file.id),
            "filename": safe_filename,
            "file_size": file_size,
            "message": f"File '{original_filename}' uploaded successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.get("/files/list")
async def list_event_files(
    event_name: str = Query("WHX Dubai 2026"),
    category: str = Query(""),
    user_id: str = Query(...),
    api_key: str = Depends(verify_api_key),
):
    """List available event files for download."""
    session = await get_db_session()
    try:
        query = select(EventFileDB).where(
            EventFileDB.event_name == event_name,
            EventFileDB.is_active == True
        )

        if category:
            query = query.where(EventFileDB.category == category)

        query = query.order_by(EventFileDB.created_at.desc())
        result = await session.execute(query)
        files = result.scalars().all()

        return {
            "status": "success",
            "total": len(files),
            "files": [
                {
                    "id": str(f.id),
                    "filename": f.original_filename,
                    "file_type": f.file_type,
                    "file_size": f.file_size,
                    "file_size_mb": round(f.file_size / 1024 / 1024, 2),
                    "description": f.description or "",
                    "category": f.category or "general",
                    "download_count": f.download_count or 0,
                    "created_at": f.created_at.isoformat() if f.created_at else None,
                }
                for f in files
            ]
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.get("/files/download/{file_id}")
async def download_event_file(
    file_id: str,
    user_id: str = Query(...),
    api_key: str = Depends(verify_api_key),
):
    """Download an event file."""
    from fastapi.responses import Response

    session = await get_db_session()
    try:
        result = await session.execute(
            select(EventFileDB).where(
                EventFileDB.id == uuid.UUID(file_id),
                EventFileDB.is_active == True
            )
        )
        file_obj = result.scalar_one_or_none()

        if not file_obj:
            raise HTTPException(status_code=404, detail="File not found")

        # Increment download count
        file_obj.download_count = (file_obj.download_count or 0) + 1
        await session.commit()

        return Response(
            content=file_obj.file_data,
            media_type=file_obj.mime_type,
            headers={
                "Content-Disposition": f'attachment; filename="{file_obj.original_filename}"',
                "Content-Length": str(file_obj.file_size),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.delete("/admin/files/{file_id}")
async def delete_event_file(
    file_id: str,
    user_id: str = Query(...),
    api_key: str = Depends(verify_api_key),
):
    """Delete an event file (soft delete). Admin only."""
    await verify_admin(user_id, api_key)

    session = await get_db_session()
    try:
        result = await session.execute(
            select(EventFileDB).where(EventFileDB.id == uuid.UUID(file_id))
        )
        file_obj = result.scalar_one_or_none()

        if not file_obj:
            raise HTTPException(status_code=404, detail="File not found")

        file_obj.is_active = False
        await session.commit()

        return {
            "status": "success",
            "message": f"File '{file_obj.original_filename}' deleted"
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


# --- VOICE NOTE ENDPOINTS ---

@app.post("/contact/{contact_id}/audio_note")
async def add_audio_note(
    contact_id: str,
    request: AudioNoteRequest,
    user_id: str = Query(...),
    api_key: str = Depends(verify_api_key),
):
    """Save an audio note (transcript + optional audio) to a contact."""
    session = await get_db_session()
    try:
        result = await session.execute(
            select(ContactDB).where(
                ContactDB.id == uuid.UUID(contact_id),
                ContactDB.user_id == uuid.UUID(user_id),
            )
        )
        contact = result.scalar_one_or_none()
        if not contact:
            raise HTTPException(status_code=404, detail="Contact not found")

        now = datetime.now(timezone.utc).strftime("%m/%d %H:%M")

        # Append to audio_notes JSON array
        audio_notes = contact.audio_notes or []
        audio_notes.append({
            "transcript": request.transcript or "",
            "audio_base64": request.audio_base64 or "",
            "timestamp": now,
        })
        contact.audio_notes = audio_notes

        # Also append transcript to text notes
        if request.transcript:
            prefix = f"\n[Voice Note {now}] "
            contact.notes = (contact.notes or "") + prefix + request.transcript

        await session.commit()

        # Update FAISS index
        try:
            summary = f"{contact.name}, {contact.email}, {contact.phone}, {contact.linkedin}, {contact.company_name}"
            if contact.notes:
                summary += f", {contact.notes[:200]}"
            faiss_index.update_contact(user_id, contact_id, summary)
        except Exception as faiss_err:
            print(f"[FAISS] Failed to update contact {contact_id} after audio note: {faiss_err}")
            traceback.print_exc()
            # Don't fail the request - audio note is still saved in database

        return {"status": "success", "message": "Audio note saved", "total_audio_notes": len(audio_notes)}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.post("/transcribe_audio/")
async def transcribe_audio(
    file: UploadFile = File(...),
    user_id: str = Query(...),
    api_key: str = Depends(verify_api_key),
):
    """Fallback: transcribe audio using Gemini when Web Speech API is unavailable."""
    if not gemini_configured:
        raise HTTPException(status_code=503, detail="Gemini not configured")

    try:
        audio_data = await file.read()
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        model = genai.GenerativeModel("gemini-2.5-flash")
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    model.generate_content,
                    [
                        "Transcribe this audio recording accurately. Return ONLY the transcribed text, nothing else.",
                        {"mime_type": file.content_type or "audio/webm", "data": audio_b64},
                    ]
                ),
                timeout=20
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Transcription timed out — Gemini API unresponsive")

        transcript = response.text.strip() if response.text else ""
        return {"status": "success", "transcript": transcript}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


# --- DIGITAL BUSINESS CARD ENDPOINTS ---

@app.get("/user/card/")
async def get_user_card(
    user_id: str = Query(..., description="User ID"),
    api_key: str = Depends(verify_api_key),
):
    """Get user's digital business card."""
    session = await get_db_session()
    try:
        result = await session.execute(
            select(UserCardDB).where(UserCardDB.user_id == uuid.UUID(user_id))
        )
        card = result.scalar_one_or_none()

        if not card:
            return {"status": "success", "card": {}, "shareable_url": None}

        shareable_url = None
        if card.shareable_token and card.is_active:
            # Generate full URL (use env var or request base URL in production)
            shareable_url = f"https://event-scout-delta.vercel.app/card/{card.shareable_token}"

        return {
            "status": "success",
            "card": card.card_data,
            "shareable_url": shareable_url,
            "is_active": card.is_active,
            "updated_at": card.updated_at.isoformat() if card.updated_at else None
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.put("/user/card/")
async def update_user_card(
    card_update: UserCardUpdate,
    user_id: str = Query(..., description="User ID"),
    api_key: str = Depends(verify_api_key),
):
    """Update user's digital business card."""
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
            select(UserCardDB).where(UserCardDB.user_id == uuid.UUID(user_id))
        )
        card = result.scalar_one_or_none()

        update_data = card_update.model_dump(exclude_unset=True)
        if not update_data:
            raise HTTPException(status_code=400, detail="No fields provided for update")

        if card:
            # Update existing card — MUST copy dict to trigger SQLAlchemy change detection
            existing = dict(card.card_data or {})
            for key, value in update_data.items():
                # Only store non-None and non-empty strings
                if value is not None and (not isinstance(value, str) or value.strip()):
                    existing[key] = value
            card.card_data = existing
            card.updated_at = datetime.now(timezone.utc)
        else:
            # Create new card - only store non-None and non-empty strings
            card = UserCardDB(
                user_id=uuid.UUID(user_id),
                card_data={k: v for k, v in update_data.items() if v is not None and (not isinstance(v, str) or v.strip())},
            )
            session.add(card)

        await session.commit()

        shareable_url = None
        if card.shareable_token and card.is_active:
            shareable_url = f"https://event-scout-delta.vercel.app/card/{card.shareable_token}"

        return {
            "status": "success",
            "card": card.card_data,
            "shareable_url": shareable_url
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.post("/user/card/generate_token/")
async def generate_card_token(
    user_id: str = Query(..., description="User ID"),
    api_key: str = Depends(verify_api_key),
):
    """Generate a shareable token for user's digital card. Creates QR code."""
    session = await get_db_session()
    try:
        result = await session.execute(
            select(UserCardDB).where(UserCardDB.user_id == uuid.UUID(user_id))
        )
        card = result.scalar_one_or_none()

        if not card:
            # Create card if it doesn't exist
            card = UserCardDB(
                user_id=uuid.UUID(user_id),
                card_data={},
                shareable_token=str(uuid.uuid4()),
                is_active=True
            )
            session.add(card)
        elif not card.shareable_token:
            # Generate token if it doesn't exist
            card.shareable_token = str(uuid.uuid4())
            card.is_active = True

        await session.commit()

        # Generate QR code
        shareable_url = f"https://event-scout-delta.vercel.app/card/{card.shareable_token}"

        qr = qrcode.QRCode(version=1, box_size=10, border=4)
        qr.add_data(shareable_url)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white")

        # Save QR code
        qr_path = os.path.join(QR_DIR, f"card_{user_id}.png")
        qr_img.save(qr_path)

        # Convert to base64 for frontend
        buffered = BytesIO()
        qr_img.save(buffered, format="PNG")
        qr_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {
            "status": "success",
            "shareable_url": shareable_url,
            "shareable_token": card.shareable_token,
            "qr_code_base64": qr_base64,
            "message": "Share this QR code or URL with people you meet!"
        }
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.get("/card/{token}")
async def view_card_public(token: str):
    """Public endpoint to view a digital business card (no auth required)."""
    session = await get_db_session()
    try:
        result = await session.execute(
            select(UserCardDB).where(UserCardDB.shareable_token == token)
        )
        card = result.scalar_one_or_none()

        if not card:
            raise HTTPException(status_code=404, detail="Card not found")

        if not card.is_active:
            raise HTTPException(status_code=403, detail="This card is no longer active")

        return {
            "status": "success",
            "card": card.card_data,
            "created_at": card.created_at.isoformat() if card.created_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.post("/contact/{contact_id}/accepted")
async def contact_accepted_webhook(
    contact_id: str,
    user_id: str = Query(..., description="User ID"),
    api_key: str = Depends(verify_api_key),
):
    """Triggered when user accepts a scanned card. Sends webhook to n8n for email automation."""
    session = await get_db_session()
    try:
        # Get contact details
        result = await session.execute(
            select(ContactDB).where(ContactDB.id == uuid.UUID(contact_id), ContactDB.user_id == uuid.UUID(user_id))
        )
        contact = result.scalar_one_or_none()

        if not contact:
            raise HTTPException(status_code=404, detail="Contact not found")

        # Get user details
        user_result = await session.execute(
            select(UserDB).where(UserDB.id == uuid.UUID(user_id))
        )
        user = user_result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Prepare webhook payload
        webhook_payload = {
            "contact_email": contact.email,
            "contact_name": contact.name,
            "contact_phone": contact.phone,
            "contact_company": contact.company_name,
            "contact_linkedin": contact.linkedin,
            "user_name": user.name,
            "user_email": user.email,
            "user_id": user_id,
            "contact_id": contact_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "card_accepted"
        }

        # Send to n8n webhook (if configured)
        if WEBHOOK_URL:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(WEBHOOK_URL, json=webhook_payload)
                    webhook_sent = response.status_code == 200
                    webhook_message = "Webhook sent successfully" if webhook_sent else f"Webhook failed: {response.status_code}"
            except Exception as webhook_err:
                print(f"[WEBHOOK ERROR] {webhook_err}")
                webhook_sent = False
                webhook_message = f"Webhook error: {str(webhook_err)}"
        else:
            webhook_sent = False
            webhook_message = "WEBHOOK_URL not configured"

        return {
            "status": "success",
            "message": "Contact acceptance recorded",
            "webhook_sent": webhook_sent,
            "webhook_message": webhook_message,
            "contact": {
                "id": str(contact.id),
                "name": contact.name,
                "email": contact.email
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


# --- ADMIN COMMAND CENTER ENDPOINTS ---

@app.get("/admin/search")
async def admin_search_contacts(
    q: str = Query(..., min_length=1, description="Search term"),
    admin_id: str = Depends(verify_admin),
    limit: int = Query(50, description="Max results"),
):
    """Search all contacts across all users by name, email, or company."""
    session = await get_db_session()
    try:
        search_term = f"%{q}%"
        query = (
            select(ContactDB)
            .where(
                or_(
                    ContactDB.name.ilike(search_term),
                    ContactDB.email.ilike(search_term),
                    ContactDB.company_name.ilike(search_term),
                    ContactDB.notes.ilike(search_term),
                )
            )
            .order_by(ContactDB.created_at.desc())
            .limit(limit)
        )
        result = await session.execute(query)
        contacts = result.scalars().all()

        # Get user names
        user_ids = list(set(str(c.user_id) for c in contacts))
        user_map = {}
        if user_ids:
            users_result = await session.execute(
                select(UserDB).where(UserDB.id.in_([uuid.UUID(uid) for uid in user_ids]))
            )
            for u in users_result.scalars().all():
                user_map[str(u.id)] = u.name

        return {
            "status": "success",
            "query": q,
            "total": len(contacts),
            "contacts": [
                {
                    "id": str(c.id),
                    "user_id": str(c.user_id),
                    "user_name": user_map.get(str(c.user_id), "Unknown"),
                    "name": c.name, "email": c.email or "N/A",
                    "phone": c.phone or "N/A", "company_name": c.company_name or "N/A",
                    "linkedin": c.linkedin or "N/A",
                    "lead_score": c.lead_score, "lead_temperature": c.lead_temperature,
                    "source": c.source or "manual",
                    "notes": c.notes or "",
                    "admin_notes": getattr(c, 'admin_notes', '') or "",
                    "created_at": c.created_at.isoformat() if c.created_at else "",
                }
                for c in contacts
            ],
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.post("/admin/broadcast")
async def admin_send_broadcast(
    admin_id: str = Depends(verify_admin),
    message: str = Query(..., min_length=1),
    priority: str = Query("normal"),
):
    """Send a broadcast message to the entire team."""
    session = await get_db_session()
    try:
        broadcast = AdminBroadcastDB(
            admin_id=uuid.UUID(admin_id),
            message=message,
            priority=priority if priority in ("normal", "urgent") else "normal",
        )
        session.add(broadcast)
        await session.commit()

        return {
            "status": "success",
            "broadcast_id": str(broadcast.id),
            "message": "Broadcast sent to team",
        }
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.get("/broadcasts/active")
async def get_active_broadcasts(
    user_id: str = Query(...),
    api_key: str = Depends(verify_api_key),
):
    """Get active broadcast messages for team members."""
    session = await get_db_session()
    try:
        result = await session.execute(
            select(AdminBroadcastDB, UserDB.name.label('admin_name'))
            .join(UserDB, AdminBroadcastDB.admin_id == UserDB.id)
            .where(AdminBroadcastDB.is_active == True)
            .order_by(AdminBroadcastDB.created_at.desc())
            .limit(10)
        )

        broadcasts = []
        for broadcast, admin_name in result:
            broadcasts.append({
                "id": str(broadcast.id),
                "message": broadcast.message,
                "priority": broadcast.priority,
                "admin_name": admin_name,
                "created_at": broadcast.created_at.isoformat() if broadcast.created_at else "",
            })

        return {"status": "success", "broadcasts": broadcasts}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.post("/broadcasts/{broadcast_id}/dismiss")
async def dismiss_broadcast(
    broadcast_id: str,
    admin_id: str = Depends(verify_admin),
):
    """Dismiss/deactivate a broadcast (admin only)."""
    session = await get_db_session()
    try:
        result = await session.execute(
            select(AdminBroadcastDB).where(AdminBroadcastDB.id == uuid.UUID(broadcast_id))
        )
        broadcast = result.scalar_one_or_none()
        if not broadcast:
            raise HTTPException(status_code=404, detail="Broadcast not found")

        broadcast.is_active = False
        await session.commit()

        return {"status": "success", "message": "Broadcast dismissed"}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.put("/admin/contact/{contact_id}/note")
async def admin_update_contact_note(
    contact_id: str,
    admin_id: str = Depends(verify_admin),
    note: str = Query("", description="Admin note/intel"),
):
    """Add or update admin notes on any contact."""
    session = await get_db_session()
    try:
        result = await session.execute(
            select(ContactDB).where(ContactDB.id == uuid.UUID(contact_id))
        )
        contact = result.scalar_one_or_none()
        if not contact:
            raise HTTPException(status_code=404, detail="Contact not found")

        contact.admin_notes = note
        await session.commit()

        return {
            "status": "success",
            "contact_id": contact_id,
            "admin_notes": note,
            "message": "Admin note updated",
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.get("/admin/export")
async def admin_export_all_contacts(
    admin_id: str = Depends(verify_admin),
    format: str = Query("csv"),
):
    """Export all contacts across all users as CSV."""
    from fastapi.responses import StreamingResponse
    import csv
    import io

    session = await get_db_session()
    try:
        result = await session.execute(
            select(ContactDB).order_by(ContactDB.created_at.desc())
        )
        contacts = result.scalars().all()

        # Get user names
        users_result = await session.execute(select(UserDB))
        user_map = {str(u.id): u.name for u in users_result.scalars().all()}

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Name", "Email", "Phone", "LinkedIn", "Company", "Notes", "Admin Notes",
            "Lead Score", "Lead Temperature", "Source", "Scanned By", "Created At"
        ])

        for c in contacts:
            writer.writerow([
                c.name or "", c.email or "", c.phone or "", c.linkedin or "",
                c.company_name or "", (c.notes or "").replace("\n", " "),
                (getattr(c, 'admin_notes', '') or "").replace("\n", " "),
                c.lead_score or "", c.lead_temperature or "",
                c.source or "manual",
                user_map.get(str(c.user_id), "Unknown"),
                c.created_at.isoformat() if c.created_at else "",
            ])

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=event_scout_all_contacts_{datetime.now().strftime('%Y%m%d')}.csv"}
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


# ==================== CONTACT FILE ENDPOINTS ====================

CONTACT_FILE_ALLOWED_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.ms-powerpoint": "ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "image/png": "png",
    "image/jpeg": "jpg",
}


@app.post("/admin/contact/{contact_id}/files")
async def upload_contact_file(
    contact_id: str,
    file: UploadFile = File(...),
    description: str = Query(""),
    category: str = Query("research"),
    user_id: str = Query(...),
    api_key: str = Depends(verify_api_key),
):
    """Upload a document to a specific contact. Admin only."""
    await verify_admin(user_id, api_key)

    if file.content_type not in CONTACT_FILE_ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"File type not allowed: {file.content_type}. Allowed: PDF, PPT, PPTX, DOC, DOCX, PNG, JPG")

    file_data = await file.read()
    if len(file_data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum 20MB.")

    original_filename = file.filename or "document"
    safe_filename = re.sub(r'[^\w\s\-\.]', '_', original_filename)

    session = await get_db_session()
    try:
        # Verify contact exists
        result = await session.execute(select(ContactDB).where(ContactDB.id == uuid.UUID(contact_id)))
        contact = result.scalar_one_or_none()
        if not contact:
            raise HTTPException(status_code=404, detail="Contact not found")

        db_file = ContactFileDB(
            contact_id=uuid.UUID(contact_id),
            filename=safe_filename,
            original_filename=original_filename,
            file_type=CONTACT_FILE_ALLOWED_TYPES[file.content_type],
            mime_type=file.content_type,
            file_size=len(file_data),
            file_data=file_data,
            description=description,
            category=category if category in ('research', 'pitch', 'brief', 'other') else 'research',
            uploaded_by=uuid.UUID(user_id),
        )
        session.add(db_file)
        await session.commit()

        return {
            "status": "success",
            "file_id": str(db_file.id),
            "filename": original_filename,
            "file_size": len(file_data),
            "contact_name": contact.name,
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.get("/contact/{contact_id}/files")
async def list_contact_files(
    contact_id: str,
    user_id: str = Query(...),
    api_key: str = Depends(verify_api_key),
):
    """List all documents attached to a contact. Any authenticated user."""
    session = await get_db_session()
    try:
        result = await session.execute(
            select(ContactFileDB)
            .where(ContactFileDB.contact_id == uuid.UUID(contact_id))
            .where(ContactFileDB.is_active == True)
            .order_by(ContactFileDB.created_at.desc())
        )
        files = result.scalars().all()

        return {
            "status": "success",
            "total": len(files),
            "files": [{
                "id": str(f.id),
                "filename": f.original_filename,
                "file_type": f.file_type,
                "file_size": f.file_size,
                "file_size_mb": round(f.file_size / (1024 * 1024), 2),
                "description": f.description or "",
                "category": f.category,
                "download_count": f.download_count,
                "created_at": f.created_at.isoformat() if f.created_at else "",
            } for f in files]
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.get("/contact/files/download/{file_id}")
async def download_contact_file(
    file_id: str,
    user_id: str = Query(...),
    api_key: str = Depends(verify_api_key),
):
    """Download a contact document. Any authenticated user."""
    from fastapi.responses import Response

    session = await get_db_session()
    try:
        result = await session.execute(
            select(ContactFileDB).where(
                ContactFileDB.id == uuid.UUID(file_id),
                ContactFileDB.is_active == True
            )
        )
        db_file = result.scalar_one_or_none()
        if not db_file:
            raise HTTPException(status_code=404, detail="File not found")

        # Increment download count
        db_file.download_count = (db_file.download_count or 0) + 1
        await session.commit()

        return Response(
            content=db_file.file_data,
            media_type=db_file.mime_type,
            headers={
                "Content-Disposition": f'attachment; filename="{db_file.original_filename}"',
                "Content-Length": str(db_file.file_size),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


@app.delete("/admin/contact/files/{file_id}")
async def delete_contact_file(
    file_id: str,
    user_id: str = Query(...),
    api_key: str = Depends(verify_api_key),
):
    """Delete a contact document. Admin only. Soft delete."""
    await verify_admin(user_id, api_key)

    session = await get_db_session()
    try:
        result = await session.execute(
            select(ContactFileDB).where(ContactFileDB.id == uuid.UUID(file_id))
        )
        db_file = result.scalar_one_or_none()
        if not db_file:
            raise HTTPException(status_code=404, detail="File not found")

        db_file.is_active = False
        await session.commit()
        return {"status": "success", "message": f"File '{db_file.original_filename}' deleted"}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await session.close()


# ==================== DATABASE BACKUP ENDPOINTS ====================

@app.post("/admin/backup")
async def admin_trigger_backup(
    admin_id: str = Depends(verify_admin),
):
    """Trigger a full database backup to the secondary Postgres instance."""
    if not ASYNC_BACKUP_URL:
        raise HTTPException(status_code=400, detail="Backup database not configured. Set BACKUP_DATABASE_URL env var.")

    result = await run_backup()
    return result


@app.get("/admin/backup/status")
async def admin_backup_status(
    admin_id: str = Depends(verify_admin),
):
    """Get the last backup status and row count comparison between primary and backup."""
    if not ASYNC_BACKUP_URL:
        return {"backup_configured": False, "message": "Set BACKUP_DATABASE_URL env var to enable backups"}

    # Count rows in primary
    primary_session = await get_db_session()
    primary_counts = {}
    try:
        for table_name, model in [
            ("users", UserDB), ("contacts", ContactDB), ("exhibitors", ExhibitorDB),
            ("conversations", ConversationDB), ("user_cards", UserCardDB),
            ("event_files", EventFileDB), ("contact_pipelines", ContactPipelineDB), ("admin_broadcasts", AdminBroadcastDB),
        ]:
            result = await primary_session.execute(select(func.count(model.id)))
            primary_counts[table_name] = result.scalar() or 0
    finally:
        await primary_session.close()

    # Count rows in backup
    backup_factory = get_backup_session_factory()
    backup_counts = {}
    if backup_factory:
        backup_session = backup_factory()
        try:
            for table_name, model in [
                ("users", UserDB), ("contacts", ContactDB), ("exhibitors", ExhibitorDB),
                ("conversations", ConversationDB), ("user_cards", UserCardDB),
                ("event_files", EventFileDB), ("contact_pipelines", ContactPipelineDB), ("admin_broadcasts", AdminBroadcastDB),
            ]:
                result = await backup_session.execute(select(func.count(model.id)))
                backup_counts[table_name] = result.scalar() or 0
        except Exception as e:
            backup_counts = {"error": str(e)}
        finally:
            await backup_session.close()

    return {
        "backup_configured": True,
        "last_backup": _last_backup_result,
        "primary_counts": primary_counts,
        "backup_counts": backup_counts,
        "in_sync": primary_counts == backup_counts if isinstance(backup_counts, dict) and "error" not in backup_counts else False
    }


# ==================== PIPELINE ENDPOINTS ====================

@app.get("/contact/{contact_id}/pipeline")
async def get_pipeline_status(
    contact_id: str,
    user_id: str = Query(...),
    api_key: str = Depends(verify_api_key),
):
    """Get pipeline status and generated content for a contact."""
    session = await get_db_session()
    try:
        result = await session.execute(
            select(ContactPipelineDB)
            .where(ContactPipelineDB.contact_id == uuid.UUID(contact_id))
            .order_by(ContactPipelineDB.created_at.desc())
        )
        pipeline = result.scalar_one_or_none()
        if not pipeline:
            return {"status": "none", "message": "No pipeline run for this contact"}

        return {
            "id": str(pipeline.id),
            "status": pipeline.status,
            "current_step": pipeline.current_step,
            "error_message": pipeline.error_message,
            "research_summary": pipeline.research_summary,
            "research_data": pipeline.research_data,
            "score_completed": pipeline.score_completed,
            "pitch_angle": pipeline.pitch_angle,
            "pitch_email_subject": pipeline.pitch_email_subject,
            "pitch_email_body": pipeline.pitch_email_body,
            "pitch_slides_content": pipeline.pitch_slides_content,
            "deck_file_id": str(pipeline.deck_file_id) if pipeline.deck_file_id else None,
            "presenton_presentation_id": pipeline.presenton_presentation_id,
            "started_at": pipeline.started_at.isoformat() if pipeline.started_at else None,
            "completed_at": pipeline.completed_at.isoformat() if pipeline.completed_at else None,
        }
    finally:
        await session.close()


@app.post("/contact/{contact_id}/pipeline/run")
async def trigger_pipeline(
    contact_id: str,
    user_id: str = Query(...),
    api_key: str = Depends(verify_api_key),
):
    """Manually trigger the intelligence pipeline for a contact."""
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=400, detail="OpenRouter API key not configured. Pipeline requires AI access.")

    # Verify contact exists and belongs to user (or user is admin)
    session = await get_db_session()
    try:
        result = await session.execute(
            select(ContactDB).where(ContactDB.id == uuid.UUID(contact_id))
        )
        contact = result.scalar_one_or_none()
        if not contact:
            raise HTTPException(status_code=404, detail="Contact not found")

        # Check if pipeline already running
        existing = await session.execute(
            select(ContactPipelineDB)
            .where(
                ContactPipelineDB.contact_id == uuid.UUID(contact_id),
                ContactPipelineDB.status.in_(["pending", "researching", "scoring", "pitching", "generating_deck", "attaching"])
            )
        )
        if existing.scalar_one_or_none():
            return {"status": "already_running", "message": "Pipeline is already running for this contact"}
    finally:
        await session.close()

    # Fire pipeline in background
    asyncio.create_task(run_contact_pipeline(contact_id, user_id))
    return {"status": "started", "message": f"Pipeline started for {contact.name}"}


@app.post("/contact/{contact_id}/pipeline/retry")
async def retry_pipeline(
    contact_id: str,
    user_id: str = Query(...),
    api_key: str = Depends(verify_api_key),
):
    """Retry a failed pipeline for a contact."""
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=400, detail="OpenRouter API key not configured")

    session = await get_db_session()
    try:
        # Delete old failed pipeline
        await session.execute(
            delete(ContactPipelineDB).where(
                ContactPipelineDB.contact_id == uuid.UUID(contact_id),
                ContactPipelineDB.status == "failed"
            )
        )
        await session.commit()
    finally:
        await session.close()

    asyncio.create_task(run_contact_pipeline(contact_id, user_id))
    return {"status": "retrying", "message": "Pipeline restarted"}


@app.get("/admin/pipelines")
async def admin_list_pipelines(
    admin_id: str = Depends(verify_admin),
    status_filter: str = Query(None),
    limit: int = Query(50),
):
    """List all pipelines across all users (admin only)."""
    session = await get_db_session()
    try:
        query = select(ContactPipelineDB).order_by(ContactPipelineDB.created_at.desc()).limit(limit)
        if status_filter:
            query = query.where(ContactPipelineDB.status == status_filter)

        result = await session.execute(query)
        pipelines = result.scalars().all()

        # Get contact names for display
        contact_ids = [p.contact_id for p in pipelines]
        contacts_result = await session.execute(
            select(ContactDB).where(ContactDB.id.in_(contact_ids))
        )
        contacts_map = {str(c.id): c.name for c in contacts_result.scalars().all()}

        # Get user names
        user_ids = list(set(p.user_id for p in pipelines))
        users_result = await session.execute(
            select(UserDB).where(UserDB.id.in_(user_ids))
        )
        users_map = {str(u.id): u.name for u in users_result.scalars().all()}

        # Count by status
        count_result = await session.execute(
            select(ContactPipelineDB.status, func.count(ContactPipelineDB.id))
            .group_by(ContactPipelineDB.status)
        )
        status_counts = dict(count_result.all())

        return {
            "pipelines": [
                {
                    "id": str(p.id),
                    "contact_id": str(p.contact_id),
                    "contact_name": contacts_map.get(str(p.contact_id), "Unknown"),
                    "user_id": str(p.user_id),
                    "user_name": users_map.get(str(p.user_id), "Unknown"),
                    "status": p.status,
                    "current_step": p.current_step,
                    "error_message": p.error_message,
                    "started_at": p.started_at.isoformat() if p.started_at else None,
                    "completed_at": p.completed_at.isoformat() if p.completed_at else None,
                }
                for p in pipelines
            ],
            "status_counts": status_counts,
            "auto_pipeline_enabled": AUTO_PIPELINE_ENABLED,
        }
    finally:
        await session.close()


@app.post("/admin/pipeline/settings")
async def admin_pipeline_settings(
    admin_id: str = Depends(verify_admin),
    auto_enabled: bool = Query(None),
):
    """Toggle auto-pipeline settings (admin only)."""
    global AUTO_PIPELINE_ENABLED
    result = {}
    if auto_enabled is not None:
        AUTO_PIPELINE_ENABLED = auto_enabled
        result["auto_pipeline_enabled"] = AUTO_PIPELINE_ENABLED
        print(f"[PIPELINE] Auto-pipeline {'enabled' if auto_enabled else 'disabled'} by admin")

    result["current_settings"] = {
        "auto_pipeline_enabled": AUTO_PIPELINE_ENABLED,
        "presenton_configured": bool(PRESENTON_API_URL),
        "openrouter_configured": bool(OPENROUTER_API_KEY),
    }
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
