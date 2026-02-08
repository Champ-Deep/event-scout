import os
import re
import json
import uuid
import qrcode
from pyzbar.pyzbar import decode as decode_qr
from PIL import Image
import base64
import traceback
import numpy as np
import faiss
from io import BytesIO
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

from sentence_transformers import SentenceTransformer

import bcrypt

import google.generativeai as genai

from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from database import (
    UserDB, ContactDB, SharedContactDB, UserProfileDB, ConversationDB, ExhibitorDB, UserCardDB,
    get_engine, get_session_factory, init_db, ASYNC_DATABASE_URL
)

import httpx

# --- CONFIG ---
APP_API_KEY = os.environ.get("APP_API_KEY", "1234")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")  # n8n webhook endpoint
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
ADMIN_EMAILS = [e.strip().lower() for e in os.environ.get("ADMIN_EMAILS", "deep@lakeb2b.com").split(",") if e.strip()]

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
        raise HTTPException(status_code=503, detail="Database not available")
    async with factory() as session:
        return session


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

            profile_context += "\n\nCRITICAL: Use ALL of the above context when giving advice. Every pitch suggestion must reference the user's actual products and value props. Every prioritization must consider their target market. Every briefing must connect to their event goals."
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
                    ex_line += f" | {ex['description'][:100]}"
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

    model_names = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite']
    img = Image.open(image_path)
    print(f"[GEMINI] Image opened: {img.size}, mode={img.mode}")

    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != 'RGB':
        img = img.convert('RGB')

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
    for model_name in model_names:
        try:
            print(f"[GEMINI] Trying model: {model_name}")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([prompt, img])
            if not response or not response.text:
                print(f"[GEMINI] Empty response from {model_name}")
                continue

            response_text = response.text.strip()
            print(f"[GEMINI] Raw response from {model_name}: {response_text[:500]}")

            if response_text.startswith("```"):
                parts = response_text.split("```")
                if len(parts) >= 3:
                    response_text = parts[1]
                else:
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

            print(f"[GEMINI] Extracted: {fields}")
            return fields

        except json.JSONDecodeError as e:
            print(f"[GEMINI] JSON parse error with {model_name}: {e}")
            last_error = e
            continue
        except Exception as e:
            print(f"[GEMINI] Error with {model_name}: {e}")
            traceback.print_exc()
            last_error = e
            continue

    print(f"[GEMINI] All models failed. Last error: {last_error}")
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
        response = model.generate_content(prompt)
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


# --- CONTACT LOGIC (now using Postgres) ---
async def add_contact_logic(contact: Contact, user_id: str, source: str = "manual") -> dict:
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
    with open(temp_filename, "wb") as f:
        f.write(content)
    try:
        fields = extract_contact_from_image_with_gemini(temp_filename)
        has_info = any(v != "N/A" for k, v in fields.items() if k != "linkedin")
        if not has_info:
            raise HTTPException(status_code=400, detail="No contact information found in image")

        # LinkedIn auto-lookup if not found on card
        if fields.get("linkedin", "N/A") == "N/A":
            linkedin_url = lookup_linkedin_with_gemini(fields["name"], fields.get("company_name", "N/A"))
            if linkedin_url:
                fields["linkedin"] = linkedin_url
                fields["linkedin_source"] = "ai_detected"
                print(f"[LINKEDIN] Auto-detected: {linkedin_url}")

        contact_obj = Contact(**{k: v for k, v in fields.items() if k in Contact.model_fields})
        result = await add_contact_logic(contact_obj, user_id, source="scan")

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

        return {
            "status": "success",
            "message": "Contact added from image",
            "extracted_fields": fields,
            "contact_id": result["contact_id"],
            "qr_base64": result["qr_base64"]
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
            response = model.generate_content(prompt)
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


# --- FASTAPI APP ---
app = FastAPI(title="Contact Assistant API - Multi-User (PostgreSQL)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def startup_event():
    print("=" * 50)
    print("[STARTUP] Event Scout Intelligence API v3.1 Starting...")
    print(f"[STARTUP] Database URL configured: {bool(ASYNC_DATABASE_URL)}")
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

    print("[STARTUP] Ready!")


@app.get("/")
async def root():
    return {"message": "Event Scout Intelligence API", "version": "3.1.0"}


@app.get("/health/")
async def health_check():
    session = await get_db_session()
    try:
        result = await session.execute(select(func.count(UserDB.id)))
        total_users = result.scalar() or 0
        return {
            "status": "healthy",
            "database": "postgresql",
            "total_users": total_users,
            "gemini_configured": gemini_configured,
            "openrouter_configured": bool(OPENROUTER_API_KEY),
            "webhook_configured": bool(WEBHOOK_URL),
            "version": "3.1.0",
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
    return await add_contact_from_image(file, user_id)


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
        retrieved_contacts = faiss_index.search(query.user_id, query.query, k=query.top_k or 4)

        session = await get_db_session()
        try:
            # Load user profile
            result = await session.execute(
                select(UserProfileDB).where(UserProfileDB.user_id == uuid.UUID(query.user_id))
            )
            profile_row = result.scalar_one_or_none()
            user_profile = profile_row.profile_data if profile_row else {}

            # Load all contacts summary for broader context
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

            # Load exhibitors if query mentions event/exhibitor/booth/visit keywords
            exhibitors = []
            query_lower = query.query.lower()
            exhibitor_keywords = ["exhibitor", "booth", "visit", "hall", "expo", "vendor", "stand", "floor"]
            if any(kw in query_lower for kw in exhibitor_keywords):
                ex_result = await session.execute(select(ExhibitorDB).limit(50))
                ex_rows = ex_result.scalars().all()
                exhibitors = [
                    {
                        "name": e.name, "booth": e.booth or "", "hall": e.hall or "",
                        "category": e.category or "", "country": e.country or "",
                        "description": e.description or "",
                    }
                    for e in ex_rows
                ]
        finally:
            await session.close()

        # Get model info for response metadata
        model_key = user_profile.get("preferred_ai_model", "claude-opus")
        model_info = AVAILABLE_MODELS.get(model_key, AVAILABLE_MODELS["claude-opus"])

        response_text = await intelligence_engine.generate_response(
            query=query.query,
            retrieved_contacts=retrieved_contacts,
            conversation_history=query.conversation_history,
            user_profile=user_profile,
            all_contacts=all_contacts,
            exhibitors=exhibitors,
        )

        return {
            "status": "success",
            "response": response_text,
            "model": model_info["name"],
            "retrieved_contacts": [
                {
                    "name": meta.get("name", "N/A"),
                    "email": meta.get("email", "N/A"),
                    "phone": meta.get("phone", "N/A"),
                    "linkedin": meta.get("linkedin", "N/A"),
                    "company_name": meta.get("company_name", "N/A"),
                }
                for (text, meta) in retrieved_contacts
            ],
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
        score_result = score_contact_with_gemini(contact_meta, user_profile)

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
                score_result = score_contact_with_gemini(contact_meta, user_profile)
                c.lead_score = score_result["score"]
                c.lead_temperature = score_result["temperature"]
                c.lead_score_reasoning = score_result["reasoning"]
                c.lead_score_breakdown = score_result.get("breakdown", {})
                c.lead_recommended_actions = score_result.get("recommended_actions", [])
                scored += 1
            except Exception as e:
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
async def admin_list_users(admin_id: str = Depends(verify_admin)):
    """List all users with their contact counts."""
    session = await get_db_session()
    try:
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
    except Exception as e:
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
                    "lead_score": c.lead_score, "lead_temperature": c.lead_temperature,
                    "source": c.source or "manual",
                    "notes": (c.notes or "")[:100],
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


@app.post("/admin/webhook/test")
async def test_webhook_connection(
    admin_id: str = Depends(verify_admin),
):
    """Test n8n webhook connectivity from admin dashboard."""
    if not WEBHOOK_URL:
        return {
            "status": "error",
            "message": "WEBHOOK_URL not configured. Set the WEBHOOK_URL environment variable."
        }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                WEBHOOK_URL,
                json={
                    "test": True,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "message": "Test webhook from Event Scout admin dashboard",
                    "source": "admin_test"
                }
            )

            success = response.status_code == 200
            return {
                "status": "success" if success else "error",
                "status_code": response.status_code,
                "message": "Webhook reachable" if success else f"Webhook returned HTTP {response.status_code}",
                "webhook_url": WEBHOOK_URL
            }
    except httpx.TimeoutException:
        return {
            "status": "error",
            "message": "Webhook timeout after 10 seconds",
            "webhook_url": WEBHOOK_URL
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Webhook error: {str(e)}",
            "webhook_url": WEBHOOK_URL
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
        for u in users:
            cnt = (await session.execute(
                select(func.count(ContactDB.id)).where(ContactDB.user_id == u.id)
            )).scalar() or 0
            per_user.append({
                "user_id": str(u.id), "name": u.name, "email": u.email,
                "contact_count": cnt, "is_admin": u.is_admin,
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
        response = model.generate_content([
            "Transcribe this audio recording accurately. Return ONLY the transcribed text, nothing else.",
            {"mime_type": file.content_type or "audio/webm", "data": audio_b64},
        ])

        transcript = response.text.strip() if response.text else ""
        return {"status": "success", "transcript": transcript}
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
            # Update existing card
            existing = card.card_data or {}
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
