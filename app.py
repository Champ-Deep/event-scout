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
    UserDB, ContactDB, SharedContactDB, UserProfileDB, ConversationDB,
    get_engine, get_session_factory, init_db, ASYNC_DATABASE_URL
)

# --- CONFIG ---
APP_API_KEY = os.environ.get("APP_API_KEY", "1234")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")  # n8n webhook endpoint

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


# --- AUTH ---
def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key


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


# --- GEMINI CONVERSATIONAL ENGINE ---
class GeminiConversationEngine:
    def __init__(self):
        self.model_name = "gemini-2.5-flash"

    def _build_system_prompt(self, user_profile: dict = None) -> str:
        base_prompt = """You are an intelligent Event Scout Assistant — an AI-powered partner for professionals at trade shows, conferences, and networking events. Your role is to help users find contacts, prioritize leads, suggest pitch angles, and provide strategic advice.

When answering questions:
1. Use the provided contact information from the database to answer queries
2. Be helpful, strategic, and concise
3. If no relevant contacts are found, politely inform the user
4. If the query is ambiguous, ask clarifying questions
5. Format contact details clearly when presenting them
6. You can help with: finding contacts, prioritizing leads, suggesting pitch angles, recommending who to follow up with, summarizing contacts by industry/role, and answering questions about people the user has met
7. When suggesting actions, be specific — reference actual contact data and user context
8. If contacts have lead scores, use them to prioritize recommendations

Important: Only use information from the provided contact context. Do not make up contact details."""

        if user_profile and any(user_profile.values()):
            profile_context = "\n\nUSER CONTEXT (use this to personalize your advice):"
            if user_profile.get("full_name"):
                profile_context += f"\n- User: {user_profile['full_name']}"
            if user_profile.get("job_title"):
                profile_context += f"\n- Role: {user_profile['job_title']}"
            if user_profile.get("company_name"):
                profile_context += f"\n- Company: {user_profile['company_name']}"
            if user_profile.get("products"):
                products_str = ", ".join(p.get("name", "") for p in user_profile["products"] if p.get("name"))
                if products_str:
                    profile_context += f"\n- Products/Services: {products_str}"
            if user_profile.get("target_industries"):
                profile_context += f"\n- Target Industries: {', '.join(user_profile['target_industries'])}"
            if user_profile.get("target_roles"):
                profile_context += f"\n- Target Roles: {', '.join(user_profile['target_roles'])}"
            if user_profile.get("value_propositions"):
                profile_context += f"\n- Value Propositions: {'; '.join(user_profile['value_propositions'])}"
            if user_profile.get("pitch_style"):
                profile_context += f"\n- Pitch Style: {user_profile['pitch_style']}"
            if user_profile.get("current_event_name"):
                profile_context += f"\n- Current Event: {user_profile['current_event_name']}"
            if user_profile.get("current_event_description"):
                profile_context += f"\n- Event Details: {user_profile['current_event_description']}"
            if user_profile.get("event_goals"):
                profile_context += f"\n- Event Goals: {', '.join(user_profile['event_goals'])}"

            profile_context += "\n\nUse this context to tailor your responses. When suggesting who to prioritize, consider alignment with the user's products, target market, and event goals. When suggesting pitch angles, reference the user's value propositions and pitch style."
            base_prompt += profile_context

        return base_prompt

    def _build_context_from_contacts(self, contacts: List[tuple]) -> str:
        if not contacts:
            return "No contacts found in the database matching your query."

        context_parts = ["Here are the relevant contacts from your database:\n"]
        for i, (text, meta) in enumerate(contacts, 1):
            contact_info = f"""
Contact {i}:
- Name: {meta.get('name', 'N/A')}
- Email: {meta.get('email', 'N/A')}
- Phone: {meta.get('phone', 'N/A')}
- LinkedIn: {meta.get('linkedin', 'N/A')}
- Company: {meta.get('company_name', 'N/A')}"""
            if meta.get('notes'):
                contact_info += f"\n- Notes: {meta['notes']}"
            if meta.get('lead_score') is not None:
                contact_info += f"\n- Lead Score: {meta['lead_score']}/100 ({meta.get('lead_temperature', 'unscored')})"
            if meta.get('lead_score_reasoning'):
                contact_info += f"\n- Score Reasoning: {meta['lead_score_reasoning']}"
            context_parts.append(contact_info)
        return "\n".join(context_parts)

    def _build_conversation_history(self, history: Optional[List[Dict[str, str]]]) -> List[Dict]:
        if not history:
            return []
        gemini_history = []
        for msg in history:
            role = "user" if msg.get("role") == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg.get("content", "")]})
        return gemini_history

    def _generate_fallback_response(self, query: str, retrieved_contacts: List[tuple]) -> str:
        if not retrieved_contacts:
            return f"No contacts found matching your query: '{query}'. Try adding some contacts first."
        response_parts = [f"Found {len(retrieved_contacts)} contact(s):\n"]
        for i, (text, meta) in enumerate(retrieved_contacts, 1):
            response_parts.append(f"""
**{i}. {meta.get('name', 'N/A')}**
- Email: {meta.get('email', 'N/A')}
- Phone: {meta.get('phone', 'N/A')}
- LinkedIn: {meta.get('linkedin', 'N/A')}
- Company: {meta.get('company_name', 'N/A')}
""")
        return "\n".join(response_parts)

    def generate_response(self, query, retrieved_contacts, conversation_history=None, user_profile=None):
        if not GEMINI_API_KEY or not gemini_configured:
            return self._generate_fallback_response(query, retrieved_contacts)
        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self._build_system_prompt(user_profile)
            )
            contact_context = self._build_context_from_contacts(retrieved_contacts)
            full_prompt = f"""Based on the following contact information from the database:

{contact_context}

User Query: {query}

Please provide a helpful, conversational response to the user's query using the contact information above."""

            if conversation_history:
                chat = model.start_chat(history=self._build_conversation_history(conversation_history))
                response = chat.send_message(full_prompt)
            else:
                response = model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            print(f"[GEMINI] API error: {e}")
            traceback.print_exc()
            return self._generate_fallback_response(query, retrieved_contacts)


gemini_engine = GeminiConversationEngine()


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

    import httpx

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
            import asyncio
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
    print("[STARTUP] Contact Assistant API Starting...")
    print(f"[STARTUP] Database URL configured: {bool(ASYNC_DATABASE_URL)}")
    print(f"[STARTUP] Gemini configured: {gemini_configured}")
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
    return {"message": "Contact Assistant API - Multi-User (PostgreSQL)", "version": "3.0.0"}


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
            "webhook_configured": bool(WEBHOOK_URL),
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
        db_user = UserDB(
            id=user_id,
            name=user.name,
            email=user.email,
            password_hash=hash_password(user.password),
        )
        session.add(db_user)
        await session.commit()

        # Initialize empty FAISS index for user
        faiss_index.build_for_user(str(user_id), [])

        print(f"[USER] Created new user: {user.email} with ID: {user_id}")
        return {
            "status": "success",
            "message": "User registered successfully",
            "user_id": str(user_id),
            "name": user.name,
            "email": user.email,
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

        return {
            "status": "success",
            "message": "Login successful",
            "user_id": str(user.id),
            "name": user.name,
            "email": user.email,
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

        # Load user profile
        session = await get_db_session()
        try:
            result = await session.execute(
                select(UserProfileDB).where(UserProfileDB.user_id == uuid.UUID(query.user_id))
            )
            profile_row = result.scalar_one_or_none()
            user_profile = profile_row.profile_data if profile_row else {}
        finally:
            await session.close()

        response_text = gemini_engine.generate_response(
            query=query.query,
            retrieved_contacts=retrieved_contacts,
            conversation_history=query.conversation_history,
            user_profile=user_profile,
        )

        return {
            "status": "success",
            "response": response_text,
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
