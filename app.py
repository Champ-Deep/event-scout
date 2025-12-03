import os
import re
import json
import pickle
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

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer

import google.generativeai as genai

# --- CONFIG ---
APP_API_KEY = os.environ.get("APP_API_KEY", "1234")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDezZZCXGTjHFk1-y7WfzPFwEmTeFDZ95A")  # Set your Gemini API key

BASE_DIR = os.getcwd()
TEMP_DIR = os.path.join(BASE_DIR, "temp_images")
QR_DIR = os.path.join(BASE_DIR, "saved_qr")
DATA_DIR = os.path.join(BASE_DIR, "data")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.pickle")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")

def initialize_directories_and_files():
    """Initialize all required directories and files on startup."""
    # Create directories
    for directory in [TEMP_DIR, QR_DIR, DATA_DIR]:
        os.makedirs(directory, exist_ok=True)
        print(f"[INIT] Directory ensured: {directory}")

    # Initialize empty metadata file if it doesn't exist
    if not os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "wb") as f:
            pickle.dump({"doc_texts": [], "doc_metadata": []}, f)
        print(f"[INIT] Created empty metadata file: {METADATA_PATH}")
    else:
        print(f"[INIT] Metadata file exists: {METADATA_PATH}")

    # Log FAISS index status
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"[INIT] FAISS index exists: {FAISS_INDEX_PATH}")
    else:
        print(f"[INIT] FAISS index will be created when first contact is added")

# Run initialization on module load
initialize_directories_and_files()

# --- Configure Gemini ---
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- Pydantic models ---
class Contact(BaseModel):
    name: str
    email: str
    phone: str
    linkedin: str
    company_name: str = "N/A"

class SearchQuery(BaseModel):
    query: str

class ConverseQuery(BaseModel):
    query: str
    conversation_history: Optional[List[Dict[str, str]]] = None  # For multi-turn conversations
    top_k: Optional[int] = 4  # Number of contacts to retrieve for context

class ContactUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    company_name: Optional[str] = None

# --- AUTH ---
def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

# --- EMBEDDING MODEL ---
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # small but powerful

def get_embedding(text: str) -> np.ndarray:
    return embedding_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]

# --- FAISS MANAGER ---
class FAISSManager:
    def __init__(self):
        self.index: faiss.IndexFlatIP = None
        self.doc_texts: List[str] = []
        self.doc_metadata: List[dict] = []
        self._ensure_data_files()
        self._load_metadata()
        self._load_index()
        if self.index is None:
            self.rebuild_index()
        print(f"[FAISS] Initialized with {len(self.doc_texts)} contacts")

    def _ensure_data_files(self):
        """Ensure data directory and metadata file exist."""
        os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "wb") as f:
                pickle.dump({"doc_texts": [], "doc_metadata": []}, f)
            print(f"[FAISS] Created new metadata file")

    def _load_metadata(self):
        try:
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, "rb") as f:
                    data = pickle.load(f)
                    self.doc_texts = data.get("doc_texts", [])
                    self.doc_metadata = data.get("doc_metadata", [])
        except Exception as e:
            print(f"[FAISS] Error loading metadata, starting fresh: {e}")
            self.doc_texts = []
            self.doc_metadata = []
            self._save_metadata()

    def _save_metadata(self):
        with open(METADATA_PATH, "wb") as f:
            pickle.dump({"doc_texts": self.doc_texts, "doc_metadata": self.doc_metadata}, f)

    def _load_index(self):
        if os.path.exists(FAISS_INDEX_PATH):
            try:
                self.index = faiss.read_index(FAISS_INDEX_PATH)
            except:
                self.index = None

    def rebuild_index(self):
        if not self.doc_texts:
            self.index = None
            return
        vecs = np.array([get_embedding(t) for t in self.doc_texts]).astype("float32")
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        faiss.write_index(self.index, FAISS_INDEX_PATH)

    def add_document(self, text: str, metadata: dict):
        self.doc_texts.append(text)
        self.doc_metadata.append(metadata)
        self._save_metadata()
        self.rebuild_index()

    def search(self, query: str, k: int = 4):
        if self.index is None or self.index.ntotal == 0:
            return []
        q_vec = get_embedding(query).astype("float32").reshape(1, -1)
        D, I = self.index.search(q_vec, min(k, len(self.doc_texts)))
        results = []
        for idx in I[0]:
            if idx < len(self.doc_texts):
                results.append((self.doc_texts[idx], self.doc_metadata[idx]))
        return results

    def find_index_by_id(self, contact_id: str) -> int:
        """Find the index of a document by contact ID. Returns -1 if not found."""
        for idx, meta in enumerate(self.doc_metadata):
            if meta.get("id") == contact_id:
                return idx
        return -1

    def delete_document(self, contact_id: str) -> bool:
        """Delete a document by contact ID. Returns True if deleted, False if not found."""
        idx = self.find_index_by_id(contact_id)
        if idx == -1:
            return False

        # Remove from lists
        self.doc_texts.pop(idx)
        self.doc_metadata.pop(idx)

        # Save metadata and rebuild index
        self._save_metadata()
        self.rebuild_index()

        # Also save empty index if no documents left
        if not self.doc_texts:
            if os.path.exists(FAISS_INDEX_PATH):
                os.remove(FAISS_INDEX_PATH)

        return True

    def update_document(self, contact_id: str, new_text: str, new_metadata: dict) -> bool:
        """Update a document by contact ID. Returns True if updated, False if not found."""
        idx = self.find_index_by_id(contact_id)
        if idx == -1:
            return False

        # Update the document
        self.doc_texts[idx] = new_text
        self.doc_metadata[idx] = new_metadata

        # Save metadata and rebuild index
        self._save_metadata()
        self.rebuild_index()

        return True

faiss_manager = FAISSManager()

# --- GEMINI CONVERSATIONAL ENGINE ---
class GeminiConversationEngine:
    def __init__(self):
        self.model_name = "gemini-3-pro-preview"
        
    def _build_system_prompt(self) -> str:
        return """You are an intelligent Contact Assistant. Your role is to help users find and manage their contacts in a friendly, conversational manner.

When answering questions:
1. Use the provided contact information from the database to answer queries
2. Be helpful and concise
3. If no relevant contacts are found, politely inform the user
4. If the query is ambiguous, ask clarifying questions
5. Format contact details clearly when presenting them
6. You can help with: finding contacts, suggesting who to reach out to, summarizing contact info, etc.

Important: Only use information from the provided contact context. Do not make up contact details."""

    def _build_context_from_contacts(self, contacts: List[tuple]) -> str:
        if not contacts:
            return "No contacts found in the database matching your query."
        
        context_parts = ["Here are the relevant contacts from your database:\n"]
        for i, (text, meta) in enumerate(contacts, 1):
            context_parts.append(f"""
Contact {i}:
- Name: {meta.get('name', 'N/A')}
- Email: {meta.get('email', 'N/A')}
- Phone: {meta.get('phone', 'N/A')}
- LinkedIn: {meta.get('linkedin', 'N/A')}
- Company: {meta.get('company_name', 'N/A')}
""")
        return "\n".join(context_parts)

    def _build_conversation_history(self, history: Optional[List[Dict[str, str]]]) -> List[Dict]:
        """Convert conversation history to Gemini format."""
        if not history:
            return []
        
        gemini_history = []
        for msg in history:
            role = "user" if msg.get("role") == "user" else "model"
            gemini_history.append({
                "role": role,
                "parts": [msg.get("content", "")]
            })
        return gemini_history

    def generate_response(
        self, 
        query: str, 
        retrieved_contacts: List[tuple],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate a conversational response using Gemini."""
        
        if not GEMINI_API_KEY:
            raise HTTPException(
                status_code=500, 
                detail="Gemini API key not configured. Set GEMINI_API_KEY environment variable."
            )
        
        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self._build_system_prompt()
            )
            
            # Build the context from retrieved contacts
            contact_context = self._build_context_from_contacts(retrieved_contacts)
            
            # Build the full prompt
            full_prompt = f"""Based on the following contact information from the database:

{contact_context}

User Query: {query}

Please provide a helpful, conversational response to the user's query using the contact information above."""

            # Handle conversation history for multi-turn
            if conversation_history:
                chat = model.start_chat(
                    history=self._build_conversation_history(conversation_history)
                )
                response = chat.send_message(full_prompt)
            else:
                response = model.generate_content(full_prompt)
            
            return response.text
            
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

gemini_engine = GeminiConversationEngine()

# --- QR GENERATOR ---
def create_qr(contact: Contact, contact_id: str = None):
    """Generate QR code for a contact. If contact_id provided, use it; otherwise generate new one."""
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
    """Retrieve QR code as base64 for a given contact_id."""
    qr_path = os.path.join(QR_DIR, f"qr_{contact_id}.png")
    if os.path.exists(qr_path):
        with open(qr_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None

# --- QR SCANNER ---
def parse_vcard(vcard_text: str) -> dict:
    """Parse vCard format and extract contact fields."""
    fields = {"name": "N/A", "email": "N/A", "phone": "N/A", "linkedin": "N/A", "company_name": "N/A"}

    # Extract FN (Full Name)
    fn_match = re.search(r"FN[;:]([^\r\n]+)", vcard_text)
    if fn_match:
        fields["name"] = fn_match.group(1).strip()
    else:
        # Try N field
        n_match = re.search(r"^N[;:]([^\r\n]+)", vcard_text, re.MULTILINE)
        if n_match:
            fields["name"] = n_match.group(1).strip()

    # Extract Email
    email_match = re.search(r"EMAIL[^:]*:([^\r\n]+)", vcard_text)
    if email_match:
        fields["email"] = email_match.group(1).strip()

    # Extract Phone
    tel_match = re.search(r"TEL[^:]*:([^\r\n]+)", vcard_text)
    if tel_match:
        fields["phone"] = tel_match.group(1).strip()

    # Extract Organization
    org_match = re.search(r"ORG[;:]([^\r\n]+)", vcard_text)
    if org_match:
        fields["company_name"] = org_match.group(1).strip()

    # Extract URL (LinkedIn)
    url_match = re.search(r"URL[^:]*:([^\r\n]+)", vcard_text)
    if url_match:
        url = url_match.group(1).strip()
        fields["linkedin"] = url

    return fields

def scan_qr_image(image_path: str) -> str:
    """Decode QR code from image and return the data."""
    img = Image.open(image_path)
    decoded_objects = decode_qr(img)
    if not decoded_objects:
        return None
    return decoded_objects[0].data.decode("utf-8")

async def add_contact_from_qr(file: UploadFile):
    """Scan QR code, extract vCard data, save contact to FAISS, and return details."""
    temp_filename = os.path.join(TEMP_DIR, file.filename or f"{uuid.uuid4()}.png")
    content = await file.read()
    with open(temp_filename, "wb") as f:
        f.write(content)

    try:
        # Decode QR code
        qr_data = scan_qr_image(temp_filename)
        if not qr_data:
            raise HTTPException(status_code=400, detail="No QR code found in image")

        # Check if it's a vCard
        if "BEGIN:VCARD" not in qr_data.upper():
            raise HTTPException(status_code=400, detail="QR code does not contain vCard data")

        # Parse vCard
        fields = parse_vcard(qr_data)

        # Create contact and save to FAISS
        contact_obj = Contact(**fields)
        result = add_contact_logic(contact_obj)

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

# --- IMAGE TEXT EXTRACTION WITH GEMINI ---
def extract_contact_from_image_with_gemini(image_path: str) -> dict:
    """Extract contact information directly from image using Gemini Vision."""
    fields = {"name": "N/A", "email": "N/A", "phone": "N/A", "linkedin": "N/A", "company_name": "N/A"}

    if not GEMINI_API_KEY:
        print("[GEMINI] API key not configured")
        return fields

    try:
        # Load image
        img = Image.open(image_path)

        # Use Gemini to extract contact info directly from image
        model = genai.GenerativeModel('gemini-3-pro-preview')

        prompt = """Analyze this business card or contact image and extract the following information.
Return ONLY a JSON object with these exact keys (use "N/A" if not found):
{
    "name": "full name of the person",
    "email": "email address",
    "phone": "phone number",
    "linkedin": "linkedin profile URL",
    "company_name": "company or organization name"
}

Important: Return ONLY the JSON object, no other text or markdown formatting."""

        response = model.generate_content([prompt, img])
        response_text = response.text.strip()

        # Clean up response - remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        # Parse JSON response
        extracted = json.loads(response_text)

        # Update fields with extracted values
        for key in fields:
            if key in extracted and extracted[key] and extracted[key] != "N/A":
                fields[key] = extracted[key]

        print(f"[GEMINI] Extracted: {fields}")
        return fields

    except json.JSONDecodeError as e:
        print(f"[GEMINI] JSON parse error: {e}, response: {response_text}")
        return fields
    except Exception as e:
        print(f"[GEMINI] Error extracting contact from image: {e}")
        traceback.print_exc()
        return fields

# --- CONTACT LOGIC ---
def add_contact_logic(contact: Contact):
    """Add contact to FAISS index and auto-generate QR code."""
    contact_id = str(uuid.uuid4())
    summary = f"{contact.name}, {contact.email}, {contact.phone}, {contact.linkedin}, {contact.company_name}"
    metadata = contact.model_dump()
    metadata["summary"] = summary
    metadata["id"] = contact_id
    
    # Auto-generate QR code and store path in metadata
    qr_result = create_qr(contact, contact_id)
    metadata["qr_path"] = qr_result["qr_path"]
    
    faiss_manager.add_document(summary, metadata)
    return {"contact_id": contact_id, "qr_base64": qr_result["qr_base64"]}

async def add_contact_from_image(file: UploadFile):
    temp_filename = os.path.join(TEMP_DIR, file.filename or f"{uuid.uuid4()}.png")
    content = await file.read()
    with open(temp_filename, "wb") as f:
        f.write(content)

    try:
        # Use Gemini to extract contact info directly from image
        fields = extract_contact_from_image_with_gemini(temp_filename)

        # Check if at least some contact info was extracted
        has_info = any(v != "N/A" for k, v in fields.items() if k != "linkedin")
        if not has_info:
            raise HTTPException(status_code=400, detail="No contact information found in image")

        contact_obj = Contact(**fields)
        result = add_contact_logic(contact_obj)

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

def search_logic(query: str):
    results = faiss_manager.search(query, k=4)
    return [{"text": r[0], "meta": r[1]} for r in results]

# --- FASTAPI APP ---
app = FastAPI(title="Contact Assistant API (OpenSource Embedding)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup_event():
    """Log startup information and verify all systems are ready."""
    print("=" * 50)
    print("[STARTUP] Contact Assistant API Starting...")
    print(f"[STARTUP] Data directory: {DATA_DIR}")
    print(f"[STARTUP] QR directory: {QR_DIR}")
    print(f"[STARTUP] Metadata path: {METADATA_PATH}")
    print(f"[STARTUP] FAISS index path: {FAISS_INDEX_PATH}")
    print(f"[STARTUP] Contacts loaded: {len(faiss_manager.doc_metadata)}")
    print(f"[STARTUP] FAISS index ready: {faiss_manager.index is not None}")
    print("=" * 50)

@app.get("/")
async def root(): return {"message": "Contact Assistant API", "version": "1.0.0"}

@app.get("/health/")
async def health_check():
    return {"status": "healthy", "contacts_indexed": len(faiss_manager.doc_metadata), "faiss_ready": faiss_manager.index is not None}

@app.post("/add_contact/")
async def add_contact_route(contact: Contact, api_key: str = Depends(verify_api_key)):
    try: 
        result = add_contact_logic(contact)
        return {
            "status": "success", 
            "message": "Contact added",
            "contact_id": result["contact_id"],
            "qr_base64": result["qr_base64"]
        }
    except Exception as e: traceback.print_exc(); raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_contact_from_image/")
async def add_contact_image_route(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    return await add_contact_from_image(file)

@app.post("/scan_qr/")
async def scan_qr_route(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    """
    Scan a QR code image containing vCard data, save the contact to the database,
    and return the extracted contact details.
    """
    return await add_contact_from_qr(file)

@app.post("/generate_qr/")
async def generate_qr_route(contact: Contact, api_key: str = Depends(verify_api_key)):
    try: 
        result = create_qr(contact)
        return {"status": "success", **result}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/")
async def search_route(query: SearchQuery, api_key: str = Depends(verify_api_key)):
    try: return {"status": "success", "results": search_logic(query.query)}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_contacts/")
async def list_contacts_route(api_key: str = Depends(verify_api_key)):
    """
    List all contacts with their QR codes as base64.
    Returns all contact details including the QR code for each contact.
    """
    try:
        contacts = []
        for meta in faiss_manager.doc_metadata:
            contact_data = {
                "id": meta.get("id", "N/A"),
                "name": meta.get("name", "N/A"),
                "email": meta.get("email", "N/A"),
                "phone": meta.get("phone", "N/A"),
                "linkedin": meta.get("linkedin", "N/A"),
                "company_name": meta.get("company_name", "N/A"),
                "qr_base64": None
            }
            
            # Fetch QR code base64 if available
            contact_id = meta.get("id")
            if contact_id:
                qr_base64 = get_qr_base64(contact_id)
                if qr_base64:
                    contact_data["qr_base64"] = qr_base64
            
            contacts.append(contact_data)
        
        return {
            "status": "success",
            "total_contacts": len(contacts),
            "contacts": contacts
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/contact/{contact_id}")
async def get_contact_route(contact_id: str, api_key: str = Depends(verify_api_key)):
    """
    Get a specific contact by ID with QR code.
    """
    try:
        for meta in faiss_manager.doc_metadata:
            if meta.get("id") == contact_id:
                contact_data = {
                    "id": meta.get("id", "N/A"),
                    "name": meta.get("name", "N/A"),
                    "email": meta.get("email", "N/A"),
                    "phone": meta.get("phone", "N/A"),
                    "linkedin": meta.get("linkedin", "N/A"),
                    "company_name": meta.get("company_name", "N/A"),
                    "qr_base64": get_qr_base64(contact_id)
                }
                return {"status": "success", "contact": contact_data}

        raise HTTPException(status_code=404, detail="Contact not found")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/contact/{contact_id}")
async def delete_contact_route(contact_id: str, api_key: str = Depends(verify_api_key)):
    """
    Delete a contact by ID.
    Also removes the associated QR code file.
    """
    try:
        # Check if contact exists first
        idx = faiss_manager.find_index_by_id(contact_id)
        if idx == -1:
            raise HTTPException(status_code=404, detail="Contact not found")

        # Get contact info before deletion for response
        contact_meta = faiss_manager.doc_metadata[idx]
        contact_name = contact_meta.get("name", "Unknown")

        # Delete QR code file if it exists
        qr_path = os.path.join(QR_DIR, f"qr_{contact_id}.png")
        if os.path.exists(qr_path):
            os.remove(qr_path)

        # Delete from FAISS
        deleted = faiss_manager.delete_document(contact_id)

        if deleted:
            return {
                "status": "success",
                "message": f"Contact '{contact_name}' deleted successfully",
                "deleted_contact_id": contact_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete contact")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/contact/{contact_id}")
async def update_contact_route(contact_id: str, contact_update: ContactUpdate, api_key: str = Depends(verify_api_key)):
    """
    Update a contact by ID.
    Only fields provided in the request body will be updated.
    Automatically regenerates the QR code with updated information.
    """
    try:
        # Check if contact exists
        idx = faiss_manager.find_index_by_id(contact_id)
        if idx == -1:
            raise HTTPException(status_code=404, detail="Contact not found")

        # Get current metadata
        current_meta = faiss_manager.doc_metadata[idx].copy()

        # Update only provided fields
        update_data = contact_update.model_dump(exclude_unset=True)
        if not update_data:
            raise HTTPException(status_code=400, detail="No fields provided for update")

        for field, value in update_data.items():
            if value is not None:
                current_meta[field] = value

        # Rebuild summary text
        new_summary = f"{current_meta.get('name', 'N/A')}, {current_meta.get('email', 'N/A')}, {current_meta.get('phone', 'N/A')}, {current_meta.get('linkedin', 'N/A')}, {current_meta.get('company_name', 'N/A')}"
        current_meta["summary"] = new_summary

        # Regenerate QR code with updated info
        updated_contact = Contact(
            name=current_meta.get("name", "N/A"),
            email=current_meta.get("email", "N/A"),
            phone=current_meta.get("phone", "N/A"),
            linkedin=current_meta.get("linkedin", "N/A"),
            company_name=current_meta.get("company_name", "N/A")
        )

        # Delete old QR code if exists
        old_qr_path = os.path.join(QR_DIR, f"qr_{contact_id}.png")
        if os.path.exists(old_qr_path):
            os.remove(old_qr_path)

        # Generate new QR code with same contact_id
        qr_result = create_qr(updated_contact, contact_id)
        current_meta["qr_path"] = qr_result["qr_path"]

        # Update in FAISS
        updated = faiss_manager.update_document(contact_id, new_summary, current_meta)

        if updated:
            return {
                "status": "success",
                "message": "Contact updated successfully",
                "contact": {
                    "id": contact_id,
                    "name": current_meta.get("name", "N/A"),
                    "email": current_meta.get("email", "N/A"),
                    "phone": current_meta.get("phone", "N/A"),
                    "linkedin": current_meta.get("linkedin", "N/A"),
                    "company_name": current_meta.get("company_name", "N/A"),
                    "qr_base64": qr_result["qr_base64"]
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update contact")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/converse/")
async def converse_route(query: ConverseQuery, api_key: str = Depends(verify_api_key)):
    """
    Conversational endpoint that uses Gemini to provide natural language responses
    based on contact data retrieved from the FAISS database.
    
    Example queries:
    - "Who do I know at Google?"
    - "Find me someone in marketing"
    - "What's John's email address?"
    """
    try:
        # Step 1: Retrieve relevant contacts from FAISS
        retrieved_contacts = faiss_manager.search(query.query, k=query.top_k or 4)
        
        # Step 2: Generate conversational response using Gemini
        response_text = gemini_engine.generate_response(
            query=query.query,
            retrieved_contacts=retrieved_contacts,
            conversation_history=query.conversation_history
        )
        
        # Step 3: Return structured response
        return {
            "status": "success",
            "response": response_text,
            "retrieved_contacts": [
                {
                    "name": meta.get("name", "N/A"),
                    "email": meta.get("email", "N/A"),
                    "phone": meta.get("phone", "N/A"),
                    "linkedin": meta.get("linkedin", "N/A"),
                    "company_name": meta.get("company_name", "N/A")
                }
                for (text, meta) in retrieved_contacts
            ],
            "query": query.query
        }
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Conversation error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
