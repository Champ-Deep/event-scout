# Event Scout - Project Context

## What This Is

Event Scout is a **multi-user contact management and event intelligence platform** built for networking at trade shows, conferences, and business events. Users scan business cards, manage contacts, and get AI-powered insights — all from a mobile-first web app.

**Current Version:** 2.0.0 (multi-user with authentication)
**Stack:** FastAPI (Python) + FAISS vector DB + Google Gemini AI + vanilla JS/Tailwind frontend
**Deployment:** Docker on Railway (3 persistent volumes)
**Frontend:** Single HTML file (`mobile-frontend.html`) deployed to Vercel/Netlify

## End Goal Vision

Event Scout is being transformed into a **full event intelligence partner** — not just a contact scanner, but an AI-powered system that helps users:

1. **Pre-Event:** Identify targets, get exhibitor intelligence, plan booth visits based on user profile
2. **During Event:** Scan cards, get instant lead scores, receive pitch angle suggestions, have a smart assistant that remembers everything
3. **Post-Event:** Auto-research contacts (company + role + pain points), generate personalized pitch decks (Google Slides), trigger email sequences (ChampMail), prepare LinkedIn messages, generate voice call scripts (Voicebox)

See `EVENT_SCOUT_VISION.md` for the full product roadmap.

## Architecture

### Backend (`app.py` — single file, ~1264 lines)

```
FastAPI App
├── Auth: bcrypt passwords + API key header (X-API-Key)
├── Users: JSON file storage (users/users.json)
├── Contacts: FAISS vector DB + pickle metadata (per-user isolation)
├── AI: Google Gemini (2.5-flash primary, 2.0-flash/lite fallbacks)
│   ├── Vision OCR: Business card text extraction
│   └── Conversational: Chat about contacts with RAG
├── QR: vCard QR code generation + scanning
└── Webhook: /contact/{id}/enrich (n8n integration point)
```

### Data Storage

```
users/
├── users.json                    # All user credentials {user_id: {name, email, password_hash}}
└── {user_id}/
    ├── metadata.pickle           # Contact data + embeddings metadata
    ├── faiss.index               # FAISS vector index for semantic search
    └── profile.json              # User personalization settings (Phase 1 addition)
```

Contact metadata is a flexible Python dict stored in pickle — new fields can be added without migration.

### Frontend (`mobile-frontend.html` — single file, ~1681 lines)

Vanilla JS + Tailwind CSS (CDN). Screens:
- **Auth:** Login/Register
- **Contacts:** List with search, CRUD, QR viewing
- **Scan:** Camera-based business card capture → Gemini OCR
- **Chat:** AI assistant powered by Gemini + FAISS retrieval
- **Profile:** User info, export (CSV/JSON), settings

Config is at the top of the file:
```javascript
const CONFIG = {
    API_BASE_URL: 'https://event-scout-production.up.railway.app',
    API_KEY: '...'  // Note: hardcoded in frontend
};
```

## API Endpoints (22 existing + new Phase 1 additions)

### Auth (no API key required)
- `POST /register/` — Create user account
- `POST /login/` — Authenticate, returns user_id
- `GET /health/` — Health check
- `GET /` — Root

### Contacts (require X-API-Key + user_id)
- `POST /add_contact/` — Add contact manually
- `GET /list_contacts/?user_id=` — List all contacts
- `GET /contact/{id}?user_id=` — Get single contact
- `PUT /contact/{id}?user_id=` — Update contact
- `DELETE /contact/{id}?user_id=` — Delete contact
- `POST /search/` — Semantic search (FAISS)
- `POST /converse/` — AI chat about contacts (Gemini)
- `POST /add_contact_from_image/` — Business card OCR scan
- `POST /scan_qr/` — Scan QR with vCard data
- `POST /generate_qr/` — Generate QR (no save)
- `GET /export_contacts/?user_id=&format=csv|json` — Export

### Webhook/Integration
- `POST /contact/{id}/enrich` — n8n enrichment webhook (appends notes/links)

### Phase 1 Additions (feature/event-intelligence branch)
- `GET /user/profile/?user_id=` — Get personalization settings
- `PUT /user/profile/?user_id=` — Update personalization settings
- `POST /contact/{id}/score?user_id=` — Score a lead using Gemini
- `GET /dashboard?user_id=` — Aggregated stats

## Integration Ecosystem

Event Scout connects to other internal tools via **n8n webhooks** (self-hosted on cloud):

| Tool | Location | Purpose | Status |
|------|----------|---------|--------|
| **n8n** | Cloud (self-hosted) | Workflow orchestration bus | Running |
| **Scrapper** | `/projects/Scrapper` | B2B web scraping (Playwright + FireCrawl) | Not deployed |
| **ChampMail** | `/projects/ChampMail` | Email outreach (built-in SMTP, AI sequences) | Not deployed |
| **Voicebox** | `/projects/Voice_Studio/voicebox` | Voice synthesis + call scripts | Available |
| **gcc_interactive_pitch** | `/projects/gcc_interactive_pitch` | HTML pitch deck template | Reference only |
| **Deepu-N8N** | `/projects/Deepu-N8N` | n8n workflow library (lead research, routing) | Reference |
| **Perplexity API** | External | Deep research (interim until Scrapper deployed) | Via n8n |
| **Google Slides API** | External | Pitch deck generation | Phase 2 |

### Integration Pattern
```
Event Scout → webhook → n8n → [Scrapper/ChampMail/Voicebox/Perplexity/Slides] → callback → Event Scout
```

Event Scout never calls tools directly. n8n orchestrates everything and calls back to Event Scout's callback endpoints.

## Environment Variables

```bash
APP_API_KEY=<secure-token>        # Required: API authentication
GEMINI_API_KEY=<google-api-key>   # Required: Gemini AI features
N8N_WEBHOOK_BASE_URL=<url>        # Phase 2: n8n webhook base URL
N8N_API_KEY=<key>                 # Phase 2: n8n authentication
```

## Development Guidelines

### Branch Strategy
- **master** — Production. Do not push untested code here.
- **feature/event-intelligence** — All new intelligence features. Merge to master only after testing.

### Critical Rules
1. **Don't break existing functionality.** Event is imminent. All new features are additive.
2. **Functionality first.** Get things working, polish later.
3. **No schema migrations needed.** Contact metadata is a flexible dict — just add new fields.
4. **New files go under existing user directories.** Profile at `users/{user_id}/profile.json`, pitches at `users/{user_id}/pitches/`.
5. **Test on branch before merging.** Run against Railway deployment or local Docker.

### Key Code Patterns
- All protected routes use `api_key: str = Depends(verify_api_key)` and `user_id: str = Query(...)`
- FAISS manager is instantiated per-request: `faiss_mgr = get_user_faiss_manager(user_id)`
- Contact metadata is a dict, not a Pydantic model — read existing fields with `.get("field", default)`
- Gemini has multi-model fallback: try 2.5-flash → 2.0-flash → 2.0-flash-lite
- QR codes auto-generate on contact add and store at `saved_qr/qr_{contact_id}.png`

### Adding New Endpoints
Follow the existing pattern:
```python
@app.post("/new_endpoint/")
async def new_endpoint(
    some_param: str = Query(...),
    api_key: str = Depends(verify_api_key)
):
    try:
        # ... logic ...
        return {"status": "success", ...}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
```

## Phase Roadmap

| Phase | Timeline | Features |
|-------|----------|----------|
| **1** | This week | User personalization, enhanced AI chat, lead scoring |
| **2** | 1-2 weeks post-event | n8n research workflows, Google Slides pitch decks, Intel dashboard |
| **3** | 3-4 weeks | Email sequences (ChampMail), LinkedIn prep, voice call scripts |
| **4** | Future | LinkedIn OAuth, pre-event intelligence, exhibitor suggestions, notifications |

## Reusable Code from Ecosystem

| What | Where | Use For |
|------|-------|---------|
| Perplexity research prompts | `Deepu-N8N/n8n maker/lead-research-pitch-workflow.json` | Contact research n8n workflow |
| Pitch generation prompts | Same file | Pitch deck content generation |
| HTML slide template | `gcc_interactive_pitch/mobile-pitch.html` | Fallback pitch deck format |
| Webhook patterns | `ChampMail/backend/app/api/v1/webhooks.py` | Email event handling, n8n events |
| Voice call scripts | `Deepu-N8N/n8n maker/CHAMP Calling Agent/` | Voice qualification workflow |
| Email workflows | `Email_automation_master/` | n8n email automation patterns |
