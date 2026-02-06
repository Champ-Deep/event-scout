# Event Scout - Contact Management System

**Multi-user contact management system with QR code generation, semantic search, and AI-powered conversational queries.**

---

## Features

- 🔐 **User Authentication**: Secure registration & login with bcrypt password hashing
- 👥 **Contact Management**: Create, read, update, delete contacts with per-user isolation
- 📱 **QR Code Generation**: Auto-generate vCard QR codes for each contact
- 🔍 **Semantic Search**: FAISS-powered vector search for intelligent contact lookup
- 🤖 **AI Assistant**: Conversational queries powered by Google Gemini
- 📸 **Business Card Scanner**: Extract contacts from images using Gemini Vision
- 🔄 **Data Persistence**: User data and contacts survive container restarts

---

## Tech Stack

- **Backend**: FastAPI + Python 3.11
- **Database**: FAISS (vector database) + JSON (user data)
- **AI**: Google Gemini API (flash-1.5)
- **Embeddings**: SentenceTransformer (all-MiniLM-L6-v2)
- **Auth**: bcrypt + API key authentication
- **Deployment**: Docker + Railway (production)

---

## Quick Start (Local Development)

### Prerequisites

- Python 3.11+
- Docker (optional)

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd Round_Table_Conferrance_app_Authenticated
```

### 2. Set Up Environment Variables

```bash
# Generate a secure API key
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Create .env file
cp .env.example .env
```

Edit `.env` and add your keys:

```env
APP_API_KEY=<your-generated-key>
GEMINI_API_KEY=<your-gemini-key>
```

Get your Gemini API key from: https://aistudio.google.com/app/apikey

### 3. Run with Docker (Recommended)

```bash
docker-compose up --build
```

Backend will be available at: http://localhost:8000

### 4. Run without Docker

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

---

## Railway Deployment (Production)

### Step 1: Create Railway Project

1. Go to [Railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub"
3. Select your repository
4. Railway will auto-detect the Dockerfile

### Step 2: Configure Environment Variables

In Railway dashboard, add these variables:

```env
APP_API_KEY=<your-secure-key>
GEMINI_API_KEY=<your-gemini-key>
```

### Step 3: Configure Persistent Volumes

Railway needs volume mounts to persist data across deployments:

**In Railway dashboard → Settings → Volumes:**

1. **Add Volume 1:**
   - Mount Path: `/app/users`
   - Size: 1GB

2. **Add Volume 2:**
   - Mount Path: `/app/saved_qr`
   - Size: 1GB

3. **Add Volume 3:**
   - Mount Path: `/app/data`
   - Size: 1GB

### Step 4: Deploy

Railway will automatically deploy when you push to your main branch.

### Step 5: Verify Deployment

Check health endpoint:

```bash
curl https://<your-railway-domain>.railway.app/health/
```

Expected response:

```json
{
  "status": "healthy",
  "multi_user": true,
  "total_users": 0,
  "gemini_configured": true
}
```

### Step 6: Get Your Railway Domain

- In Railway dashboard, click "Settings" → "Generate Domain"
- Your API will be available at: `https://<your-app>.railway.app`

---

## API Endpoints

### Authentication

**Register User**
```bash
POST /register/
{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "secure_password"
}
```

**Login**
```bash
POST /login/
{
  "email": "john@example.com",
  "password": "secure_password"
}
```

### Contact Management

All endpoints require:
- Header: `X-API-Key: <your-app-api-key>`
- Query/Body param: `user_id`

**Add Contact**
```bash
POST /add_contact/
{
  "contact": {
    "name": "Jane Smith",
    "email": "jane@example.com",
    "phone": "+1234567890",
    "linkedin": "https://linkedin.com/in/janesmith",
    "company_name": "Acme Corp"
  },
  "user_id": "<user-id-from-login>"
}
```

**List Contacts**
```bash
GET /list_contacts/?user_id=<user-id>
```

**Search Contacts**
```bash
POST /search/
{
  "query": "software engineers at tech companies",
  "user_id": "<user-id>"
}
```

**Delete Contact**
```bash
DELETE /contact/<contact-id>?user_id=<user-id>
```

**Update Contact**
```bash
PUT /contact/<contact-id>?user_id=<user-id>
{
  "name": "Jane Doe",
  "phone": "+9876543210"
}
```

### AI Features

**Conversational Query**
```bash
POST /converse/
{
  "query": "Who should I reach out to about product management?",
  "user_id": "<user-id>",
  "top_k": 4
}
```

**Scan Business Card**
```bash
POST /add_contact_from_image/
- Form data: file (image)
- Query param: user_id
```

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Frontend (HTML)                 │
│              (Login/Register/CRUD UI)            │
└─────────────────┬───────────────────────────────┘
                  │ HTTPS
                  │
┌─────────────────▼───────────────────────────────┐
│              FastAPI Backend                     │
│           (Railway Deployment)                   │
│  ┌──────────────────────────────────────┐       │
│  │  User Auth (bcrypt + API key)        │       │
│  └──────────────────────────────────────┘       │
│  ┌──────────────────────────────────────┐       │
│  │  Per-User FAISS Indexes              │       │
│  │  /app/users/<user-id>/               │       │
│  └──────────────────────────────────────┘       │
│  ┌──────────────────────────────────────┐       │
│  │  Gemini AI Integration               │       │
│  │  (Conversations + Vision)            │       │
│  └──────────────────────────────────────┘       │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│          Persistent Volumes (Railway)            │
│  - /app/users (user FAISS indexes)              │
│  - /app/saved_qr (QR code images)               │
│  - /app/data (legacy metadata)                  │
│  - /app/users.json (user credentials)           │
└──────────────────────────────────────────────────┘
```

---

## Security Best Practices

✅ **Implemented:**
- bcrypt password hashing (salt rounds: 12)
- API key authentication for all routes
- Environment-based secrets (no hardcoded keys)
- CORS configured for frontend access
- Per-user data isolation
- Volume persistence verification on startup

⚠️ **Production Hardening (Future):**
- Rate limiting (protect Gemini API quota)
- HTTPS enforcement
- JWT token-based auth (replace API key)
- Email verification
- Password reset flow
- Input sanitization (prevent injection attacks)
- Secrets rotation

---

## Testing

Run the multi-user test suite:

```bash
python test_multi_user_flow.py
```

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for detailed test scenarios.

---

## Troubleshooting

### Volume Persistence Issues

**Symptom:** Data disappears after container restart

**Solution:**
1. Check Railway logs for volume mount errors:
   ```
   [INIT] ✗ WARNING: Volume persistence test failed
   ```
2. Verify volumes are configured in Railway dashboard
3. Ensure mount paths match: `/app/users`, `/app/saved_qr`, `/app/data`

### Gemini API Errors

**Symptom:** `[GEMINI] API error: ...`

**Solution:**
1. Verify `GEMINI_API_KEY` is set in Railway environment
2. Check API quota: https://aistudio.google.com/app/apikey
3. App will fallback to non-AI search if Gemini unavailable

### Health Check Failures

**Symptom:** Railway shows "unhealthy" status

**Solution:**
1. Check logs for startup errors
2. Verify all dependencies installed (requirements.txt)
3. Ensure port 8000 is exposed in Dockerfile

---

## Development Roadmap

- [x] Multi-user authentication
- [x] Contact CRUD operations
- [x] QR code generation/scanning
- [x] Semantic search with FAISS
- [x] Gemini AI conversations
- [x] Docker deployment
- [x] Volume persistence
- [ ] Frontend UI (HTML/CSS/JS)
- [ ] Email verification
- [ ] Password reset
- [ ] CSV import/export
- [ ] Admin dashboard
- [ ] Rate limiting
- [ ] Observability/APM

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

MIT License - see LICENSE file for details

---

## Support

- **Issues**: https://github.com/<your-repo>/issues
- **Docs**: https://github.com/<your-repo>/wiki
- **Email**: <your-email>

---

**Built with ❤️ for Champions Ranch**
