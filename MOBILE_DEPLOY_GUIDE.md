# Event Scout — Mobile Web App Deployment Guide

## Quick Deploy to Vercel (5 minutes)

### Prerequisites
- Vercel account (free tier works)
- Railway backend deployed and running
- Your `APP_API_KEY` value

### Step 1: Update Config

Open `mobile-frontend.html` and find the CONFIG block near line 1300:

```javascript
const CONFIG = {
    API_BASE_URL: 'https://your-railway-app.railway.app', // Your Railway URL
    API_KEY: 'your-app-api-key-here' // Your production API key
};
```

Replace both values with your actual Railway backend URL and API key.

### Step 2: Deploy to Vercel

**Option A — Vercel CLI (fastest):**
```bash
npm i -g vercel
cd /path/to/Round_Table_Conferrance_app_Authenticated
vercel --prod
```

**Option B — Vercel Dashboard:**
1. Go to https://vercel.com/new
2. Import your GitHub repo (or drag-drop the folder)
3. Framework preset: "Other"
4. Root directory: `.` (the folder with mobile-frontend.html)
5. Click Deploy

The `vercel.json` file auto-configures:
- `/` → redirects to `mobile-frontend.html`
- Security headers (nosniff, frame deny, XSS protection)
- Mascot image caching (1 year, immutable)
- Manifest JSON content type

### Step 3: Test Deployment

Open the Vercel URL on your phone and check:
- [ ] Splash screen shows mascot with bounce animation
- [ ] Login/Register screens work
- [ ] Contacts load and display
- [ ] Search (both client-side and semantic) works
- [ ] Add/Edit/Delete contacts work
- [ ] Scanner opens camera and captures
- [ ] AI Chat sends and receives messages
- [ ] PWA "Add to Home Screen" prompt appears

### Step 4: Add to Home Screen (PWA)

**iOS Safari:**
1. Open the Vercel URL
2. Tap Share → "Add to Home Screen"
3. The app icon will be the Scout mascot

**Android Chrome:**
1. Open the Vercel URL
2. Tap the install banner, or Menu → "Add to Home Screen"

---

## File Structure

```
mobile-frontend.html   ← The complete app (single file, zero build)
manifest.json          ← PWA manifest for home screen install
vercel.json            ← Vercel deployment config
Circle Scout Mascot.jpeg ← Mascot image (1024x1024)
frontend.html          ← Original desktop frontend (kept as backup)
app.py                 ← Backend (deployed separately on Railway)
```

## Architecture

```
┌─────────────────┐      ┌──────────────────────┐
│  Mobile Frontend │ ←──→ │   FastAPI Backend     │
│  (Vercel/CDN)   │ HTTPS │   (Railway)           │
│                  │      │                        │
│  - HTML/CSS/JS   │      │  - Auth (bcrypt)       │
│  - Mascot assets │      │  - FAISS vector search │
│  - PWA manifest  │      │  - Gemini AI chat      │
│  - Camera API    │      │  - Business card OCR   │
│  - localStorage  │      │  - QR generation       │
└─────────────────┘      └──────────────────────┘
```

## Feature Overview

| Feature | Status | Notes |
|---------|--------|-------|
| Mobile-first responsive UI | ✅ | 480px max-width, safe area support |
| Scout mascot integration | ✅ | Splash, auth, header, empty states, chat avatar |
| Login / Register | ✅ | Full auth flow with localStorage persistence |
| Contact CRUD | ✅ | Add, edit, delete with QR code display |
| Client-side search | ✅ | Instant filter by name, email, company, phone |
| Semantic AI search | ✅ | FAISS vector search via backend |
| Business card scanner | ✅ | Camera capture + image upload → Gemini OCR |
| QR code scanner | ✅ | Camera-based QR scanning via backend |
| AI Scout chat | ✅ | Natural language queries about contacts |
| PWA manifest | ✅ | Home screen installable |
| Bottom navigation | ✅ | Contacts, Scan, +Add, AI, More |
| Toast notifications | ✅ | Success/error feedback |
| Skeleton loading | ✅ | Shimmer animation during data load |

## Troubleshooting

**"Network error" on login:**
- Check that `API_BASE_URL` is correct in the CONFIG
- Check that the Railway backend is running (`/health` endpoint)
- CORS is enabled in the backend (`allow_origins=["*"]`)

**Camera not working:**
- Must be served over HTTPS (Vercel does this automatically)
- User must grant camera permission
- Fallback: use "Upload" button instead

**Contacts not loading:**
- Check `API_KEY` matches the `APP_API_KEY` env var on Railway
- Check the `X-API-Key` header is being sent

**PWA not installing:**
- Must be served over HTTPS
- `manifest.json` must be accessible at the root
- Check browser DevTools → Application → Manifest
