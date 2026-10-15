# EventScout — Complete Backend API Reference

> **Source files:** [`app.py`](file:///c:/Users/harsha.c/Desktop/EventScout/apps/backend/app.py) (legacy routes) + [`api_routes.py`](file:///c:/Users/harsha.c/Desktop/EventScout/apps/backend/api_routes.py) (unified `/api/*` routes)
> 
> **Total unique endpoints: 76**

---

## 🟢 SYSTEM / PUBLIC APIs (5 endpoints)

These require no authentication and are used for health checks, debugging, and public card viewing.

### 1. `GET /`
**Purpose:** Root endpoint — API version check  
**Payload:** None  
**Response:**
```json
{ "message": "Event Scout Intelligence API", "version": "3.4.0" }
```

### 2. `GET /health/`
**Purpose:** Health check with graceful degradation  
**Query Params:** `debug` (bool, optional)  
**Response:**
```json
{
  "status": "healthy",
  "database": "postgresql",
  "database_connected": false,
  "total_users": 0,
  "gemini_configured": true,
  "openrouter_configured": true,
  "webhook_configured": true,
  "version": "3.4.0",
  "timestamp": "2026-09-24T07:00:00Z"
}
```

### 3. `GET /api/health`
**Purpose:** Lightweight API router health check  
**Payload:** None  
**Response:**
```json
{
  "status": "healthy",
  "api": "EventScout Backend API",
  "databaseReady": true,
  "timestamp": "2026-09-24T07:00:00Z"
}
```

### 4. `GET /debug`
**Purpose:** Configuration diagnostic (no secrets)  
**Payload:** None  
**Response:**
```json
{
  "env_database_url": "postgres://***@host",
  "actual_url_in_use": "postgres://***@host",
  "network_type": "public_proxy (reliable)",
  "gemini_configured": true,
  "openrouter_configured": true,
  "webhook_configured": true,
  "environment_keys": ["DATABASE_URL", "GEMINI_API_KEY", ...]
}
```

### 5. `GET /card/{token}`
**Purpose:** Public shareable digital business card viewer (no auth)  
**Path Params:** `token` (string)  
**Response:**
```json
{
  "status": "success",
  "card": { "name": "...", "title": "...", "email": "..." },
  "created_at": "2026-09-24T07:00:00Z"
}
```

---

## 🔐 USER / AUTH APIs (8 endpoints)

### 6. `POST /register/`
**Purpose:** Create new user account  
**Payload:**
```json
{ "name": "Sarah Nair", "email": "s.nair@lakeb2b.com", "password": "password123" }
```
**Response:**
```json
{
  "status": "success",
  "message": "User registered successfully",
  "user_id": "uuid",
  "name": "Sarah Nair",
  "email": "s.nair@lakeb2b.com",
  "is_admin": false
}
```

### 7. `POST /login/`
**Purpose:** Email/password login  
**Payload:**
```json
{ "email": "s.nair@lakeb2b.com", "password": "password123" }
```
**Response:**
```json
{
  "status": "success",
  "message": "Login successful",
  "user_id": "uuid",
  "name": "Sarah Nair",
  "email": "s.nair@lakeb2b.com",
  "is_admin": false
}
```

### 8. `GET /auth/linkedin/url`
**Purpose:** Get LinkedIn OAuth authorization URL  
**Query Params:** `state` (string, optional)  
**Response:**
```json
{ "url": "https://www.linkedin.com/oauth/v2/authorization?..." }
```

### 9. `POST /auth/linkedin/callback`
**Purpose:** Exchange LinkedIn OAuth code for session  
**Payload:**
```json
{ "code": "linkedin_auth_code" }
```
**Response:**
```json
{
  "status": "success",
  "token": "uuid",
  "refresh_token": "uuid",
  "expires_in": 3600,
  "user": { "id": "uuid", "name": "...", "email": "...", "is_admin": false }
}
```

### 10. `POST /auth/refresh`
**Purpose:** Refresh auth token  
**Query Params:** `refresh_token` (string)  
**Response:**
```json
{ "status": "success", "token": "uuid", "refresh_token": "uuid", "expires_in": 3600 }
```

### 11. `GET /user/validate`
**Purpose:** Validate user session exists  
**Query Params:** `user_id` (string)  
**Response:**
```json
{ "status": "valid", "valid": true, "user_id": "uuid", "name": "...", "email": "...", "is_admin": false }
```

### 12. `GET /user/profile/`
**Purpose:** Get user profile/preferences  
**Query Params:** `user_id` (string)  
**Response:**
```json
{ "status": "success", "profile": { "company_name": "...", "icp_rules": [...], "preferred_ai_model": "claude-opus" } }
```

### 13. `PUT /user/profile/`
**Purpose:** Update user profile  
**Query Params:** `user_id` (string)  
**Payload:** JSON body with profile fields  
**Response:** `{ "status": "error", "message": "Endpoint temporarily disabled" }` *(currently disabled)*

---

## 👤 USER / CONTACT MANAGEMENT APIs (12 endpoints)

### 14. `POST /add_contact/`
**Purpose:** Add a contact manually (JSON)  
**Payload:**
```json
{ "contact": { "name": "...", "email": "...", "phone": "...", ... }, "user_id": "uuid" }
```
**Response:**
```json
{ "status": "success", "message": "Contact added", "contact_id": "uuid", "qr_base64": "base64..." }
```

### 15. `POST /add_contact_from_image/`
**Purpose:** Upload business card image → OCR → create contact  
**Content-Type:** `multipart/form-data`  
**Payload:** `file` (image), `user_id` (query param)  
**Response:** Full contact object with extracted fields + QR base64

### 16. `POST /scan_qr/`
**Purpose:** Upload QR code image → decode → create contact  
**Content-Type:** `multipart/form-data`  
**Payload:** `file` (image), `user_id` (query param)  
**Response:** Full contact object

### 17. `POST /generate_qr/`
**Purpose:** Generate vCard QR code for a contact  
**Payload:** Contact object `{ "name": "...", "email": "...", ... }`  
**Response:** `{ "status": "success", "qr_base64": "base64..." }`

### 18. `GET /list_contacts/`
**Purpose:** List user's contacts with date filters  
**Query Params:** `user_id`, `date_filter` (today|yesterday|last_week|last_month)  
**Response:**
```json
{
  "status": "success",
  "total_contacts": 42,
  "contacts": [{ "id": "uuid", "name": "...", "email": "...", "lead_score": 85, ... }]
}
```

### 19. `GET /contact/{contact_id}`
**Purpose:** Get single contact detail  
**Query Params:** `user_id`  
**Response:** `{ "status": "success", "contact": { ... full contact with score breakdown, QR, etc. } }`

### 20. `PUT /contact/{contact_id}`
**Purpose:** Update contact fields  
**Query Params:** `user_id`  
**Payload:** `ContactUpdate` (partial fields: name, email, phone, notes, etc.)  
**Response:** `{ "status": "success", "contact": { ... updated contact ... } }`

### 21. `DELETE /contact/{contact_id}`
**Purpose:** Delete a contact  
**Query Params:** `user_id`  
**Response:** `{ "status": "success", "message": "Contact 'Name' deleted", "deleted_contact_id": "uuid" }`

### 22. `POST /search/`
**Purpose:** FAISS semantic search across user's contacts  
**Payload:** `{ "query": "enterprise healthcare buyers", "user_id": "uuid" }`  
**Response:** `{ "status": "success", "results": [{ "text": "...", "meta": { ... } }] }`

### 23. `GET /shared_contacts/`
**Purpose:** List all contacts across all users (shared pool)  
**Query Params:** `limit` (int, default 100)  
**Response:** `{ "status": "success", "total": 42, "contacts": [...] }`

### 24. `POST /contact/{contact_id}/accepted`
**Purpose:** Trigger n8n webhook when user accepts a scanned card  
**Query Params:** `user_id`  
**Response:** `{ "status": "success", "message": "Webhook sent" }`

### 25. `GET /api/contacts`
**Purpose:** List contacts with filtering (unified route)  
**Query Params:** `search`, `stage`, `scoreBand`, `eventId`, `limit`, `cursor`  
**Response:** `{ "contacts": [{ ... formatted Contact ... }], "cursor": null }`

### 26. `GET /api/contacts/{contact_id}` / `POST /api/contacts` / `PATCH /api/contacts/{contact_id}` / `DELETE /api/contacts/{contact_id}`
**Purpose:** Full CRUD via unified API router  
**Create Payload:**
```json
{ "firstName": "...", "lastName": "...", "company": "...", "title": "...", "email": "...", "phone": "...", "notes": "...", "score": 85, "source": "scan" }
```
**Patch Payload:** `{ "baseVersion": 1, "patch": { "notes": "...", "score": 90 } }`  
**Response:** Formatted Contact object / `{ "status": "deleted", "id": "..." }`

---

## 🤖 AI / INTELLIGENCE APIs (9 endpoints)

### 27. `POST /converse/` (Legacy — full AI pipeline)
**Purpose:** AI Copilot chat with FAISS retrieval, exhibitor context, intent detection  
**Payload:**
```json
{
  "query": "Who are my hottest leads?",
  "user_id": "uuid",
  "conversation_history": [{ "role": "user", "content": "..." }],
  "top_k": 4,
  "admin_mode": false
}
```
**Response:**
```json
{
  "status": "success",
  "response": "### Top Leads\n...",
  "model": "Claude Opus 4.6",
  "retrieved_contacts": [{ "id": "...", "name": "...", "lead_score": 92 }],
  "ui_components": [{ "type": "contact_card", "data": { ... } }],
  "intent": "research",
  "query": "..."
}
```

### 28. `POST /api/converse` (Unified route — simpler)
**Purpose:** AI Copilot with live DB context  
**Payload:** `{ "query": "...", "model": "claude-opus", "conversationHistory": [...] }`  
**Response:** `{ "reply": "...", "modelUsed": "Claude Opus 4.6", "timestamp": "..." }`

### 29. `GET /models/` / `GET /api/models`
**Purpose:** List available AI models  
**Response:**
```json
{
  "status": "success",
  "models": [
    { "key": "claude-opus", "name": "Claude Opus 4.6", "provider": "Anthropic", "recommended": true },
    { "key": "gpt-5", "name": "GPT-5.2 Omni", "provider": "OpenAI" },
    { "key": "gemini-pro", "name": "Gemini 2.5 Pro", "provider": "Google" }
  ]
}
```

### 30. `POST /contact/{contact_id}/score`
**Purpose:** AI-powered ICP scoring using Gemini  
**Query Params:** `user_id`  
**Response:**
```json
{
  "status": "success",
  "contact_id": "uuid",
  "score": 92,
  "temperature": "hot",
  "reasoning": "Role matches ICP...",
  "breakdown": { "titleScore": 30, "companyFit": 35, "engagementScore": 25 },
  "recommended_actions": ["Schedule meeting", "Send pitch deck"]
}
```

### 31. `POST /contacts/score_all`
**Purpose:** Batch re-score all unscored contacts  
**Query Params:** `user_id`  
**Response:** `{ "status": "success", "scored": 12, "skipped": 3, "errors": 0, "total": 15 }`

### 32. `POST /contact/{contact_id}/enrich`
**Purpose:** Enrich contact with external data (notes, links)  
**Payload (legacy):** `{ "notes_append": "Met at booth B-104", "links": [{ "url": "...", "label": "..." }] }`  
**Response (unified):** `{ "status": "enriched", "contactId": "...", "data": { "companyDomain": "...", "headcount": "...", ... } }`

### 33. `POST /contact/{contact_id}/pipeline/run`
**Purpose:** Trigger AI pipeline (Research → Score → Pitch → Deck)  
**Response:** `{ "status": "initiated", "pipelineId": "...", "steps": ["research","score","pitch","deck"] }`

### 34. `POST /contact/{contact_id}/pipeline/retry`
**Purpose:** Retry failed pipeline stage  
**Response:** `{ "status": "retrying", "contactId": "...", "step": "generating_deck" }`

### 35. `GET /contact/{contact_id}/pipeline`
**Purpose:** Get pipeline status  
**Response:** `{ "contactId": "...", "status": "complete", "pitchSubject": "...", "gammaDeckUrl": "...", "completedAt": "..." }`

---

## 🎙️ VOICE & AUDIO APIs (2 endpoints)

### 36. `POST /contact/{contact_id}/audio_note`
**Purpose:** Save voice note transcript to a contact  
**Query Params:** `user_id`  
**Payload:** `{ "transcript": "Great meeting...", "audio_base64": "base64..." }`  
**Response:** `{ "status": "success", "message": "Audio note saved", "total_audio_notes": 3 }`

### 37. `POST /transcribe_audio/`
**Purpose:** Transcribe audio file using Gemini  
**Content-Type:** `multipart/form-data`  
**Payload:** `file` (audio), `user_id` (query)  
**Response:** `{ "status": "success", "transcript": "The patient presented with..." }`

---

## 💳 DIGITAL BUSINESS CARD APIs (3 endpoints)

### 38. `GET /user/card/`
**Purpose:** Get user's digital business card  
**Query Params:** `user_id`  
**Response:** `{ "status": "success", "card": { ... }, "shareable_url": "https://...", "is_active": true }`

### 39. `PUT /user/card/`
**Purpose:** Update digital business card fields  
**Query Params:** `user_id`  
**Payload:** `{ "name": "...", "title": "...", "email": "...", "phone": "...", "company": "..." }`  
**Response:** `{ "status": "success", "card": { ... }, "shareable_url": "..." }`

### 40. `POST /user/card/generate_token/`
**Purpose:** Generate shareable URL + QR code for card  
**Query Params:** `user_id`  
**Response:**
```json
{
  "status": "success",
  "shareable_url": "https://event-scout-delta.vercel.app/card/uuid",
  "shareable_token": "uuid",
  "qr_code_base64": "base64...",
  "message": "Share this QR code or URL!"
}
```

---

## 📊 USER DASHBOARD & EXPORT APIs (3 endpoints)

### 41. `GET /dashboard`
**Purpose:** User's personal dashboard metrics  
**Query Params:** `user_id`  
**Response:**
```json
{
  "status": "success",
  "total_contacts": 42,
  "scored_contacts": 38,
  "unscored_contacts": 4,
  "by_temperature": { "hot": 12, "warm": 20, "cold": 6 },
  "top_contacts": [{ "id": "...", "name": "...", "lead_score": 95 }],
  "has_profile": true,
  "current_event": "WHX Dubai 2026"
}
```

### 42. `GET /export_contacts/`
**Purpose:** Export contacts as CSV or JSON  
**Query Params:** `user_id`, `format` (csv|json)  
**Response (JSON):** `{ "status": "success", "total": 42, "contacts": [...] }`  
**Response (CSV):** StreamingResponse with `Content-Disposition: attachment`

### 43. `GET /api/contact/{contact_id}/qr`
**Purpose:** Get digital vCard QR code  
**Response:** `{ "contactId": "...", "qrDataUrl": "data:image/svg+xml;base64,..." }`

---

## 🔄 SYNC (OFFLINE-FIRST) APIs (3 endpoints)

### 44. `POST /api/sync/push`
**Purpose:** Push offline mutations to server  
**Payload:**
```json
{
  "items": [{
    "id": "outbox-uuid",
    "entityType": "contact",
    "entityId": "uuid",
    "action": "create",
    "payload": "{\"firstName\": \"...\", ...}",
    "createdAt": "2026-09-24T07:00:00Z"
  }]
}
```
**Response:** `{ "acknowledgedIds": ["outbox-uuid"], "conflicts": [] }`

### 45. `GET /api/sync/pull`
**Purpose:** Pull server changes to local device  
**Query Params:** `cursor`, `eventId`, `limit`  
**Response:** `{ "contacts": [...], "cursor": null, "serverTimestamp": "..." }`

### 46. `GET /api/sync/state` *(not in api_routes but referenced in frontend)*

---

## 🏢 EVENT & EXHIBITOR APIs (5 endpoints)

### 47. `GET /api/events`
**Purpose:** List active events  
**Response:** Array of `{ "id": "...", "name": "World Health Expo...", "location": "...", "startsAt": "...", "endsAt": "...", "activeReps": 14, "totalLeads": 312 }`

### 48. `GET /api/events/{event_id}/seats`
**Purpose:** Event rep roster  
**Response:** Array of `{ "id": "seat-uuid", "name": "...", "email": "...", "role": "admin|rep", "scansCount": 184 }`

### 49. `GET /exhibitors/` / `GET /api/exhibitors`
**Purpose:** Browse exhibitors directory  
**Query Params:** `search`, `category`, `hall`  
**Response:** Array of `{ "id": "...", "name": "...", "booth": "B-104", "hall": "Hall B", "category": "Diagnostics", "country": "USA", "website": "...", "description": "...", "tags": [...] }`

### 50. `GET /exhibitors/categories/` / `GET /api/exhibitors/categories`
**Purpose:** List exhibitor categories with counts  
**Response:** `[{ "category": "Diagnostics", "count": 12 }, ...]`

### 51. `GET /api/live`
**Purpose:** SSE live event stream for real-time telemetry  
**Response:** `text/event-stream` with JSON payloads:
```
data: {"type": "contact_scanned", "data": {"contactId": "...", "name": "...", "score": 92}, "timestamp": "..."}
```

---

## 📁 FILE MANAGEMENT APIs (6 endpoints)

### 52. `GET /files/list` / `GET /api/files/list`
**Purpose:** List uploaded event documents  
**Response:** Array of `{ "id": "...", "filename": "...", "fileType": "pdf", "category": "sales", "fileSize": 1024000, "downloadCount": 5, "createdAt": "..." }`

### 53. `GET /files/download/{file_id}`
**Purpose:** Download event file binary  
**Response:** StreamingResponse with file bytes

### 54. `GET /contact/{contact_id}/files`
**Purpose:** List files attached to a contact  
**Response:** Array of `{ "id": "...", "filename": "Tailored_Pitch_Deck.pdf", "fileType": "pdf", ... }`

### 55. `GET /contact/files/download/{file_id}`
**Purpose:** Download contact-attached file  
**Response:** StreamingResponse

### 56. `POST /api/contact/{contact_id}/files` *(Admin route in app.py)*
**Purpose:** Upload file to a contact record  
**Response:** `{ "status": "success", "file_id": "uuid" }`

### 57. `GET /broadcasts/active` / `GET /api/broadcasts/active`
**Purpose:** Get active team broadcast alerts  
**Response:** Array of `{ "id": "...", "message": "...", "priority": "urgent", "isActive": true, "createdAt": "..." }`

### 58. `POST /broadcasts/{broadcast_id}/dismiss`
**Purpose:** Dismiss an active broadcast  
**Response:** `{ "status": "dismissed", "id": "..." }`

---

## 🛡️ ADMIN APIs (28 endpoints)

### 59. `GET /admin/dashboard` / `GET /api/admin/dashboard`
**Purpose:** Real-time aggregated event metrics  
**Response:**
```json
{
  "totalScans": 1248,
  "tierALeads": 312,
  "tierBLeads": 540,
  "tierCLeads": 396,
  "activeReps": 14,
  "syncHealth": "100%",
  "conflicts": 0,
  "timestamp": "..."
}
```

### 60. `GET /admin/users` / `GET /api/admin/users`
**Purpose:** List all field reps and admins  
**Response:** Array of `{ "id": "uuid", "name": "...", "email": "...", "role": "admin|rep", "contactsCount": 42, "scansCount": 184, "status": "active", "lastActive": "..." }`

### 61. `POST /api/admin/users`
**Purpose:** Create/invite new field rep  
**Payload:** `{ "name": "...", "email": "...", "password": "...", "isAdmin": false, "role": "rep" }`  
**Response:** `{ "id": "uuid", "name": "...", "email": "...", "role": "rep", "status": "active" }`

### 62. `POST /admin/users/merge` / `POST /api/admin/users/merge`
**Purpose:** Merge duplicate user accounts  
**Payload:** `{ "sourceUserId": "uuid", "targetUserId": "uuid" }`  
**Response:** `{ "status": "success", "message": "User merged into target" }`

### 63. `GET /admin/contacts`
**Purpose:** List all contacts across all users (admin view)  
**Response:** Array of contact objects with user attribution

### 64. `DELETE /admin/contact/{contact_id}`
**Purpose:** Admin-delete any contact  
**Response:** `{ "status": "success", "message": "Contact deleted" }`

### 65. `GET /admin/contacts/duplicates` / `GET /api/admin/contacts/duplicates`
**Purpose:** Detect duplicate leads  
**Response:** Array of `{ "id": "dup-...", "confidence": 98, "reason": "Identical email", "primary": {...}, "duplicate": {...} }`

### 66. `POST /admin/contacts/merge` / `POST /api/admin/contacts/merge`
**Purpose:** Merge duplicate contacts  
**Payload:** `{ "primaryId": "uuid", "duplicateId": "uuid", "mergedFields": {} }`  
**Response:** `{ "status": "merged", "primaryId": "uuid" }`

### 67. `POST /admin/contacts/batch_delete` / `POST /api/admin/contacts/batch_delete`
**Purpose:** Bulk delete contacts  
**Payload:** `{ "contactIds": ["uuid1", "uuid2"] }`  
**Response:** `{ "status": "deleted", "count": 2 }`

### 68. `GET /admin/activity` / `GET /api/admin/activity`
**Purpose:** Live audit stream  
**Query Params:** `limit` (int, default 30)  
**Response:** Array of `{ "id": "act-scan-...", "type": "contact_scanned|lead_qualified|system", "title": "...", "subtitle": "...", "repName": "...", "timestamp": "..." }`

### 69. `GET /admin/search`
**Purpose:** Global search across all users' contacts  
**Query Params:** `q` (search string)  
**Response:** Array of matching contacts

### 70. `POST /admin/broadcast` / `POST /api/admin/broadcast`
**Purpose:** Send alert to field reps  
**Payload:** `{ "message": "Booth B closing early!", "priority": "urgent", "recipientIds": [] }`  
**Response:** `{ "id": "uuid", "status": "dispatched", "message": "...", "priority": "urgent" }`

### 71. `PUT /admin/contact/{contact_id}/note` / `PUT /api/admin/contact/{contact_id}/note`
**Purpose:** Save private admin notes on a contact  
**Payload:** `{ "notes": "High-value prospect — CEO confirmed interest" }`  
**Response:** `{ "status": "updated", "contactId": "...", "adminNotes": "..." }`

### 72. `POST /admin/import_exhibitors/` / `POST /api/admin/import_exhibitors`
**Purpose:** Bulk import exhibitors from JSON  
**Payload:** `{ "exhibitors": [{ "name": "...", "booth": "B-104", "hall": "Hall B", ... }] }`  
**Response:** `{ "status": "imported", "count": 45 }`

### 73. `POST /admin/files/upload` / `POST /api/admin/files/upload`
**Purpose:** Upload event collateral  
**Payload:** `{ "filename": "Sales_Deck.pdf", "category": "sales", "description": "...", "fileType": "pdf", "fileSize": 1024000 }`  
**Response:** `{ "id": "uuid", "filename": "...", "status": "uploaded" }`

### 74. `DELETE /admin/files/{file_id}` / `DELETE /api/admin/files/{file_id}`
**Purpose:** Delete event collateral  
**Response:** `{ "status": "deleted", "id": "..." }`

### 75. `DELETE /admin/contact/files/{file_id}`
**Purpose:** Delete contact-attached file  
**Response:** `{ "status": "deleted" }`

---

## 📋 ADMIN — CONTACT LISTS / SEGMENTS (6 endpoints)

### 76. `GET /admin/contact_lists` / `GET /api/admin/contact_lists`
**Purpose:** List all named segments  
**Response:** Array of `{ "id": "uuid", "name": "VIP Tier A", "memberCount": 12, "createdAt": "..." }`

### 77. `POST /admin/contact_lists` / `POST /api/admin/contact_lists`
**Purpose:** Create a new segment  
**Payload:** `{ "name": "Day 2 Follow-ups" }`  
**Response:** `{ "id": "uuid", "name": "...", "memberCount": 0 }`

### 78. `GET /admin/contact_lists/{list_id}/contacts`
**Purpose:** Get contacts in a segment  
**Response:** Array of formatted Contact objects

### 79. `POST /admin/contact_lists/{list_id}/contacts`
**Purpose:** Add contacts to a segment  
**Payload:** `{ "contactIds": ["uuid1", "uuid2"] }`  
**Response:** `{ "status": "added", "count": 2 }`

### 80. `DELETE /admin/contact_lists/{list_id}`
**Purpose:** Delete a segment  
**Response:** `{ "status": "deleted", "id": "..." }`

### 81. `DELETE /admin/contact_lists/{list_id}/contacts/{contact_id}`
**Purpose:** Remove a single contact from a segment  
**Response:** `{ "status": "removed" }`

---

## ⚙️ ADMIN — SYSTEM OPS (7 endpoints)

### 82. `GET /admin/pipelines` / `GET /api/admin/pipelines`
**Purpose:** Overview of all AI pipeline jobs  
**Response:** Array of `{ "id": "pipe-...", "contactName": "...", "company": "...", "score": 92, "status": "complete|researching|pitching", "pitchSubject": "...", "gammaDeckUrl": "..." }`

### 83. `POST /admin/pipeline/settings`
**Purpose:** Update pipeline configuration  
**Payload:** Settings object  
**Response:** `{ "status": "updated" }`

### 84. `POST /admin/backup` / `POST /api/admin/backup`
**Purpose:** Trigger database backup snapshot  
**Response:** `{ "status": "success", "backupFile": "eventscout_backup_20260924.sql.gz", "sizeBytes": 1420800, "message": "..." }`

### 85. `GET /admin/backup/status` / `GET /api/admin/backup/status`
**Purpose:** Check backup health  
**Response:** `{ "lastBackupAt": "...", "status": "healthy", "retentionDays": 30, "backupCount": 14, "autoBackupEnabled": true }`

### 86. `POST /admin/webhook/test` / `POST /api/admin/webhook/test`
**Purpose:** Test n8n/CRM webhook delivery  
**Response:** `{ "status": "delivered", "statusCode": 200, "latencyMs": 48, "payloadEcho": { ... } }`

### 87. `POST /admin/test_webhook`
**Purpose:** Alternative webhook test endpoint  
**Response:** Similar to above

### 88. `GET /admin/export` / `GET /api/admin/export`
**Purpose:** Export all contacts as CSV  
**Response:** CSV string with headers: ID, Name, Email, Phone, Company, Title, Score, ScoreBand, Stage, Notes, Created_At

---

## 📊 Summary by Category

| Category | Count |
|---|---|
| System / Public | 5 |
| User / Auth | 8 |
| User / Contact Management | 13 |
| AI / Intelligence | 9 |
| Voice & Audio | 2 |
| Digital Business Card | 3 |
| User Dashboard & Export | 3 |
| Sync (Offline-First) | 3 |
| Events & Exhibitors | 5 |
| File Management | 6 |
| Admin — Core Ops | 16 |
| Admin — Contact Lists | 6 |
| Admin — System Ops | 7 |
| **GRAND TOTAL** | **~88 route handlers** |

> [!NOTE]
> Some endpoints exist in both `app.py` (legacy) and `api_routes.py` (unified `/api/*` prefix). The legacy routes are user-scoped (require `user_id` query param), while the unified routes are session/admin-scoped. Counting deduplicated unique paths gives **76 unique URLs**, but there are **88 total route handler functions** across both files.
