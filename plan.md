# EventScout Polish Plan — Week of May 4th

## API Keys (all tested & confirmed working)
| Key | Variable | Status |
|-----|----------|--------|
| Gamma | GAMMA_API_KEY | ✅ 7,935 credits remaining |
| OpenRouter | OPENROUTER_API_KEY | ✅ 370 models available |
| Gemini | GEMINI_API_KEY | ✅ configured |

## Confirmed Gamma API Behavior
- Base: `POST https://public-api.gamma.app/v1.0/generations`
- Auth: `X-API-KEY` header
- Params: `textMode:"preserve"`, `cardSplit:"inputTextBreaks"`, `exportAs:"pptx"`, `themeId:"ash"`
- Slide breaks: `\n\n---\n\n` in inputText
- Async: poll `GET /v1.0/generations/{id}` every 5s until `status:"completed"`
- Returns: `gammaUrl` (web link) + `exportUrl` (PPTX, expires ~7 days)
- Cost: ~52 credits per PPTX deck

---

## PRIORITY 1 — Required for Event

### [P1-1] Fix AI Research Pipeline (research → score → pitch → deck → attach)
**Files:** `app.py`
- Replace hardcoded Lake B2B `RESEARCH_PROMPT_TEMPLATE` with `build_research_prompt(contact, user_profile)` — dynamic, uses user's actual products/services/target market/value props/pitch style/event context
- Add `GAMMA_API_KEY` config variable
- Replace `pipeline_step_deck()` Presenton logic with `pipeline_step_deck_gamma()`:
  - Convert `pitch_slides_content` list → markdown text with `---` separators
  - POST to Gamma API → poll → get `gammaUrl` + `exportUrl`
  - Download PPTX bytes from `exportUrl` (store in DB so no expiry dependency)
  - Store `gammaUrl` in `pipeline.gamma_deck_url`
- Update `pipeline_step_attach()`: store PPTX as `ContactFileDB` + append `gammaUrl` to contact `links[]`
- Pipeline now reaches `complete` status (not `complete_no_deck`)

### [P1-2] DB Schema Updates
**Files:** `database.py`
- Add `event_name VARCHAR(255) DEFAULT ''` to `ContactDB`
- Add `recipient_ids JSON DEFAULT NULL` to `AdminBroadcastDB`
- Add `gamma_deck_url TEXT DEFAULT NULL` to `ContactPipelineDB`
- Add `ContactListDB` model (id, name, admin_id, created_at)
- Add `ContactListMemberDB` model (id, list_id, contact_id)
- Add all migrations to `_init_schema()`

### [P1-3] Date-based + Event-based Contact Filtering (Admin Panel)
**Files:** `app.py`
- Add `date_filter: Optional[str]` query param to `GET /admin/contacts` (values: `today`, `yesterday`, `last_week`, `last_month`)
- Add `event_filter: Optional[str]` query param to `GET /admin/contacts`
- Add `page: int = Query(1)`, reduce default `limit` 200→50, return `total_count` + `has_more`
- Tag contacts with `event_name` from user profile when added via all add endpoints

### [P1-4] Broadcast with Team Member Selection
**Files:** `app.py`
- Update `POST /admin/broadcast` to accept `recipient_ids: Optional[str]` (comma-separated user UUIDs; omit = broadcast to all)
- Update `GET /broadcasts/active` to filter: only return broadcasts where `recipient_ids` is NULL or requesting user_id is in recipient_ids
- Store recipient_ids in `AdminBroadcastDB.recipient_ids`

### [P1-5] Select All + Add to Lists
**Files:** `app.py`
- `POST /admin/contact_lists` — create named list
- `GET /admin/contact_lists` — list all with contact counts
- `POST /admin/contact_lists/{list_id}/contacts` — bulk add contacts
- `GET /admin/contact_lists/{list_id}/contacts` — get contacts in list
- `DELETE /admin/contact_lists/{list_id}` — delete list

### [P1-6] Export with Research + Pitch Data
**Files:** `app.py`
- Update `GET /admin/export` CSV: add columns `research_summary`, `pitch_angle`, `pitch_email_subject`, `gamma_deck_url` (joined from `contact_pipelines`)
- Update `GET /export_contacts/` similarly

### [P1-7] Admin Panel Freezing Fix
**Files:** `app.py`
- Pagination on `/admin/contacts` and `/admin/users` (page + limit)
- Add `page` + `has_more` to responses

---

## PRIORITY 2 — Stretch Goals

### [P2-1] Frontend — Date/Event Filters UI
- Collapsible filter pills: All / Today / Yesterday / Last 7 Days / Last 30 Days
- Event filter dropdown from distinct event_name values
- Load More pagination at bottom of contacts list

### [P2-2] Frontend — Select All + Add to List UI
- Global "Select All" checkbox + per-contact checkboxes
- Floating action bar: Delete | Add to List | Export Selected
- List picker modal with "Create New List" option

### [P2-3] Frontend — Broadcast Recipient Picker
- Toggle: Send to All vs Send to Selected Members
- Checkboxes for each team member when "Selected" chosen

### [P2-4] Frontend — Pipeline Gamma Deck Link
- "Open Deck ↗" button when pipeline complete + gamma_deck_url exists
- Deck icon in pipeline step stepper

### [P2-5] Event Search (Lake Stream SOAP API)
- Dynamic event fetching in the event dropdown (post-event sprint)

---

## Implementation Sequence
```
Step 1: database.py — schema + models + migrations
Step 2: app.py — config + research prompt + Gamma deck pipeline
Step 3: app.py — event tagging on contact add
Step 4: app.py — admin contacts date/event filter + pagination
Step 5: app.py — export enhancement
Step 6: app.py — broadcast recipient_ids
Step 7: app.py — contact list endpoints
Step 8: mobile-frontend.html — all UI changes
```
