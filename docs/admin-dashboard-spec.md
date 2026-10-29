# Event Scout admin dashboard: recon and build spec

Card: t_d97ab745 (parent t_0308879e). Source row: TASKS.md line 724, Active, P1, captured 2026-09-14.
Row: team leads filter, rep performance side panel, lead search across all users, open full record, reassign to a rep, CSV export, system settings. Keep the team broadcast message and the all-contacts view. Admin is web only.
Context read: Efforts/Active/Event Scout 2026 Update/Event Scout 2026 Update.md (decision: admin exclusively web, users exclusively mobile, "admin UI becomes a dashboard"), Atlas/Products/Event Scout.md.

All code references are to `origin/master` at 51dd395 of Champ-Deep/event-scout. Line numbers drift fast: another branch (feat/hifi-scan-to-score) was editing `app.py` in the shared checkout while this was written. Always cite and diff against a committed ref, never the shared working tree.

## 1. Recon: what exists today

### 1.1 There is no admin app. Admin is one screen inside the mobile app

| Item | Fact | Where |
|------|------|-------|
| Frontend | One 6,592 line vanilla JS + Tailwind CDN file, no build step, no framework, no router | `mobile-frontend.html` |
| "Router" | `switchScreen(name)` toggles `#<name>Screen` divs and pushes `#<name>` to history | `mobile-frontend.html:1830` |
| Admin screen | `#adminScreen`: stats, export, search, broadcast, team list, webhook, activity feed, event files, pipeline monitor, backup | `mobile-frontend.html:971` to `1214` |
| Admin overlays | `#adminUserDrilldown` (per rep contact list, dupes, bulk select), `#adminContactDetail` (full record) | `:1216`, `:1294` |
| All-contacts view | The shared Contacts screen, which for admins loads every user's contacts | `loadContacts` `:1930`, admin branch `:1961` to `1966` |
| Data layer | Global `AppState`, `resilientFetch` (60 s, 1 retry) and `adminFetch` (120 s) | `:1525`, `:1545` |
| Auth | `POST /login/` returns `{user_id, name, email, is_admin}`. Every call sends `X-API-Key` (one shared key) and `?user_id=<admin uuid>` | `app.py:2472`, `verify_api_key` `:209`, `verify_admin` `:215` |
| Hosting | Vercel static, rewrite `/` to `mobile-frontend.html` | `vercel.json` |
| Backend | One 5,944 line FastAPI file, Postgres via `database.py`, in-memory FAISS per user rebuilt on startup (`app.py:2120` to `2136`) | `app.py`, `database.py` |
| Schema changes | No Alembic. `create_all` plus a list of `ALTER TABLE ... IF NOT EXISTS` strings | `database.py:429` to `438` |
| Tests | `tests/test_suite.py` is a live-target smoke script, not a unit suite. No CI. | `tests/` |

The PRD and Atlas note say "React 18 + TypeScript". That is stale. The repo is vanilla JS. The 14 Sep call question "React JS vs React Native" resolves to: neither, it is a single HTML file. This spec does not decide the mobile question.

### 1.2 Backend endpoints behind each feature (all `Depends(verify_admin)` unless noted)

| Concern | Endpoint | Line | Response shape (verified in code) |
|---------|----------|------|-----------------------------------|
| Overview stats, per rep stats | `GET /admin/dashboard` | 4313 | `{status, total_users, total_contacts, by_temperature{hot,warm,cold}, per_user[{user_id,name,email,contact_count,is_admin,hot_count,warm_count,cold_count,today_count,last_scan_at}], recent_contacts[10], webhook_configured, webhook_url_preview}`. 6 queries per user (N+1). |
| Activity feed | `GET /admin/activity?limit=50` | 3770 | `{status, activities[{type, contact_id, contact_name, company, user_id, user_name, source, timestamp, lead_temperature, lead_score}]}`. Works. |
| All-contacts view | `GET /admin/contacts` | 3666 | Query: `filter_user, contact_id, date_filter(today|yesterday|last_week|last_month), event_filter, limit=50, page=1`. Returns `{status,total,page,limit,has_more,distinct_events[],contacts[{id,user_id,user_name,name,email,phone,company_name,linkedin,event_name,lead_score,lead_temperature,lead_score_reasoning,lead_recommended_actions[],source,notes,admin_notes,links[],audio_notes_count,created_at,updated_at}]}`. |
| Search across all users | `GET /admin/search?q=&limit=50` | 5028 | Same contact fields minus pagination. Matches name, email, company_name, notes (`ilike`). No filters, no paging. |
| Team broadcast | `POST /admin/broadcast?message=&priority=normal|urgent&recipient_ids=a,b` | 5092 | `{status,broadcast_id,message,recipient_count}`. Message travels in the query string. |
| Broadcast list | `GET /broadcasts/active?user_id=` (API key only) | 5129 | `{status,broadcasts[{id,message,priority,admin_name,recipient_ids,created_at}]}`, newest 20, targeted ones hidden from non-recipients. |
| Broadcast dismiss | `POST /broadcasts/{id}/dismiss` | 5168 | `{status,message}`. |
| Broadcast recipient picker | `GET /admin/users` | 3594 | `{status,users[{user_id,name,email,is_admin,contact_count,created_at}],total}`. **Danger:** the same GET deletes a user with `?action=delete&target_user_id=`. The new UI must never call `/admin/users` with `action`. |
| Full record | `GET /admin/contacts?contact_id=` (exists, unused by legacy UI, which fetches 500 rows and filters client side `:5337`) | 3666 | One contact, fields above. Not returned: `lead_score_breakdown`, `photo_base64`. |
| Record pipeline | `GET /contact/{id}/pipeline?user_id=<owner>` (API key only) | 5902 | `{id,status,current_step,error_message,research_summary,research_data,score_completed,pitch_angle,pitch_email_subject,pitch_email_body,pitch_slides_content,deck_file_id,gamma_deck_url,started_at,completed_at}` or `{status:"none"}`. |
| Record files | `GET /contact/{id}/files?user_id=` (API key only) | 5381 | `{status,total,files[{id,filename,file_type,file_size,file_size_mb,description,category,download_count,created_at}]}`. Download: `GET /contact/files/download/{file_id}` (5420). |
| Admin note | `PUT /admin/contact/{id}/note?note=` | 5197 | `{status,contact_id,admin_notes,message}`. Note travels in the query string. |
| CSV export | `GET /admin/export?format=csv` | 5232 | All contacts, every user, no filters. Columns: Name, Email, Phone, LinkedIn, Company, Event, Notes, Admin Notes, Lead Score, Lead Temperature, Source, Scanned By, Created At, Research Summary, Pitch Angle, Pitch Email Subject, Gamma Deck URL. |
| Webhook test | `POST /admin/webhook/test?webhook_url=&save=` | 4254 | `save=true` only mutates an in-memory global (`WEBHOOK_URL`, `app.py:47`). Lost on restart. |
| Auto pipeline | `POST /admin/pipeline/settings?auto_enabled=` | 5683 | In-memory global (`AUTO_PIPELINE_ENABLED`, `:57`). Lost on restart. Returns `{auto_pipeline_enabled, current_settings{auto_pipeline_enabled,gamma_configured,openrouter_configured}}`. |
| Backup | `POST /admin/backup` (5495), `GET /admin/backup/status` (5507) | | status: `{backup_configured,last_backup,primary_counts,backup_counts,in_sync}`. |

### 1.3 What the row asks for that the backend does not have

| Row feature | Gap | Consequence |
|-------------|-----|-------------|
| Team leads filter | No team lead concept. `UserDB` has `id,name,email,password_hash,is_admin,created_at` only (`database.py:62`). | Needs a schema field, a read endpoint and a write endpoint. |
| Rep performance side panel | Per rep counts exist inside `/admin/dashboard` only, N+1, no time series or average score. | Needs one grouped read endpoint and one per rep endpoint. |
| Lead search plus filter plus paging | `/admin/search` has no filters or paging. `/admin/contacts` has filters but no text query, team lead, or temperature. | Needs one new list endpoint. |
| CSV of the current result set | `/admin/export` ignores filters. | Needs a filtered export that shares the list's WHERE clause. |
| Reassign to a rep | No endpoint. `PUT /contact/{id}` is owner scoped and cannot change `user_id`. FAISS is keyed per user, so changing the owner also needs both indexes rebuilt. | Needs a new endpoint. |
| System settings persisted and permission gated | Webhook URL and auto pipeline are process memory. There is one admin tier. | Needs a settings table and an owner/read-only split. |

Per the feature cards, these gaps would block the three feature workers with `kind=capability`. They are therefore specified below (section 5) and assigned to one backend card that the feature cards depend on.

## 2. Decisions

| # | Decision | Reason |
|---|----------|--------|
| D1 | New web app in `admin-web/` at the repo root: Vite, React, TypeScript, Tailwind, shadcn components sourced from 21st.dev, react-router, TanStack Query. Separate Vercel project (root directory `admin-web`). `mobile-frontend.html` is not touched by any card in this chain. | Admin is web only (row). A real router and data layer are needed. 21st.dev components are a standing requirement for Deep's front-end work. Editing the 6,592 line single file with four parallel workers would be a guaranteed merge collision. Workers must resolve current package versions through Context7 and the shadcn CLI before writing code; this spec pins none. |
| D2 | The legacy mobile admin screen stays as is until cutover. Removing it is a separate later card. | Zero regression risk. It also keeps everything not in the row (user merge and delete, duplicates, contact lists, event files, pipeline monitor) reachable. |
| D3 | All new backend code goes in a new module `admin_api.py` as a router factory, wired into `app.py` with a few lines. Legacy endpoints stay byte identical. | `app.py` is a 5,944 line collision hotspot and other branches are editing it now. Byte identical legacy endpoints protect the preserved views and the legacy admin screen. |
| D4 | Team lead = a nullable `users.team_lead_id` pointing at another user. A "team lead" is any user that some rep points at. No separate role. | Smallest model that supports the filter. The real mapping must be entered by Deep or an admin via the rep panel. Until then every rep has no lead and the filter shows only "No team lead". |
| D5 | Settings write access = email in `ADMIN_EMAILS` (`app.py:49`, owner tier). Any other `is_admin` user gets read only. | Uses the only two tiers that already exist. Makes the read only state real, not mocked. |
| D6 | Filter, search, sort and page state live in the URL query string on `/leads`. Detail and rep panel are nested routes rendered over the list. | URL shareable and refresh safe (card requirement). The list stays mounted under the drawer, so filter, search and scroll survive open and close. |
| D7 | Do not carry forward: the full activity feed with per row delete, user merge and delete, duplicate finder, contact lists, event files, pipeline monitor. | Not in the row. Still available in the legacy screen. |

If Deep wants D1 changed (for example stay inside `mobile-frontend.html`), only section 4 changes. Sections 3, 5 and 6 hold.

## 3. Information architecture and route map

Hash free `BrowserRouter`. `admin-web/vercel.json` rewrites `/(.*)` to `/index.html`. All routes except `/login` are behind `RequireAdmin` (session user with `is_admin: true`, else redirect to `/login`).

| Route | Screen | Feature | Owner |
|-------|--------|---------|-------|
| `/login` | Login form | Auth via `POST /login/`; reject non admin | Shell |
| `/` | Overview: 4 stat cards (users, contacts, hot, warm) from `/admin/dashboard`, recent activity list (10) from `/admin/activity?limit=10` | Shell landing | Shell |
| `/contacts` | **All-contacts view (preserved)** | Existing behaviour, section 6 | Shell |
| `/broadcast` | **Team broadcast message (preserved)** | Existing behaviour, section 6 | Shell |
| `/leads` | Lead list: team lead filter, global search, other filters, paging, CSV export button | Team leads filter, lead search, CSV export | Feature A |
| `/leads/contact/:contactId` | Full record drawer over the list | Open full record | Feature B |
| `/leads/rep/:userId` | Rep performance side panel over the list | Rep performance | Feature B |
| `/settings` | Settings: Integrations, Pipeline, Backup | System settings | Feature C |
| `*` | Not found | | Shell |

Sidebar order: Overview, Leads, All contacts, Broadcast, Settings. Header: current user, sign out, environment badge derived from `VITE_API_BASE_URL` (`localhost` shows "local", else "production").

Row link contract (so Feature A and Feature B never edit each other's files):
- Feature A renders every lead row name as `<Link to={"/leads/contact/" + id + location.search}>` and every rep cell as `<Link to={"/leads/rep/" + userId + location.search}>`.
- Shell's `LeadsLayout` renders `<LeadsPage/>` then `<Outlet/>`. The outlet hosts Feature B's `RecordDrawer` and `RepPanel`. Closing navigates to `"/leads" + location.search`.
- The `/contacts` view row click navigates to `/leads/contact/:id` (the one permitted deviation from the legacy overlay; same data, new host).

## 4. Component inventory and exact file paths

New tree. Paths are relative to the repo root. Nothing below exists yet.

```
admin-web/
  package.json  vite.config.ts  tsconfig.json  tsconfig.app.json  index.html
  components.json  tailwind config  postcss config  vercel.json  .env.example      [Shell]
  src/
    main.tsx                         providers: QueryClient, Router, Auth       [Shell]
    App.tsx                          route table, lazy imports, RequireAdmin     [Shell]
    lib/api.ts                       fetch wrapper (see 4.1)                     [Shell]
    lib/auth.tsx                     login, session in sessionStorage, logout    [Shell]
    lib/types.ts                     shared shapes (see 4.2)                     [Shell]
    components/ui/*                  21st.dev / shadcn primitives (added by CLI) [Shell creates; features may ADD new files only]
    layout/AppShell.tsx              sidebar, header, env badge, skip link, <Outlet/>
    layout/LeadsLayout.tsx           <LeadsPage/> + <Outlet/>                    [Shell]
    features/overview/OverviewPage.tsx                                          [Shell]
    features/contacts/ContactsPage.tsx    preserved all-contacts view            [Shell]
    features/broadcast/BroadcastPage.tsx  preserved broadcast view               [Shell]
    features/leads/index.ts               exports LeadsPage                      [Shell creates placeholder, Feature A fills]
    features/leads/LeadsPage.tsx          list, filter bar, search, table, pager [Feature A]
    features/leads/TeamLeadFilter.tsx                                           [Feature A]
    features/leads/LeadSearchBox.tsx      debounced input                        [Feature A]
    features/leads/ExportButton.tsx                                             [Feature A]
    features/leads/useLeadsQuery.ts       URL state <-> query                    [Feature A]
    features/leads/types.ts                                                     [Feature A]
    features/lead-detail/index.ts         exports RecordDrawer, RepPanel         [Shell creates placeholder, Feature B fills]
    features/lead-detail/RecordDrawer.tsx full record, notes, pipeline, files    [Feature B]
    features/lead-detail/ReassignDialog.tsx confirm, result, refresh             [Feature B]
    features/lead-detail/RepPanel.tsx     metrics, daily chart, team lead select [Feature B]
    features/lead-detail/api.ts  types.ts                                       [Feature B]
    features/settings/index.ts            exports SettingsPage                   [Shell creates placeholder, Feature C fills]
    features/settings/SettingsPage.tsx    sections, dirty state, save            [Feature C]
    features/settings/IntegrationsSection.tsx PipelineSection.tsx BackupSection.tsx   [Feature C]
    features/settings/api.ts  types.ts                                          [Feature C]
```

Backend (repo root):

```
admin_api.py                   new router factory, all new endpoints             [Backend]
database.py                    UserDB.team_lead_id, AppSettingDB, migration line [Backend]
app.py                         ONLY: import + include_router, startup settings load, CORS expose_headers   [Backend]
tests/test_admin_api.py        new                                               [Backend]
```

### 4.1 `lib/api.ts` contract (Shell)
- Base URL `import.meta.env.VITE_API_BASE_URL`. API key `import.meta.env.VITE_API_KEY`. Both go in `.env.example` as empty names, never as values.
- `api<T>(path, {method, query, body, signal})` adds `X-API-Key` and `user_id` (from session) to every call. Non 2xx throws `ApiError{status, detail}` using FastAPI's `detail`. JSON body for new endpoints. Query string for legacy endpoints.
- `apiBlob(path, query)` for CSV and file downloads.
- 60 s timeout, abort on unmount.

### 4.2 `lib/types.ts` (Shell, frozen after the shell card merges)
`SessionUser{user_id,name,email,is_admin}`, `AdminContact` (the `/admin/contacts` item shape in 1.2), `Rep` (section 5.2 item), `DashboardStats` (the `/admin/dashboard` shape), `Activity` (the `/admin/activity` item), `Broadcast`. Features import these and never edit the file. A needed change goes to the shell owner as a card comment.

## 5. Data source per feature

Auth on every call below: header `X-API-Key`, query `user_id=<admin uuid>`, enforced by `verify_admin` (admin only) unless marked "key only". New endpoints are in `admin_api.py` and use `verify_admin`.

| Feature | Route | Component | Data source |
|---------|-------|-----------|-------------|
| Team leads filter | `/leads?team_lead=<uuid|none>` | `TeamLeadFilter.tsx` | Options from `GET /admin/reps` (5.2) `team_leads[]`. Filter applied server side by `GET /admin/leads` (5.1). |
| Lead search across all users | `/leads?q=` | `LeadSearchBox.tsx`, `LeadsPage.tsx` | `GET /admin/leads?q=` (5.1). Searches every user. 300 ms debounce, loading, empty and error states. |
| Open full record | `/leads/contact/:contactId` | `RecordDrawer.tsx` | `GET /admin/leads/{contact_id}` (5.4), plus `GET /contact/{id}/pipeline?user_id=<owner>` and `GET /contact/{id}/files?user_id=<owner>` (key only, legacy) and `PUT /admin/contact/{id}/note?note=` (legacy) |
| Reassign to a rep | same route, `ReassignDialog.tsx` | `ReassignDialog.tsx` | Rep options `GET /admin/reps`. Action `POST /admin/leads/{contact_id}/reassign` (5.5). After success invalidate queries `leads`, `reps`, `lead:<id>`. |
| Rep performance side panel | `/leads/rep/:userId` | `RepPanel.tsx` | `GET /admin/reps/{user_id}/performance` (5.3). Team lead select calls `PUT /admin/users/{user_id}/team_lead` (5.6). |
| CSV export | `/leads` | `ExportButton.tsx` | `GET /admin/leads/export` (5.7) with the same query string as the list. Client downloads the blob. No client side CSV building. |
| System settings | `/settings` | `SettingsPage.tsx` and sections | `GET/PUT /admin/settings` (5.8). Webhook test: legacy `POST /admin/webhook/test?webhook_url=` (no `save`). Backup: legacy `GET /admin/backup/status`, `POST /admin/backup`. Auto pipeline read: `GET /admin/settings`. |

### 5.1 `GET /admin/leads`
Query: `q` (min 1 char, matches name, email, company_name, notes, `ilike`, same four columns as `/admin/search`), `team_lead` (user uuid, or the literal `none` for reps without a lead), `rep` (user uuid), `temperature` (`hot|warm|cold|unscored`), `event`, `date_filter` (same four values as `/admin/contacts`), `sort` (`created_desc` default, `score_desc`, `name_asc`), `page=1`, `limit=50` (max 200).
`team_lead=<uuid>` matches contacts owned by users whose `team_lead_id` is that uuid, plus the lead's own contacts.
Response:
```json
{ "status": "success", "total": 0, "page": 1, "limit": 50, "has_more": false,
  "contacts": [ { "id": "", "user_id": "", "user_name": "", "team_lead_id": null, "team_lead_name": null,
    "name": "", "company_name": "", "email": "", "phone": "", "event_name": "",
    "lead_score": null, "lead_temperature": null, "source": "", "created_at": "" } ] }
```
On screen columns and CSV columns, in this order: Name, Company, Email, Phone, Rep, Team lead, Event, Score, Temperature, Source, Created. Empty values show as empty, not "N/A".

### 5.2 `GET /admin/reps`
Single grouped query. Response:
```json
{ "status": "success",
  "reps": [ { "user_id": "", "name": "", "email": "", "is_admin": false, "team_lead_id": null, "team_lead_name": null,
    "contact_count": 0, "hot_count": 0, "warm_count": 0, "cold_count": 0, "today_count": 0, "last_scan_at": null } ],
  "team_leads": [ { "user_id": "", "name": "", "rep_count": 0 } ] }
```

### 5.3 `GET /admin/reps/{user_id}/performance?days=14`
`days` 1 to 90. Response: `{status, rep:<5.2 item>, avg_score: number|null, daily:[{date:"YYYY-MM-DD", count}] (every day in range, zero filled, UTC), by_event:[{event_name, count}] (top 5)}`. 404 if user unknown.

### 5.4 `GET /admin/leads/{contact_id}`
Response: `{status, contact:{...every field in the 1.2 `/admin/contacts` item, plus lead_score_breakdown, photo_base64 (nullable), linkedin_source, team_lead_id, team_lead_name}}`. 404 if unknown.

### 5.5 `POST /admin/leads/{contact_id}/reassign`
Body `{"to_user_id": "<uuid>"}`. Steps: validate contact and target user exist (404), reject same owner (400, "Already owned by this rep"), update `contacts.user_id` and `contact_pipelines.user_id` for that contact in one transaction, commit, then rebuild FAISS for both users from the DB (same builder as `app.py:2120` to `2136`). Contact files and QR are keyed by contact id, no change.
Response: `{status:"success", contact_id, from_user_id, from_name, to_user_id, to_name, faiss_synced:true}`. If the index rebuild throws after commit: still 200 with `faiss_synced:false` and a logged line; the DB is the source of truth and the index is rebuilt on startup. The UI shows a warning toast in that case. A failed DB step returns an error and changes nothing; the UI must show it, never swallow it.
shortcut: no audit table, one `[REASSIGN]` log line with admin id, contact id, from, to; add an audit table when Deep needs history.

### 5.6 `PUT /admin/users/{user_id}/team_lead`
Body `{"team_lead_id": "<uuid>|null"}`. 400 if equal to `user_id`, 404 if either user is unknown. Response `{status, user_id, team_lead_id, team_lead_name}`.

### 5.7 `GET /admin/leads/export`
Same query params as 5.1 except `page` and `limit`; `sort` is honoured so row order matches the screen. Built from the same shared WHERE builder as 5.1 (one function, `_lead_conditions(...)`), so the two cannot diverge. Returns `text/csv`, UTF-8 with BOM, header row = the 11 columns in 5.1, cells starting with `=`, `+`, `-`, `@` prefixed with `'` (spreadsheet formula injection). Headers: `Content-Disposition: attachment; filename=event_scout_leads_<YYYYMMDD>.csv` and `X-Total-Count: <rows>`. Refuses more than 20,000 rows with 422 `"Too many rows, narrow the filter"`. `app.py` CORS middleware gets `expose_headers=["X-Total-Count","Content-Disposition"]` so the browser can read them.
shortcut: built in memory, fine to 20,000 rows; stream if that cap is raised. Local CSV guard duplicates `safe_cell` in the unmerged branch feat/harbinger-csv-export; switch to `export_csv.safe_cell` once that branch lands.

### 5.8 `GET /admin/settings` and `PUT /admin/settings`
Table `app_settings(key text primary key, value jsonb not null, updated_at timestamptz, updated_by uuid)`. Keys: `webhook_url`, `auto_pipeline_enabled`. Startup loads them and overrides the env defaults in the `WEBHOOK_URL` and `AUTO_PIPELINE_ENABLED` globals; the PUT updates the table and those globals.
GET response:
```json
{ "status": "success", "can_write": true,
  "settings": { "webhook_url": "", "auto_pipeline_enabled": false },
  "info": { "gamma_configured": false, "openrouter_configured": false, "backup_configured": false } }
```
PUT body: any subset of `{webhook_url, auto_pipeline_enabled}`. Rules: `webhook_url` is empty (clears it) or an `https://` URL with a host, at most 500 chars, else 422 `{detail}`; `auto_pipeline_enabled` must be a boolean. 403 `"Owner access required"` when the caller's email is not in `ADMIN_EMAILS`. Response is the GET shape after the change. `can_write` is false for non owner admins and the UI renders every control disabled with a visible "Read only" notice.

## 6. Preserved unchanged (acceptance checks for the shell card)

Legacy endpoints that must stay byte identical in `app.py` (backend card must prove it with a diff of those function bodies against `origin/master`): `/admin/contacts`, `/admin/dashboard`, `/admin/activity`, `/admin/search`, `/admin/broadcast`, `/broadcasts/active`, `/broadcasts/{id}/dismiss`, `/admin/users` (all actions), `/admin/export`, `/admin/webhook/test`, `/admin/pipeline/settings`, `/admin/backup`, `/admin/backup/status`, `/admin/contact/{id}/note`.

### 6.1 Team broadcast message (`/broadcast`, `BroadcastPage.tsx`)
Reproduce `sendBroadcast`, `toggleBroadcastRecipients`, `loadBroadcastTeamPicker`, `loadBroadcastHistory`, `dismissBroadcast` (`mobile-frontend.html:5081` to `5168`):
- Message textarea, priority select (`normal`, `urgent`), target radio `All members` / `Selected members`.
- Selected: checkbox list from `GET /admin/users`, excluding the current admin, each row `name (N contacts)`; at least one required.
- Send: `POST /admin/broadcast` with `message`, `priority`, and `recipient_ids` (comma joined) only when Selected. Empty message is rejected client side. Success clears the form, resets to All, reloads history. Error shows the server `detail`.
- Active broadcasts list from `GET /broadcasts/active`: message, relative time, urgent styling, dismiss button calling `POST /broadcasts/{id}/dismiss`, then reload.
- Permissions: admin only.
Not preserved on purpose: the red or blue Tailwind markup. Behaviour, data and permissions are.

### 6.2 All-contacts view (`/contacts`, `ContactsPage.tsx`)
Reproduce the admin branch of `loadContacts` (`:1930`, `:1961` to `1966`):
- `GET /admin/contacts?limit=200[&date_filter=today|yesterday|last_week|last_month]`. All users' contacts, newest first.
- Each row: name, company, owner badge (`user_name`), temperature and score, source, created time.
- Date filter chips: All, Today, Yesterday, Last 7d, Last 30d. A count label shows `N contacts found` when a filter is active, using `total`.
- Row click opens `/leads/contact/:id` (see section 3).
- Permissions: admin only.

## 7. File ownership split

One owner per path. A card edits only paths it owns. Anything else is read only for that card.

| Card | Owns (create or edit) | Must not touch |
|------|----------------------|----------------|
| Backend (new card, section 9) | `admin_api.py`, `database.py`, `tests/test_admin_api.py`, and in `app.py` only: the `admin_api` import and `include_router`, the startup settings load, `expose_headers` on the CORS middleware | `admin-web/**`, `mobile-frontend.html`, every legacy endpoint body in `app.py` |
| Shell t_f2926c75 | everything under `admin-web/` not listed for a feature below, including `components/ui/*` and the three placeholder `index.ts` files | `features/leads/*` except `index.ts`, `features/lead-detail/*` except `index.ts`, `features/settings/*` except `index.ts`; all Python |
| Feature A t_55170523 | `admin-web/src/features/leads/*` (except `index.ts`, whose export names are fixed by the shell) | all else |
| Feature B t_a2156696 | `admin-web/src/features/lead-detail/*` (except `index.ts`) | all else |
| Feature C t_fd3dc3b9 | `admin-web/src/features/settings/*` (except `index.ts`) | all else |
| Verification t_918607bd | nothing in the repo; evidence goes on its card | all code |

Rules that make the split hold:
1. Order is enforced by the card graph: Spec, then Shell and Backend in parallel, then A, B, C in parallel. Shell files are frozen once its card is done. A feature worker who needs a shell change posts a card comment to the shell owner instead of editing.
2. Shared shapes live in `lib/types.ts` (Shell). Feature specific shapes live in the feature's own `types.ts`.
3. New shadcn primitives a feature needs are added as new files in `components/ui/` through the CLI; existing files there are never edited.
4. `admin-web/package.json` and the lockfile are Shell only. A feature that needs a new dependency asks the shell owner by comment. Prefer none.
5. Route table `App.tsx` is Shell only and already points at `features/*/index.ts`. A feature never edits it.
6. Backend rule: if the backend card needs a response shape different from section 5, it updates this spec in the same commit and comments on the three feature cards. Shapes in section 5 are the contract.

## 8. Verification hooks for the later cards

- Backend: pytest against a throwaway Postgres (the repo has none; the card sets one up) covering: filter composition, `export` row count equals list `total` for the same query, reassign moves ownership and pipeline owner, reassign to same owner is 400, settings PUT 403 for a non owner, 422 for `http://`, legacy endpoint bodies unchanged vs `origin/master`.
- Shell: `npm run build`, every route in section 3 resolves, screenshots of the shell plus `/contacts` and `/broadcast`, keyboard Tab reaches the nav, visible focus.
- Features: evidence items are in each card's acceptance criteria. Feature A evidence must quote the query string and the CSV row count.

## 9. Follow-up card created from this spec

Backend card "Event Scout admin API: team lead, leads list and export, reps, reassign, settings" (assignee default). It is a parent of t_55170523, t_a2156696 and t_fd3dc3b9. The Shell card t_f2926c75 does not depend on it (it only uses legacy endpoints).

## 10. Risks Deep should know about

1. **Shared API key in the browser.** Auth is one static `X-API-Key` plus an admin `user_id` query parameter. Anyone holding both is an admin. The key is hardcoded in the committed `mobile-frontend.html`, so it is in git history and in every served copy. The new app will read it from a build env var but it still ships to the browser. Recommend a separate card: rotate the key and move admin calls to a signed session token. Out of scope here.
2. **Destructive GET.** `GET /admin/users?action=delete` deletes a user and their data. Any link prefetch or crawler with the key can trigger it. Left alone (legacy), not exposed in the new UI.
3. **Notes and broadcast text in URLs.** `PUT /admin/contact/{id}/note?note=` and `POST /admin/broadcast?message=` put free text in the query string (logged by proxies). Preserved as is for zero behaviour change.
4. **Feature branch in the shared checkout.** `~/repos/event-scout` was on `feat/hifi-scan-to-score` with an uncommitted `app.py`. The backend worker must work in its own worktree from `origin/master`, never in that checkout.
5. **Unmerged CSV branch.** `origin/feat/harbinger-csv-export` also edits both existing export routes in `app.py`. This spec avoids the conflict by leaving them untouched. It has not been tested or reviewed here.
6. **Not verified.** Nothing in this spec was run against a live database. Response shapes come from reading the code at 51dd395. The 21st.dev and shadcn install path, and current versions of Vite, React, react-router and TanStack Query, were not checked and must be resolved by the Shell worker through Context7.
