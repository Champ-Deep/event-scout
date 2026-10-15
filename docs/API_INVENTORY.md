# EventScout API inventory

Generated 2026-10-14T06:48:20+00:00 by `scripts/api_inventory.py` from the code — re-run it after API changes. Machine-readable: `docs/api-inventory.json`.

## Summary

| | Total | Used | Unused |
|---|---|---|---|
| **All backend routes** | 209 | 144 | 65 |
| Admin APIs | 100 | 67 | 33 |
| User APIs | 87 | 58 | 29 |
| Public APIs | 22 | 19 | 3 |
| of the unused: legacy `app.py` routes (old web frontend) | 60 | — | 60 |

| | Count |
|---|---|
| Used by the mobile app | 69 |
| Used by the admin console | 92 |
| Used from outside (webhooks, public links, health check) | 8 |
| Mobile client functions (`api.ts`) | 79 — used 66, unused 13 |
| Broken calls | 2 |
| Fake / dummy APIs | 2 — fake 1, partly fake 0, stub 1, bug 0, suspected 0 (2 still called by a client) |
| New APIs created | 14 |
| Were unused, now used in the mobile app | 7 |

Legend — ⚠️ fake / dummy · 🆕 new API · ✅ newly used in the app · legacy = old `app.py` route (pre-`/api`)

## 1. Fake / dummy APIs

| Kind | Method | Path | Access | Called by | What is fake | Where |
|---|---|---|---|---|---|---|
| fake | GET | `/api/sync/state` | user | mobile (api.getSyncState) | Marked "Mock sync state": always 0 pending / 0 failed and "last synced" = now. Not used by the apps any more (the phone reads its own outbox). | `api_routes.py:5247` |
| stub | POST | `/api/contact/{contact_id}/enrich` | user | admin (ContactDetailDrawer.tsx) | Always 501 "not connected to a data provider" (no enrichment provider is set up). | `api_routes.py:4853` |

Were fake / dummy — now real:

| Method | Path | Now |
|---|---|---|
| GET | `/api/admin/dashboard` | Only real counts (no 1248 / "14 reps" / "100% sync"); levels, 7-day chart, events, pipelines, workspace totals. |
| GET | `/api/admin/activity` | Real timeline with real names: contacts added, AI scores, pipelines, announcements, files, deletions, new accounts. |
| GET | `/api/admin/users` | Real contact counts, added today and last activity (was 35×n and "now"). |
| POST | `/api/admin/users` | Real hashed password the person signs in with (was a placeholder hash); duplicate email → 409. |
| GET | `/api/admin/contacts/duplicates` | Real email / phone / name+company matches per owner; no invented "dup-temp" pair. |
| GET | `/api/admin/contact_lists/{list_id}/contacts` | Empty list returns [] (was 3 random contacts). |
| POST | `/api/admin/contact_lists/{list_id}/contacts` | Skips unknown ids and contacts already in the list; reports counts. |
| GET | `/api/admin/pipelines` | Real pipeline runs (ContactPipelineDB) with their research, pitch, slides and deck link. |
| POST | `/api/contact/{contact_id}/pipeline/run` | Starts the real research → score → pitch → deck → attach pipeline. |
| POST | `/api/contact/{contact_id}/pipeline/retry` | Clears failed runs and starts the real pipeline again. |
| GET | `/api/contact/{contact_id}/pipeline` | The contact's latest real run (status none if never run). |
| POST | `/api/admin/backup` | Real backup: to BACKUP_DATABASE_URL, else a consistent SQLite snapshot in backend/backups/. |
| GET | `/api/admin/backup/status` | Real target, last backup and snapshots on disk. |
| POST | `/api/admin/webhook/test` | Really POSTs a test event to WEBHOOK_URL and reports status / latency / response. |
| GET | `/api/admin/export` | Real CSV file download with the Title column fixed and the owner added. |
| GET | `/api/models` | The OpenRouter models this server really uses, and what for. |

Demo data in the database (returned by real APIs):

- Switched off on 2026-09-30 (is_active = 0, not deleted; backup: apps/backend/eventscout.db.bak-2026-09-30-before-demo-retire): 3 seeded announcements and 4 seeded placeholder files. /api/broadcasts/active and /api/files/list now return only real rows.
- Still in the database from the old demo seeder (SEED_DEMO_DATA, now off): 20 demo contacts owned by the admin account (+1 (555) phone numbers, event "WHX 2026") and 3 demo contact lists. They show in the admin console and in that account's app.

## 2. Broken

| Client | Caller | Method | Path | Called by the app? | Problem |
|---|---|---|---|---|---|
| mobile | `api.joinInvite (api.ts:886)` | POST | `/auth/join/${inviteId}` | no | No backend route with this method and path (would 404 / 405). |
| admin | `apps/admin/src/lib/api.ts` | GET | `/api/admin/export${qs(f)}` | yes | No backend route with this method and path (would 404 / 405). |

## 3. New APIs created

| Method | Path | Purpose | Used by |
|---|---|---|---|
| POST | `/api/exhibitors` | Add a booth to an event from the app (no duplicates; can mark it visited) | mobile, admin |
| PATCH | `/api/exhibitors/{exhibitor_id}` | Edit a booth you added (admins: any) | mobile, admin |
| DELETE | `/api/exhibitors/{exhibitor_id}` | Remove a booth you added (admins: any) | mobile, admin |
| PUT | `/api/exhibitors/{exhibitor_id}/visit` | Mark a booth as visited by you (counts in "Booths met") | mobile |
| DELETE | `/api/exhibitors/{exhibitor_id}/visit` | Un-mark a booth as visited | mobile |
| GET | `/api/files/{file_id}/download` | Content of a team / event file (open or save it in the app) | mobile, admin |
| GET | `/api/contact/files/{file_id}/download` | Content of a file attached to one of your contacts | mobile, admin |
| POST | `/api/contact/{contact_id}/files` | Attach a file to one of your contacts (Contact details → Files → Attach) | mobile, admin |
| DELETE | `/api/contact/files/{file_id}` | Remove a file from one of your contacts | mobile, admin |
| PATCH | `/api/admin/users/{user_id}` | Admin: rename an account or change its role | admin |
| DELETE | `/api/admin/users/{user_id}` | Admin: delete a rep's account and data | admin |
| POST | `/api/admin/contacts/duplicates/dismiss` | Admin: mark two contacts as not the same person | admin |
| PATCH | `/api/admin/contact_lists/{list_id}` | Admin: rename a list | admin |
| DELETE | `/api/admin/contact_lists/{list_id}/contacts/{contact_id}` | Admin: remove a contact from a list | admin |

## 4. Were unused — now used in the mobile app

| Method | Path | Where in the app |
|---|---|---|
| GET | `/api/broadcasts/active` | Feed → team announcements banner (only the ones sent to everyone / to you) |
| GET | `/api/files/list` | Me → Team files; Event page → Event files (demo placeholders hidden) |
| GET | `/api/events/{event_id}/seats` | Event page → Your team here (teammates + scan counts) |
| GET | `/api/exhibitors/categories` | Event page → Add booth: category suggestions (this event's first) |
| PATCH | `/api/exhibitors/{exhibitor_id}` | Event page → a booth you added → Edit |
| GET | `/api/contact/{contact_id}/files` | Contact details → Files card (attach, open, save, remove) |
| POST | `/auth/refresh` | Background: renews the sign-in token when it expires within a day |

## 5. Mobile client functions the app never calls

| Function | Calls | Why unused | Line |
|---|---|---|---|
| `api.getLiveEvents` | GET `/api/live` | Admin-only route: a rep would be refused. | `apps/mobile/src/lib/api.ts:638` |
| `api.getActivity` | GET `/api/admin/activity` | Admin-only route: a rep would be refused. | `apps/mobile/src/lib/api.ts:643` |
| `api.getAIModels` | GET `/api/models` | Admin-only route: a rep would be refused. | `apps/mobile/src/lib/api.ts:648` |
| `api.getActiveBroadcasts` | GET `/api/broadcasts/active` | Not called by any screen. | `apps/mobile/src/lib/api.ts:722` |
| `api.enrichContact` | POST `/api/contact/${contactId}/enrich` | Stub: Always 501 "not connected to a data provider" (no enrichment provider is set up). | `apps/mobile/src/lib/api.ts:791` |
| `api.runContactPipeline` | POST `/api/contact/${contactId}/pipeline/run` | Admin-only route: a rep would be refused. | `apps/mobile/src/lib/api.ts:818` |
| `api.retryContactPipeline` | POST `/api/contact/${contactId}/pipeline/retry` | Admin-only route: a rep would be refused. | `apps/mobile/src/lib/api.ts:822` |
| `api.getContactPipeline` | GET `/api/contact/${contactId}/pipeline` | Admin-only route: a rep would be refused. | `apps/mobile/src/lib/api.ts:826` |
| `api.joinInvite` | POST `/auth/join/${inviteId}` | Broken: the route does not exist. | `apps/mobile/src/lib/api.ts:886` |
| `api.addContactLegacy` | POST `/add_contact/` | Old duplicate of an API the app already uses. | `apps/mobile/src/lib/api.ts:950` |
| `api.searchContactsLegacy` | POST `/search/` | Old duplicate of an API the app already uses. | `apps/mobile/src/lib/api.ts:957` |
| `api.getUserProfile` | GET `/user/profile/` | Old duplicate of an API the app already uses. | `apps/mobile/src/lib/api.ts:1214` |
| `api.updateUserProfile` | PUT `/user/profile/` | Old duplicate of an API the app already uses. | `apps/mobile/src/lib/api.ts:1218` |

## 6. All APIs by access

### 6.1 Admin APIs (admin sign-in) — 100 total, 67 used, 33 unused

**Used (67)**

| Method | Path | Used by | Flags | Notes | Handler |
|---|---|---|---|---|---|
| GET | `/api/admin/activity` | admin (api.ts) |  |  | `api_routes.py:2797` get_admin_activity |
| GET | `/api/admin/activity/calendar` | admin (api.ts) |  |  | `api_routes.py:2813` get_admin_activity_calendar |
| POST | `/api/admin/backup` | admin (api.ts) |  |  | `api_routes.py:5056` trigger_admin_backup |
| GET | `/api/admin/backup/snapshots/{file}` | admin (api.ts) |  |  | `api_routes.py:5144` download_backup_snapshot |
| GET | `/api/admin/backup/status` | admin (api.ts) |  |  | `api_routes.py:5117` get_backup_status |
| POST | `/api/admin/broadcast` | admin (api.ts) |  |  | `api_routes.py:3460` send_broadcast |
| GET | `/api/admin/broadcasts/history` | admin (api.ts) |  |  | `api_routes.py:3534` broadcast_history |
| PUT | `/api/admin/contact/{contact_id}/note` | admin (ContactDetailDrawer.tsx, api.ts) |  |  | `api_routes.py:4872` update_contact_admin_note |
| GET | `/api/admin/contact_lists` | admin (api.ts) |  |  | `api_routes.py:3764` get_contact_lists |
| POST | `/api/admin/contact_lists` | admin (api.ts) |  |  | `api_routes.py:3776` create_contact_list |
| DELETE | `/api/admin/contact_lists/{list_id}` | admin (api.ts) |  |  | `api_routes.py:3868` delete_contact_list |
| PATCH | `/api/admin/contact_lists/{list_id}` | admin (api.ts) | 🆕 new |  | `api_routes.py:3855` rename_contact_list |
| GET | `/api/admin/contact_lists/{list_id}/contacts` | admin (api.ts) |  |  | `api_routes.py:3806` get_list_contacts |
| POST | `/api/admin/contact_lists/{list_id}/contacts` | admin (api.ts) |  | The admin console has no "add contacts to list" action yet. | `api_routes.py:3818` add_contacts_to_list |
| DELETE | `/api/admin/contact_lists/{list_id}/contacts/{contact_id}` | admin (api.ts) | 🆕 new |  | `api_routes.py:3840` remove_contact_from_list |
| POST | `/api/admin/contacts/batch_delete` | admin (api.ts) |  |  | `api_routes.py:3717` batch_delete_contacts |
| GET | `/api/admin/contacts/deleted` | admin (api.ts) |  |  | `api_routes.py:2834` list_admin_deleted_contacts |
| GET | `/api/admin/contacts/deleted/{contact_id}` | admin (api.ts) |  |  | `api_routes.py:2847` get_admin_deleted_contact |
| GET | `/api/admin/contacts/duplicates` | admin (api.ts) |  |  | `api_routes.py:3585` find_duplicate_contacts |
| POST | `/api/admin/contacts/duplicates/dismiss` | admin (api.ts) | 🆕 new |  | `api_routes.py:3659` dismiss_duplicate_pair |
| POST | `/api/admin/contacts/merge` | admin (api.ts) |  |  | `api_routes.py:3685` merge_contacts |
| GET | `/api/admin/contacts/{contact_id}/duplicates` | admin (api.ts) |  |  | `api_routes.py:3621` admin_contact_duplicates |
| GET | `/api/admin/crm` | admin (api.ts) |  |  | `crm_routes.py:74` list_crms |
| GET | `/api/admin/crm/jobs/{job_id}` | admin (api.ts) |  |  | `crm_routes.py:142` get_job |
| POST | `/api/admin/crm/jobs/{job_id}/cancel` | admin (api.ts) |  |  | `crm_routes.py:150` cancel_job |
| GET | `/api/admin/crm/{key}/csv` | admin (api.ts) |  |  | `crm_routes.py:98` crm_csv |
| POST | `/api/admin/crm/{key}/push` | admin (api.ts) |  |  | `crm_routes.py:117` start_push |
| POST | `/api/admin/crm/{key}/test` | admin (api.ts) |  |  | `crm_routes.py:85` test_crm |
| GET | `/api/admin/dashboard` | admin (api.ts) |  |  | `api_routes.py:1921` get_admin_dashboard |
| GET | `/api/admin/events` | admin (api.ts) |  |  | `api_routes.py:1751` admin_list_events |
| GET | `/api/admin/events/detail` | admin (api.ts) |  |  | `api_routes.py:1823` admin_event_detail |
| POST | `/api/admin/files/upload` | admin (api.ts) |  |  | `api_routes.py:4465` upload_event_file |
| DELETE | `/api/admin/files/{file_id}` | admin (api.ts) |  |  | `api_routes.py:4527` delete_event_file |
| POST | `/api/admin/import_exhibitors` | admin (api.ts) |  |  | `api_routes.py:4348` import_exhibitors |
| GET | `/api/admin/me` | admin (api.ts) |  |  | `api_routes.py:3056` get_my_account |
| PATCH | `/api/admin/me` | admin (api.ts) |  |  | `api_routes.py:3063` update_my_account |
| POST | `/api/admin/me/password` | admin (api.ts) |  |  | `api_routes.py:3109` change_my_password |
| GET | `/api/admin/pipelines` | admin (api.ts) |  |  | `api_routes.py:3964` list_pipelines |
| POST | `/api/admin/pipelines/settings` | admin (api.ts) |  |  | `api_routes.py:3985` update_pipeline_settings |
| GET | `/api/admin/planner/discovery` | admin (api.ts) |  |  | `api_routes.py:5602` planner_discovery_status |
| GET | `/api/admin/planner/events` | admin (api.ts) |  |  | `api_routes.py:5429` planner_list_events |
| POST | `/api/admin/planner/events` | admin (api.ts) |  |  | `api_routes.py:5435` planner_create_event |
| DELETE | `/api/admin/planner/events/{event_id}` | admin (api.ts) |  |  | `api_routes.py:5481` planner_delete_event |
| PATCH | `/api/admin/planner/events/{event_id}` | admin (api.ts) |  |  | `api_routes.py:5452` planner_update_event |
| POST | `/api/admin/planner/events/{event_id}/announce` | admin (api.ts) |  |  | `api_routes.py:5533` planner_announce |
| PUT | `/api/admin/planner/events/{event_id}/team` | admin (api.ts) |  |  | `api_routes.py:5495` planner_set_team |
| POST | `/api/admin/planner/extract` | admin (api.ts) |  |  | `api_routes.py:5557` planner_extract |
| GET | `/api/admin/sales-profile` | admin (api.ts) |  |  | `api_routes.py:3184` get_admin_sales_profile |
| PATCH | `/api/admin/sales-profile` | admin (api.ts) |  |  | `api_routes.py:3192` update_admin_sales_profile |
| GET | `/api/admin/segments` | admin (api.ts) |  |  | `api_routes.py:2411` get_segments |
| GET | `/api/admin/segments/{dim}/contacts` | admin (api.ts) |  |  | `api_routes.py:2444` get_segment_contacts |
| GET | `/api/admin/users` | admin (api.ts) |  |  | `api_routes.py:2935` list_admin_users |
| POST | `/api/admin/users` | admin (api.ts) |  |  | `api_routes.py:2951` create_admin_user |
| POST | `/api/admin/users/merge` | admin (api.ts) |  |  | `api_routes.py:3323` merge_users |
| DELETE | `/api/admin/users/{user_id}` | admin (api.ts) | 🆕 new |  | `api_routes.py:3225` delete_admin_user |
| PATCH | `/api/admin/users/{user_id}` | admin (api.ts) | 🆕 new |  | `api_routes.py:2970` update_admin_user |
| POST | `/api/admin/webhook/test` | admin (api.ts) |  |  | `api_routes.py:5156` test_webhook |
| POST | `/api/broadcasts/{broadcast_id}/dismiss` | admin (api.ts) |  |  | `api_routes.py:3517` dismiss_broadcast |
| GET | `/api/contact/{contact_id}/admin_score` | admin (api.ts) |  |  | `api_routes.py:4678` get_admin_score |
| GET | `/api/contact/{contact_id}/pipeline` | admin (api.ts) |  |  | `api_routes.py:4057` get_contact_pipeline |
| POST | `/api/contact/{contact_id}/pipeline/retry` | admin (api.ts) |  |  | `api_routes.py:4051` retry_contact_pipeline |
| POST | `/api/contact/{contact_id}/pipeline/run` | admin (ContactDetailDrawer.tsx, api.ts) |  |  | `api_routes.py:4044` run_contact_pipeline |
| POST | `/api/converse` | mobile (api.converseAI)<br>admin (api.ts) |  |  | `api_routes.py:4969` converse_ai |
| POST | `/api/exhibitors` | mobile (api.addBooth)<br>admin (api.ts) | 🆕 new |  | `api_routes.py:4237` add_booth |
| GET | `/api/live` | admin (__root.tsx) |  |  | `api_routes.py:2459` live_event_stream |
| GET | `/api/models` | admin (api.ts) |  |  | `api_routes.py:4934` get_available_models |
| POST | `/api/transcribe` | admin (api.ts) |  |  | `api_routes.py:5021` transcribe_question |

**Unused (33)**

| Method | Path | Flags | Notes | Handler |
|---|---|---|---|---|
| GET | `/api/admin/export` |  |  | `api_routes.py:5223` export_all_contacts_csv |
| POST | `/api/converse/stream` |  |  | `api_routes.py:4985` converse_ai_stream |
| POST | `/api/converse/stream/files` |  |  | `api_routes.py:4995` converse_ai_stream_files |
| GET | `/admin/activity` | legacy |  | `app.py:5681` admin_activity_feed |
| POST | `/admin/backup` | legacy |  | `app.py:7356` admin_trigger_backup |
| GET | `/admin/backup/status` | legacy |  | `app.py:7368` admin_backup_status |
| POST | `/admin/broadcast` | legacy |  | `app.py:6952` admin_send_broadcast |
| DELETE | `/admin/contact/files/{file_id}` | legacy |  | `app.py:7323` delete_contact_file |
| DELETE | `/admin/contact/{contact_id}` | legacy |  | `app.py:5720` admin_delete_contact |
| POST | `/admin/contact/{contact_id}/files` | legacy |  | `app.py:7179` upload_contact_file |
| PUT | `/admin/contact/{contact_id}/note` | legacy |  | `app.py:7057` admin_update_contact_note |
| GET | `/admin/contact_lists` | legacy |  | `app.py:7591` get_contact_lists |
| POST | `/admin/contact_lists` | legacy |  | `app.py:7571` create_contact_list |
| DELETE | `/admin/contact_lists/{list_id}` | legacy |  | `app.py:7716` delete_contact_list |
| GET | `/admin/contact_lists/{list_id}/contacts` | legacy |  | `app.py:7671` get_contacts_in_list |
| POST | `/admin/contact_lists/{list_id}/contacts` | legacy |  | `app.py:7627` add_contacts_to_list |
| DELETE | `/admin/contact_lists/{list_id}/contacts/{contact_id}` | legacy |  | `app.py:7739` remove_contact_from_list |
| GET | `/admin/contacts` | legacy |  | `app.py:5577` admin_list_contacts |
| POST | `/admin/contacts/batch_delete` | legacy |  | `app.py:6036` admin_batch_delete_contacts |
| GET | `/admin/contacts/duplicates` | legacy |  | `app.py:5917` admin_find_duplicates |
| POST | `/admin/contacts/merge` | legacy |  | `app.py:5871` admin_merge_contacts |
| GET | `/admin/dashboard` | legacy |  | `app.py:6133` admin_dashboard |
| GET | `/admin/export` | legacy |  | `app.py:7092` admin_export_all_contacts |
| POST | `/admin/files/upload` | legacy |  | `app.py:6284` upload_event_file |
| DELETE | `/admin/files/{file_id}` | legacy |  | `app.py:6441` delete_event_file |
| POST | `/admin/import_exhibitors/` | legacy |  | `app.py:5375` import_exhibitors_route |
| POST | `/admin/pipeline/settings` | legacy |  | `app.py:7544` admin_pipeline_settings |
| GET | `/admin/pipelines` | legacy |  | `app.py:7484` admin_list_pipelines |
| GET | `/admin/search` | legacy |  | `app.py:6888` admin_search_contacts |
| POST | `/admin/test_webhook` | legacy |  | `app.py:6227` admin_test_webhook |
| GET | `/admin/users` | legacy |  | `app.py:5501` admin_list_users |
| POST | `/admin/users/merge` | legacy |  | `app.py:5769` admin_merge_users |
| POST | `/admin/webhook/test` | legacy |  | `app.py:6074` test_webhook_connection |

### 6.2 User APIs (any signed-in user — the mobile app) — 87 total, 58 used, 29 unused

**Used (58)**

| Method | Path | Used by | Flags | Notes | Handler |
|---|---|---|---|---|---|
| POST | `/add_contact_from_image/` | mobile (api.addContactFromImage) |  |  | `app.py:3792` add_contact_image_route |
| GET | `/api/broadcasts/active` | admin (api.ts) | ✅ newly used in app |  | `api_routes.py:3382` get_active_broadcasts |
| GET | `/api/broadcasts/mine` | mobile (api.getMyBroadcasts) |  |  | `api_routes.py:3409` my_broadcasts |
| DELETE | `/api/contact/files/{file_id}` | mobile (api.deleteContactFile)<br>admin (api.ts) | 🆕 new |  | `api_routes.py:4608` remove_contact_file |
| GET | `/api/contact/files/{file_id}/download` | mobile (api.downloadContactFile)<br>admin (api.ts) | 🆕 new |  | `api_routes.py:4628` download_contact_file |
| POST | `/api/contact/{contact_id}/enrich` | admin (ContactDetailDrawer.tsx) | ⚠️ stub | Always 501 "not connected to a data provider" (no enrichment provider is set up). | `api_routes.py:4853` enrich_contact |
| GET | `/api/contact/{contact_id}/files` | mobile (api.getContactFiles)<br>admin (ContactDetailDrawer.tsx, api.ts) | ✅ newly used in app |  | `api_routes.py:4540` get_contact_files |
| POST | `/api/contact/{contact_id}/files` | mobile (api.attachContactFile)<br>admin (api.ts) | 🆕 new |  | `api_routes.py:4573` attach_contact_file |
| GET | `/api/contact/{contact_id}/qr` | mobile (api.getContactQR)<br>admin (ContactDetailDrawer.tsx, api.ts) |  |  | `api_routes.py:4894` get_contact_qr |
| POST | `/api/contact/{contact_id}/score` | mobile (api.scoreContact)<br>admin (ContactDetailDrawer.tsx) |  |  | `api_routes.py:4724` score_single_contact |
| GET | `/api/contacts` | mobile (api.listContacts)<br>admin (useContactsQuery.ts, api.ts) |  |  | `api_routes.py:649` list_contacts |
| POST | `/api/contacts` | mobile (api.createContact)<br>admin (api.ts) |  |  | `api_routes.py:1370` create_contact |
| POST | `/api/contacts/check-duplicates` | mobile (api.checkDuplicates) |  |  | `api_routes.py:1259` check_duplicates |
| DELETE | `/api/contacts/deleted` | mobile (api.emptyDeletedContacts)<br>admin (ContactDetailDrawer.tsx, api.ts) |  |  | `api_routes.py:986` empty_deleted_contacts |
| GET | `/api/contacts/deleted` | mobile (api.listDeletedContacts)<br>admin (api.ts) |  |  | `api_routes.py:971` list_deleted_contacts |
| DELETE | `/api/contacts/deleted/{contact_id}` | mobile (api.purgeDeletedContact) |  |  | `api_routes.py:1036` purge_deleted_contact |
| POST | `/api/contacts/deleted/{contact_id}/restore` | mobile (api.restoreContact)<br>admin (api.ts) |  |  | `api_routes.py:1010` restore_deleted_contact |
| GET | `/api/contacts/duplicates` | mobile (api.listDuplicatePairs)<br>admin (api.ts) |  |  | `api_routes.py:1232` list_duplicate_pairs |
| POST | `/api/contacts/score_all` | mobile (api.scoreAllContacts)<br>admin (api.ts) |  |  | `api_routes.py:4781` score_all_contacts |
| DELETE | `/api/contacts/{contact_id}` | mobile (api.deleteContact)<br>admin (ContactDetailDrawer.tsx, api.ts) |  |  | `api_routes.py:1647` delete_contact |
| GET | `/api/contacts/{contact_id}` | mobile (api.getContact)<br>admin (api.ts) |  |  | `api_routes.py:1319` get_contact |
| PATCH | `/api/contacts/{contact_id}` | mobile (api.updateContact)<br>admin (api.ts) |  |  | `api_routes.py:1443` patch_contact |
| GET | `/api/contacts/{contact_id}/duplicates` | mobile (api.getContactDuplicates) |  |  | `api_routes.py:1282` get_contact_duplicates |
| POST | `/api/contacts/{contact_id}/duplicates/{other_id}/dismiss` | mobile (api.dismissDuplicate) |  |  | `api_routes.py:1295` dismiss_duplicate |
| POST | `/api/contacts/{contact_id}/linkedin/attach` | mobile (api.attachLinkedIn) |  |  | `api_routes.py:1588` linkedin_attach |
| POST | `/api/contacts/{contact_id}/linkedin/search` | mobile (api.findLinkedIn) |  |  | `api_routes.py:1541` linkedin_search |
| GET | `/api/contacts/{contact_id}/share-links` | mobile (api.listShareLinks) |  |  | `share_routes.py:142` list_share_links |
| POST | `/api/contacts/{contact_id}/share-links` | mobile (api.createShareLink) |  |  | `share_routes.py:101` create_share_link |
| DELETE | `/api/contacts/{contact_id}/share-links/{token}` | mobile (api.revokeShareLink) |  |  | `share_routes.py:160` revoke_share_link |
| GET | `/api/events` | mobile (api.listEvents)<br>admin (api.ts) |  |  | `api_routes.py:1663` list_events |
| GET | `/api/events/{event_id}/seats` | mobile (api.getEventSeats) | ✅ newly used in app |  | `api_routes.py:1690` get_event_seats |
| GET | `/api/exhibitors` | mobile (api.listExhibitors)<br>admin (api.ts) |  |  | `api_routes.py:4076` list_exhibitors |
| GET | `/api/exhibitors/categories` | mobile (api.getExhibitorCategories) | ✅ newly used in app |  | `api_routes.py:4165` list_exhibitor_categories |
| DELETE | `/api/exhibitors/{exhibitor_id}` | mobile (api.deleteBooth)<br>admin (api.ts) | 🆕 new |  | `api_routes.py:4301` delete_booth |
| PATCH | `/api/exhibitors/{exhibitor_id}` | mobile (api.updateBooth)<br>admin (api.ts) | 🆕 new ✅ newly used in app |  | `api_routes.py:4275` update_booth |
| DELETE | `/api/exhibitors/{exhibitor_id}/visit` | mobile (api.setBoothVisited) | 🆕 new |  | `api_routes.py:4327` unmark_booth_visited |
| PUT | `/api/exhibitors/{exhibitor_id}/visit` | mobile (api.setBoothVisited) | 🆕 new |  | `api_routes.py:4315` mark_booth_visited |
| GET | `/api/files/list` | mobile (api.listEventFiles)<br>admin (api.ts) | ✅ newly used in app |  | `api_routes.py:4419` list_event_files |
| GET | `/api/files/{file_id}/download` | mobile (api.downloadEventFile)<br>admin (api.ts) | 🆕 new |  | `api_routes.py:4507` download_event_file |
| GET | `/api/me/business-card` | mobile (api.getMyBusinessCard) |  |  | `card_routes.py:140` get_my_business_card |
| GET | `/api/me/card` | mobile (api.getMyCard) |  |  | `card_routes.py:120` get_my_card |
| POST | `/api/me/card/reset-link` | mobile (api.resetMyCardLink) |  |  | `card_routes.py:157` reset_my_card_link |
| GET | `/api/sync/pull` | mobile (api.syncPull) |  |  | `api_routes.py:2651` sync_pull |
| POST | `/api/sync/push` | mobile (api.syncPush, sync/engine.ts) |  |  | `api_routes.py:2525` sync_push |
| GET | `/api/sync/state` | mobile (api.getSyncState) | ⚠️ fake | Marked "Mock sync state": always 0 pending / 0 failed and "last synced" = now. Not used by the apps any more (the phone reads its own outbox). | `api_routes.py:5247` get_sync_state |
| POST | `/contact/{contact_id}/accepted` | external: n8n / ChampMail callback webhook |  |  | `app.py:6809` contact_accepted_webhook |
| POST | `/contact/{contact_id}/enrich` | external: n8n enrichment webhook |  |  | `app.py:4602` enrich_contact_route |
| POST | `/converse/` | mobile (api.converse) |  |  | `app.py:4391` converse_route |
| POST | `/converse/stream/` | mobile (api.converseStream) |  |  | `app.py:4409` converse_stream_route |
| POST | `/converse/with_files/` | mobile (api.converseWithFiles) |  |  | `app.py:4448` converse_with_files_route |
| POST | `/converse/with_files/stream/` | mobile (api.converseStream) |  |  | `app.py:4470` converse_with_files_stream_route |
| GET | `/dashboard` | mobile (api.getDashboard) |  |  | `app.py:4780` get_dashboard |
| GET | `/dashboard` | mobile (api.getDashboard) |  |  | `app.py:5258` dashboard_route |
| GET | `/export_contacts/` | mobile (api.exportContacts) |  |  | `app.py:4493` export_contacts_route |
| POST | `/scan_qr/` | mobile (api.scanQRCode) |  |  | `app.py:3821` scan_qr_route |
| POST | `/transcribe_audio/` | mobile (api.transcribeAudio) |  |  | `app.py:6554` transcribe_audio |
| GET | `/user/profile/` | mobile (api.getProfile) |  |  | `app.py:4686` get_user_profile_route |
| PUT | `/user/profile/` | mobile (api.updateProfile) |  |  | `app.py:4707` update_user_profile_route |

**Unused (29)**

| Method | Path | Flags | Notes | Handler |
|---|---|---|---|---|
| GET | `/api/broadcasts/stream` |  |  | `api_routes.py:3429` my_broadcast_stream |
| POST | `/add_contact/` | legacy |  | `app.py:3773` add_contact_route |
| GET | `/broadcasts/active` | legacy |  | `app.py:6989` get_active_broadcasts |
| POST | `/broadcasts/{broadcast_id}/dismiss` | legacy |  | `app.py:7028` dismiss_broadcast |
| GET | `/contact/files/download/{file_id}` | legacy |  | `app.py:7281` download_contact_file |
| DELETE | `/contact/{contact_id}` | legacy |  | `app.py:3962` delete_contact_route |
| GET | `/contact/{contact_id}` | legacy |  | `app.py:3912` get_contact_route |
| PUT | `/contact/{contact_id}` | legacy |  | `app.py:4008` update_contact_route |
| POST | `/contact/{contact_id}/audio_note` | legacy |  | `app.py:6479` add_audio_note |
| GET | `/contact/{contact_id}/files` | legacy |  | `app.py:7241` list_contact_files |
| GET | `/contact/{contact_id}/pipeline` | legacy |  | `app.py:7763` get_pipeline_status_enhanced |
| POST | `/contact/{contact_id}/pipeline/retry` | legacy |  | `app.py:7457` retry_pipeline |
| POST | `/contact/{contact_id}/pipeline/run` | legacy |  | `app.py:7419` trigger_pipeline |
| POST | `/contact/{contact_id}/score` | legacy |  | `app.py:5150` score_contact_route |
| POST | `/contacts/score_all` | legacy |  | `app.py:5201` score_all_contacts_route |
| GET | `/debug` | legacy | Public diagnostic output — remove or lock down. | `app.py:3052` debug_info |
| GET | `/exhibitors/` | legacy |  | `app.py:5428` list_exhibitors_route |
| GET | `/exhibitors/categories/` | legacy |  | `app.py:5476` list_exhibitor_categories |
| GET | `/files/download/{file_id}` | legacy |  | `app.py:6398` download_event_file |
| GET | `/files/list` | legacy |  | `app.py:6351` list_event_files |
| POST | `/generate_qr/` | legacy |  | `app.py:3832` generate_qr_route |
| GET | `/list_contacts/` | legacy |  | `app.py:3853` list_contacts_route |
| GET | `/models/` | legacy |  | `app.py:5356` list_models |
| POST | `/search/` | legacy |  | `app.py:3841` search_route |
| GET | `/shared_contacts/` | legacy |  | `app.py:5315` list_shared_contacts |
| GET | `/user/card/` | legacy |  | `app.py:6619` get_user_card |
| PUT | `/user/card/` | legacy |  | `app.py:6654` update_user_card |
| POST | `/user/card/generate_token/` | legacy |  | `app.py:6718` generate_card_token |
| GET | `/user/validate` | legacy |  | `app.py:3096` validate_user |

### 6.3 Public APIs (no sign-in) — 22 total, 19 used, 3 unused

**Used (19)**

| Method | Path | Used by | Flags | Notes | Handler |
|---|---|---|---|---|---|
| GET | `/api/health` | mobile (api.healthCheck)<br>admin (api.ts) |  |  | `api_routes.py:607` api_health |
| POST | `/auth/linkedin/callback` | mobile (api.exchangeLinkedInCode) |  |  | `app.py:3244` linkedin_oauth_callback |
| GET | `/auth/linkedin/link` | mobile (api.getLinkedInLinkStatus) |  |  | `app.py:3341` get_linkedin_link_status |
| POST | `/auth/linkedin/link` | mobile (api.linkLinkedIn) |  |  | `app.py:3361` link_linkedin_account |
| POST | `/auth/linkedin/unlink` | mobile (api.unlinkLinkedIn) |  |  | `app.py:3476` unlink_linkedin_account |
| GET | `/auth/linkedin/url` | mobile (api.getLinkedInAuthUrl) |  |  | `app.py:3231` get_linkedin_auth_endpoint |
| POST | `/auth/password/forgot` | mobile (api.requestPasswordReset) |  |  | `app.py:3583` request_password_reset |
| POST | `/auth/password/reset` | mobile (api.resetPassword) |  |  | `app.py:3650` reset_password_with_code |
| POST | `/auth/password/set` | mobile (api.setPassword) |  |  | `app.py:3442` set_account_password |
| POST | `/auth/refresh` | mobile (api.refreshToken) | ✅ newly used in app |  | `app.py:3740` refresh_auth_token |
| GET | `/auth/session` | mobile (api.checkSession) |  |  | `app.py:3720` check_session |
| GET | `/c/{token}` | external: Public "my card" page (link people open) |  |  | `card_routes.py:204` my_card_page |
| GET | `/c/{token}/card.png` | external: Linked from the public card page |  |  | `card_routes.py:192` my_card_png |
| GET | `/c/{token}/contact.vcf` | external: Linked from the public card page |  |  | `card_routes.py:176` my_card_vcard |
| GET | `/health/` | external: Docker health check (docker-compose.yml) |  |  | `app.py:2984` health_check |
| POST | `/login/` | mobile (api.login) |  |  | `app.py:3183` login_user |
| POST | `/register/` | mobile (api.register) |  |  | `app.py:3138` register_user |
| GET | `/s/{token}` | external: Public shared-contact page (link people open) |  |  | `share_routes.py:317` shared_contact_page |
| GET | `/s/{token}/contact.vcf` | external: Linked from the shared-contact page |  |  | `share_routes.py:235` shared_contact_vcard |

**Unused (3)**

| Method | Path | Flags | Notes | Handler |
|---|---|---|---|---|
| GET | `/api/shared/{token}` |  | JSON version of a shared contact; the /s/{token} page is rendered on the server instead. | `share_routes.py:229` shared_contact_json |
| GET | `/` | legacy |  | `app.py:2979` root |
| GET | `/card/{token}` | legacy | Old card page on a retired domain; the app uses /c/{token}. | `app.py:6779` view_card_public |
