# EventScout glossary

Domain words as the code uses them, and the modules that own them. Architecture words (module,
interface, seam, adapter, depth, leverage, locality) follow Matt Pocock's `codebase-design` skill.
Supersedes the older notes in CONTEXT.md where they disagree (bands are A–D, not A–F; seats have no
"owner" role).

## Domain

**Rep**: a non-admin account using the mobile app. "Reps" on every dashboard means non-admin accounts.

**Admin**: an account with console access (`users.is_admin`). Admins are granted with
`manage_admins.py` or the console's Team page — never automatically at sign-up.

**Contact**: a person a rep met, owned by exactly one rep (`contacts.user_id`).

**Event**: a name, not a table row. It appears on contacts ("met at"), in the booth directory, in
reps' event lists, on event files and in the planner. Two names are the same event when their
**event key** matches (case- and spacing-insensitive).

**Active event**: the event a rep is working right now (`profile.current_event_name`); new scans are
tagged with it, including scans taken offline.

**Rep score**: the score a rep's app gives a contact against the rep's own selling profile
(`contacts.lead_score`). Only the rep's app shows it as "the score".

**Admin score**: the console's own score for a contact against the admin sales profile
(`admin_contact_scores`). Everything in the console ranks, filters and counts by it.

**Level**: hot ≥ 85, warm 70–84, cool 50–69, cold < 50, unscored. Band letters A/B/C/D map to the
same cut-offs. One rule everywhere (the Score level module).

**Outbox**: the phone's queue of changes made offline, uploaded by sync in order, per contact.

**Conflict**: an edit to a field that someone else changed since the editor loaded it. Online the
save is refused (409) and the user chooses; offline the notes are merged and other fields keep the
newer value.

**Live feed**: the console's server-sent event stream (`/api/live`); each event names a change.

**Announcement**: a message an admin sends to reps (`admin_broadcasts`).

## Modules

**Contact write module** (`apps/backend/contact_store.py`): the only way a contact is created,
edited or deleted; validates, saves, then always updates the search index and the live feed.

**Score level module** (`apps/backend/score_levels.py`): the level / band rule, in Python and SQL.

**Acting identity module** (`apps/backend/identity.py`): who is acting, as rep or admin, on whose data.

**Scoring module** (`apps/backend/scoring.py`): scores a contact for an audience (rep or admin);
where the result is stored sits behind a seam with a rep adapter and an admin adapter.

**Event identity module** (`apps/backend/event_identity.py`): event keys and a rep's events,
active event and goals (the only code that writes `profile_data`).

**Contact read model** (`apps/backend/contact_view.py`): the named views of a contact (app, console,
lite, export, CRM, search index).

**Contact repository** (`apps/mobile/src/features/contacts/repository.ts`, exposed as
`contactsFeature`): the only way the app's screens read or write contacts — online, offline cache,
outbox, paging and the "a queued local change beats the server copy" rule behind one interface.

**Profile module** (`apps/mobile/src/features/profile`, `profileStore`): the rep's profile, active
event (with its offline copy) and the event-list rules (`eventRules`).

**Backend adapter** (`apps/mobile/src/lib/api.ts` + `requestPipeline.ts` + `apiError.ts`): every
request and stream to the backend through one pipeline (headers, token renewal, revoked session),
failing with one classified `ApiError`.

**Session** (`apps/mobile/src/features/session`, `session`): who is signed in on the phone — sign-in,
sign-out, token renewal and the offline-cache hand-over (same account keeps its offline changes, a
different account starts clean). `SessionProvider` is its React binding.

**Sync engine** (`apps/mobile/src/sync/engine.ts`, `createSyncEngine({remote, store})`): uploads the
outbox and pulls changes; the remote and local store are adapters (real in `localStore.ts`,
in-memory in tests).

**Sheet** (`apps/mobile/src/ui/SheetOverlay.tsx`, `useSheetMotion` + `SheetOverlay`; window-level
host in `OverlayHost.tsx`): slide / fade, drag-to-close and back-button behaviour under every overlay.

**Change map** (`apps/admin/src/lib/changes.ts`, `applyChange`): which console data a named change
affects; used by the live feed and by every write.

**Levels** (`apps/admin/src/lib/levels.ts`, `apps/mobile/src/features/contacts/scoreLevel.ts`): the
apps' view of the Score level rule; they prefer the `level` the server sends.

**Wire contract** (`packages/types`, `@es/types`): the shapes of the backend's JSON that both apps
read, checked at each app's backend adapter in development builds.
