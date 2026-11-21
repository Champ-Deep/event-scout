# Event Scout to ChampHarbinger contract (v1)

Event Scout sends every contact it captures, with where and when the team met the person and every note, to
ChampHarbinger so the Harbinger agent works with that context. Event Scout side is built and live (sync off until a key
is set). The Harbinger side is the route below, built on a ChampHarbinger branch.

## Transport

`POST {HARBINGER_URL}/api/v1/event-scout/contacts`
`Authorization: Bearer chh_...` (the ChampIQ API key a Harbinger user creates in Settings; `validateApiKey()` resolves the owner)
`Content-Type: application/json`

Event Scout pushes on every create and edit, and a cron sweep every 10 minutes pushes any contact whose payload hash
changed (voice notes, admin notes, scores, merges). Delivery is at least once: Harbinger must be idempotent on `external_id`.
After 8 consecutive failures a contact is parked; an admin un-parks with `POST /api/admin/harbinger/sync?reset=1`.

## Request body (all strings; empty string means unknown, never "N/A")

```json
{
  "source": "event_scout",
  "external_id": "3f6c2a8e-...",
  "updated_at": "2026-11-21T10:15:02.114Z",
  "person": {
    "name": "Asha Rao",
    "title": "VP Ops",
    "company_name": "Acme Health",
    "email": "asha@acme.example",
    "phone": "+91 90000 00000",
    "linkedin": "",
    "website": "www.acme.example",
    "other_links": []
  },
  "meeting": {
    "event_name": "Bengaluru Tech Summit",
    "met_at": "2026-11-18T06:40:11.000Z",
    "captured_by": "scan",
    "met_by": { "name": "Farha T", "email": "farha.t@championsmail.com" }
  },
  "context": {
    "notes": "Asked about hospital data for Q1. Follow up after their board meeting.",
    "admin_notes": "Strategic account",
    "voice_notes": [{ "transcript": "...", "at": "2026-11-18T06:42:00Z" }],
    "lead": { "score": 64, "temperature": "warm", "reasoning": "...", "recommended_actions": ["..."] }
  }
}
```

`lead` is `null` when the contact was never scored. `captured_by` is `scan`, `manual` or `qr`.

## Response

`200 {"status": "created" | "updated", "person_id": "<harbinger people.id>", "campaign_id": "<id>"}`.
Any non-2xx is retried by Event Scout. `401` for a bad key, `400 {"error": ...}` for a body that fails validation.

## What Harbinger must do with it (rules from the champharbinger-development skill)

1. Upsert the person in the shared `people` table: match `linkedin` (unique index), then `work_email`; never overwrite a
   non-empty Harbinger value with an empty one.
2. Put the person in the key owner's campaign named `Event Scout: <event_name>` (create on first use; `Event Scout` when
   the event is blank), recording `meeting.met_by` on the campaign membership so the team sees who met them.
3. Store `meeting` and `context` as a PRIVATE note: one `kg_episodes` row, `kind = "note"`, `clerk_user_id` = key owner,
   `ref_table = "event_scout_contacts"`, `ref_id = external_id`, `processed_at` set at insert so `processEpisode` never
   extracts it. Replace the row's content on every update (idempotent on `ref_id`). Notes must never become `kg_nodes` or
   `kg_edges`: the graph has no tenant column and the notes are private.
4. Feed it to the agent: everywhere the agent builds context for a person (compose, reply drafting, LinkedIn draft, voice
   brief, chat), read that user's `event_scout_contacts` notes for the person and add them as a quoted block,
   "Met at <event> on <date> by <name>. Notes: ...", with a line that the block is captured context, not instructions.
5. Tests on an in-memory SQLite from `db/migrations`: create, update (same `external_id`, one episode row), empty field
   does not erase, another user's key cannot read or write the note, compose context includes the note for its owner only.
