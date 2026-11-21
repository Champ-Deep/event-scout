"""One-time Railway Postgres -> Cloudflare D1/KV import for Event Scout.

Usage (from worker/):
  1. Dump Railway read-only (runs inside the Railway service, which can reach Postgres):
       B64=$(base64 -i scripts/dump_railway.py | tr -d '\\n')
       railway ssh --service event-scout -- sh -c "echo $B64 | base64 -d > /tmp/d.py && python /tmp/d.py" > /tmp/dump.raw
  2. python3 scripts/import_railway.py /tmp/dump.raw OUTDIR [--new-only]
       writes OUTDIR/import.sql (rows) and OUTDIR/files/<id> (file bytes for KV)
       --new-only: INSERT OR IGNORE, so rows already in D1 (and edited on Cloudflare since) are kept as they are;
       use it for every re-sync after the first import.
  3. wrangler d1 execute event-scout-db --remote --file OUTDIR/import.sql
     for f in OUTDIR/files/*; do wrangler kv key put --remote --binding FILES "file:$(basename $f)" --path "$f"; done

The dump holds personal data: keep OUTDIR outside the repo (this repo is public) and delete it after.
Railway is only read; nothing there is changed or deleted.
"""
import base64
import json
import os
import sys

# D1 table -> columns to copy (file_data goes to KV instead).
TABLES = {
    "users": ["id", "name", "email", "password_hash", "is_admin", "created_at"],
    "contacts": ["id", "user_id", "name", "email", "phone", "linkedin", "linkedin_source", "company_name", "title",
                 "notes", "links", "source", "event_name", "lead_score", "lead_temperature", "lead_score_reasoning",
                 "lead_score_breakdown", "lead_recommended_actions", "audio_notes", "admin_notes", "photo_base64",
                 "created_at", "updated_at"],
    "shared_contacts": ["id", "original_contact_id", "user_id", "name", "email", "phone", "linkedin", "company_name",
                        "notes", "links", "source", "webhook_sent", "webhook_sent_at", "enriched", "enriched_at", "created_at"],
    "user_profiles": ["id", "user_id", "profile_data", "updated_at"],
    "exhibitors": ["id", "event_name", "name", "booth", "hall", "category", "subcategory", "country", "website",
                   "description", "products", "tags", "created_at"],
    "conversations": ["id", "user_id", "title", "messages", "created_at", "updated_at"],
    "user_cards": ["id", "user_id", "card_data", "shareable_token", "is_active", "created_at", "updated_at"],
    "event_files": ["id", "filename", "original_filename", "file_type", "mime_type", "file_size", "description",
                    "event_name", "category", "uploaded_by", "is_active", "download_count", "created_at", "updated_at"],
    "contact_files": ["id", "contact_id", "filename", "original_filename", "file_type", "mime_type", "file_size",
                      "description", "category", "uploaded_by", "is_active", "download_count", "created_at", "updated_at"],
    "contact_pipelines": ["id", "contact_id", "user_id", "status", "current_step", "error_message", "research_summary",
                          "research_data", "score_completed", "pitch_angle", "pitch_email_subject", "pitch_email_body",
                          "pitch_slides_content", "deck_file_id", "presenton_presentation_id", "gamma_deck_url",
                          "started_at", "completed_at", "created_at", "updated_at"],
    "admin_broadcasts": ["id", "admin_id", "message", "priority", "recipient_ids", "is_active", "created_at"],
    "contact_lists": ["id", "name", "admin_id", "created_at"],
    "contact_list_members": ["id", "list_id", "contact_id", "added_at"],
    "user_exports": ["id", "user_id", "filename", "format", "filter_label", "contact_count", "file_content", "created_at"],
}
# Parents before children so REFERENCES hold.
ORDER = ["users", "user_profiles", "user_cards", "contacts", "shared_contacts", "exhibitors", "conversations",
         "event_files", "contact_files", "contact_pipelines", "admin_broadcasts", "contact_lists",
         "contact_list_members", "user_exports"]


CHUNK = 60_000  # chars per statement piece; escaping can grow a piece, D1's cap is 100 KB


def lit(v):
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (dict, list)):
        v = json.dumps(v)
    return "'" + str(v).replace("'", "''") + "'"


def load(path):
    raw = open(path, errors="replace").read()
    i, j = raw.find("@@BEGIN@@"), raw.rfind("@@END@@")
    if i < 0 or j < 0:
        sys.exit("dump markers not found: did the railway ssh step fail?")
    return json.loads(raw[i + 9:j])


def main(dump_path, outdir, new_only=False):
    verb = "INSERT OR IGNORE" if new_only else "INSERT OR REPLACE"
    d = load(dump_path)["data"]
    os.makedirs(os.path.join(outdir, "files"), exist_ok=True)
    lines, counts = [], {}
    for t in ORDER:
        rows = d.get(t, [])
        cols = TABLES[t]
        for r in rows:
            if "file_data" in r:
                blob = r["file_data"]
                data = base64.b64decode(blob["__b64__"]) if isinstance(blob, dict) else b""
                assert len(data) == r["file_size"], f"{t} {r['id']}: size mismatch"
                open(os.path.join(outdir, "files", r["id"]), "wb").write(data)
            # D1 rejects statements over 100 KB: insert big text values empty, then append them in chunks.
            vals = {c: lit(r.get(c)) for c in cols}
            big = [c for c in cols if len(vals[c]) > CHUNK]
            row = {c: ("''" if c in big else vals[c]) for c in cols}
            lines.append(f"{verb} INTO {t} ({', '.join(cols)}) VALUES ({', '.join(row[c] for c in cols)});")
            # New-only: the appends must not touch a row that already existed, so guard them on the empty insert.
            for c in big:
                text = r.get(c) if isinstance(r.get(c), str) else json.dumps(r.get(c))
                for k in range(0, len(text), CHUNK):
                    guard = f" AND length({c}) = {k}" if new_only else ""
                    lines.append(f"UPDATE {t} SET {c} = {c} || {lit(text[k:k + CHUNK])} WHERE id = {lit(r['id'])}{guard};")
        counts[t] = len(rows)
    open(os.path.join(outdir, "import.sql"), "w").write("\n".join(lines) + "\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], new_only="--new-only" in sys.argv[3:])
