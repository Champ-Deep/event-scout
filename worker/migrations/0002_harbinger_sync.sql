-- Outbound sync of Event Scout contacts into ChampHarbinger (one row per contact).
CREATE TABLE IF NOT EXISTS harbinger_sync (
  contact_id TEXT PRIMARY KEY,
  status TEXT NOT NULL DEFAULT 'pending',   -- pending | sent | failed
  payload_hash TEXT,                        -- hash of the last payload Harbinger accepted
  attempts INTEGER NOT NULL DEFAULT 0,      -- failures since the last success
  last_error TEXT,
  harbinger_person_id TEXT,
  synced_at TEXT,
  updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_harbinger_sync_status ON harbinger_sync(status);
