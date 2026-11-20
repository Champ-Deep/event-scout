-- Event Scout on D1. Mirrors the Railway Postgres schema (same tables, same columns) so the
-- migrated rows and the ported endpoints line up 1:1. UUIDs are TEXT, JSON is TEXT,
-- booleans are INTEGER 0/1, timestamps are ISO-8601 UTC strings (toISOString format).
-- File bytes live in KV (FILES binding, key file:<id>), not in these rows.

CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  email TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  is_admin INTEGER NOT NULL DEFAULT 0,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
  token_hash TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

CREATE TABLE IF NOT EXISTS contacts (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id),
  name TEXT NOT NULL,
  email TEXT DEFAULT 'N/A',
  phone TEXT DEFAULT 'N/A',
  linkedin TEXT DEFAULT 'N/A',
  linkedin_source TEXT DEFAULT 'card',
  company_name TEXT DEFAULT 'N/A',
  title TEXT DEFAULT '',
  notes TEXT DEFAULT '',
  links TEXT DEFAULT '[]',
  source TEXT DEFAULT 'manual',
  event_name TEXT DEFAULT '',
  lead_score INTEGER,
  lead_temperature TEXT,
  lead_score_reasoning TEXT DEFAULT '',
  lead_score_breakdown TEXT DEFAULT '{}',
  lead_recommended_actions TEXT DEFAULT '[]',
  audio_notes TEXT DEFAULT '[]',
  admin_notes TEXT DEFAULT '',
  photo_base64 TEXT,
  created_at TEXT,
  updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_contacts_user_id ON contacts(user_id);
CREATE INDEX IF NOT EXISTS idx_contacts_email ON contacts(email);
CREATE INDEX IF NOT EXISTS idx_contacts_created ON contacts(created_at);

CREATE TABLE IF NOT EXISTS shared_contacts (
  id TEXT PRIMARY KEY,
  original_contact_id TEXT,
  user_id TEXT,
  name TEXT DEFAULT 'N/A',
  email TEXT DEFAULT 'N/A',
  phone TEXT DEFAULT 'N/A',
  linkedin TEXT DEFAULT 'N/A',
  company_name TEXT DEFAULT 'N/A',
  notes TEXT DEFAULT '',
  links TEXT DEFAULT '[]',
  source TEXT DEFAULT 'manual',
  webhook_sent INTEGER DEFAULT 0,
  webhook_sent_at TEXT,
  enriched INTEGER DEFAULT 0,
  enriched_at TEXT,
  created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_shared_contacts_email ON shared_contacts(email);

CREATE TABLE IF NOT EXISTS user_profiles (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL UNIQUE REFERENCES users(id),
  profile_data TEXT DEFAULT '{}',
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS exhibitors (
  id TEXT PRIMARY KEY,
  event_name TEXT DEFAULT 'WHX Dubai 2026',
  name TEXT NOT NULL,
  booth TEXT DEFAULT '',
  hall TEXT DEFAULT '',
  category TEXT DEFAULT '',
  subcategory TEXT DEFAULT '',
  country TEXT DEFAULT '',
  website TEXT DEFAULT '',
  description TEXT DEFAULT '',
  products TEXT DEFAULT '[]',
  tags TEXT DEFAULT '[]',
  created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_exhibitors_event_name ON exhibitors(event_name);
CREATE INDEX IF NOT EXISTS idx_exhibitors_category ON exhibitors(category);

CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id),
  title TEXT DEFAULT 'New Chat',
  messages TEXT DEFAULT '[]',
  created_at TEXT,
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS user_cards (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL UNIQUE REFERENCES users(id),
  card_data TEXT DEFAULT '{}',
  shareable_token TEXT UNIQUE,
  is_active INTEGER DEFAULT 1,
  created_at TEXT,
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS event_files (
  id TEXT PRIMARY KEY,
  filename TEXT NOT NULL,
  original_filename TEXT NOT NULL,
  file_type TEXT NOT NULL,
  mime_type TEXT NOT NULL,
  file_size INTEGER NOT NULL,
  description TEXT DEFAULT '',
  event_name TEXT DEFAULT 'WHX Dubai 2026',
  category TEXT DEFAULT 'general',
  uploaded_by TEXT NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 1,
  download_count INTEGER NOT NULL DEFAULT 0,
  created_at TEXT,
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS contact_files (
  id TEXT PRIMARY KEY,
  contact_id TEXT NOT NULL,
  filename TEXT NOT NULL,
  original_filename TEXT NOT NULL,
  file_type TEXT NOT NULL,
  mime_type TEXT NOT NULL,
  file_size INTEGER NOT NULL,
  description TEXT DEFAULT '',
  category TEXT DEFAULT 'research',
  uploaded_by TEXT NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 1,
  download_count INTEGER NOT NULL DEFAULT 0,
  created_at TEXT,
  updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_contact_files_contact_id ON contact_files(contact_id);

CREATE TABLE IF NOT EXISTS contact_pipelines (
  id TEXT PRIMARY KEY,
  contact_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  current_step TEXT DEFAULT '',
  error_message TEXT,
  research_summary TEXT DEFAULT '',
  research_data TEXT DEFAULT '{}',
  score_completed INTEGER NOT NULL DEFAULT 0,
  pitch_angle TEXT DEFAULT '',
  pitch_email_subject TEXT DEFAULT '',
  pitch_email_body TEXT DEFAULT '',
  pitch_slides_content TEXT DEFAULT '[]',
  deck_file_id TEXT,
  presenton_presentation_id TEXT,
  gamma_deck_url TEXT,
  started_at TEXT,
  completed_at TEXT,
  created_at TEXT,
  updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_pipelines_contact_id ON contact_pipelines(contact_id);

CREATE TABLE IF NOT EXISTS admin_broadcasts (
  id TEXT PRIMARY KEY,
  admin_id TEXT NOT NULL,
  message TEXT NOT NULL,
  priority TEXT DEFAULT 'normal',
  recipient_ids TEXT,
  is_active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT
);

-- Per-user dismissals were in-memory on Railway; persisted here.
CREATE TABLE IF NOT EXISTS broadcast_dismissals (
  broadcast_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  PRIMARY KEY (broadcast_id, user_id)
);

CREATE TABLE IF NOT EXISTS contact_lists (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  admin_id TEXT NOT NULL,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS contact_list_members (
  id TEXT PRIMARY KEY,
  list_id TEXT NOT NULL REFERENCES contact_lists(id) ON DELETE CASCADE,
  contact_id TEXT NOT NULL,
  added_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_clm_list ON contact_list_members(list_id);

CREATE TABLE IF NOT EXISTS user_exports (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  filename TEXT NOT NULL,
  format TEXT NOT NULL,
  filter_label TEXT,
  contact_count INTEGER,
  file_content TEXT,
  created_at TEXT
);

-- Semantic search: one embedding per contact (Workers AI bge-small, 384 dims), JSON array.
CREATE TABLE IF NOT EXISTS contact_embeddings (
  contact_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  vector TEXT NOT NULL,
  updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_embeddings_user ON contact_embeddings(user_id);
