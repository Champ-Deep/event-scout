"""
One-time migration script: Move existing file-based data to PostgreSQL.

Usage:
    DATABASE_PUBLIC_URL=postgresql://... python migrate.py

This reads:
  - users.json (or /app/users/users.json) for user accounts
  - users/{user_id}/metadata.pickle for contacts
  - users/{user_id}/profile.json for user profiles

And inserts them into the PostgreSQL database.
"""

import os
import sys
import json
import pickle
import uuid
from datetime import datetime, timezone

# Use sync SQLAlchemy for migration (simpler)
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

# Get the database URL from env or command line
DATABASE_URL = os.environ.get("DATABASE_PUBLIC_URL") or os.environ.get("DATABASE_URL", "")

if not DATABASE_URL:
    print("ERROR: Set DATABASE_PUBLIC_URL or DATABASE_URL environment variable")
    print("Example: DATABASE_PUBLIC_URL=postgresql://postgres:xxx@host:port/railway python migrate.py")
    sys.exit(1)

# Ensure it's a proper postgresql:// URL (not postgres://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

print(f"Connecting to: {DATABASE_URL[:50]}...")

engine = create_engine(DATABASE_URL, echo=False)

# Find the users directory
USERS_DIR = None
for candidate in ["users", "/app/users", os.path.join(os.getcwd(), "users")]:
    if os.path.exists(candidate):
        USERS_DIR = candidate
        break

if not USERS_DIR:
    print("No users directory found locally.")

# Find users.json
USERS_FILE = None
for candidate in [
    os.path.join(USERS_DIR, "users.json"),
    "users.json",
    os.path.join(os.getcwd(), "users.json"),
]:
    if os.path.exists(candidate):
        USERS_FILE = candidate
        break

print(f"Users directory: {USERS_DIR}")
print(f"Users file: {USERS_FILE}")


def create_tables(engine):
    """Create tables if they don't exist."""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS contacts (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id),
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) DEFAULT 'N/A',
                phone VARCHAR(100) DEFAULT 'N/A',
                linkedin VARCHAR(500) DEFAULT 'N/A',
                linkedin_source VARCHAR(20) DEFAULT 'card',
                company_name VARCHAR(255) DEFAULT 'N/A',
                notes TEXT DEFAULT '',
                links JSONB DEFAULT '[]',
                source VARCHAR(50) DEFAULT 'manual',
                lead_score INTEGER,
                lead_temperature VARCHAR(10),
                lead_score_reasoning TEXT DEFAULT '',
                lead_score_breakdown JSONB DEFAULT '{}',
                lead_recommended_actions JSONB DEFAULT '[]',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS shared_contacts (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                original_contact_id UUID,
                user_id UUID,
                name VARCHAR(255) DEFAULT 'N/A',
                email VARCHAR(255) DEFAULT 'N/A',
                phone VARCHAR(100) DEFAULT 'N/A',
                linkedin VARCHAR(500) DEFAULT 'N/A',
                company_name VARCHAR(255) DEFAULT 'N/A',
                notes TEXT DEFAULT '',
                links JSONB DEFAULT '[]',
                source VARCHAR(50) DEFAULT 'manual',
                webhook_sent BOOLEAN DEFAULT FALSE,
                webhook_sent_at TIMESTAMPTZ,
                enriched BOOLEAN DEFAULT FALSE,
                enriched_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID UNIQUE NOT NULL REFERENCES users(id),
                profile_data JSONB DEFAULT '{}',
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS conversations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id),
                title VARCHAR(500) DEFAULT 'New Chat',
                messages JSONB DEFAULT '[]',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        # Create indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_contacts_user_id ON contacts(user_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_contacts_email ON contacts(email)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_shared_contacts_email ON shared_contacts(email)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)"))
        conn.commit()
    print("Tables created/verified.")


def migrate_users(engine, users_data: dict) -> dict:
    """Migrate users from JSON to Postgres. Returns mapping of old_id -> new_uuid."""
    id_map = {}
    migrated = 0
    skipped = 0

    with Session(engine) as session:
        for old_user_id, user_info in users_data.items():
            email = user_info.get("email", "")
            if not email:
                print(f"  Skipping user {old_user_id}: no email")
                skipped += 1
                continue

            # Check if already migrated
            result = session.execute(
                text("SELECT id FROM users WHERE email = :email"),
                {"email": email}
            )
            existing = result.fetchone()
            if existing:
                id_map[old_user_id] = str(existing[0])
                print(f"  User {email} already exists (ID: {existing[0]})")
                skipped += 1
                continue

            # Try to use the same UUID if it's valid
            try:
                new_id = uuid.UUID(old_user_id)
            except ValueError:
                new_id = uuid.uuid4()

            session.execute(
                text("""
                    INSERT INTO users (id, name, email, password_hash, created_at)
                    VALUES (:id, :name, :email, :password_hash, :created_at)
                    ON CONFLICT (email) DO NOTHING
                """),
                {
                    "id": str(new_id),
                    "name": user_info.get("name", "Unknown"),
                    "email": email,
                    "password_hash": user_info.get("password", ""),
                    "created_at": datetime.now(timezone.utc),
                }
            )
            id_map[old_user_id] = str(new_id)
            migrated += 1
            print(f"  Migrated user: {email} (ID: {new_id})")

        session.commit()

    print(f"Users: {migrated} migrated, {skipped} skipped")
    return id_map


def migrate_contacts(engine, user_id_map: dict):
    """Migrate contacts from pickle files to Postgres."""
    total_migrated = 0
    total_skipped = 0

    with Session(engine) as session:
        for old_user_id, new_user_id in user_id_map.items():
            metadata_path = os.path.join(USERS_DIR, old_user_id, "metadata.pickle")
            if not os.path.exists(metadata_path):
                continue

            try:
                with open(metadata_path, "rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                print(f"  Error loading pickle for {old_user_id}: {e}")
                continue

            doc_metadata = data.get("doc_metadata", [])
            if not doc_metadata:
                continue

            print(f"  User {old_user_id[:8]}...: {len(doc_metadata)} contacts")

            for meta in doc_metadata:
                contact_name = meta.get("name", "N/A")
                contact_email = meta.get("email", "N/A")

                # Check if already migrated (by name + user_id)
                result = session.execute(
                    text("SELECT id FROM contacts WHERE user_id = :user_id AND name = :name AND email = :email LIMIT 1"),
                    {"user_id": new_user_id, "name": contact_name, "email": contact_email}
                )
                if result.fetchone():
                    total_skipped += 1
                    continue

                contact_id = meta.get("id", str(uuid.uuid4()))
                try:
                    uuid.UUID(contact_id)
                except ValueError:
                    contact_id = str(uuid.uuid4())

                session.execute(
                    text("""
                        INSERT INTO contacts (id, user_id, name, email, phone, linkedin, company_name, notes, links, source, lead_score, lead_temperature, lead_score_reasoning, lead_score_breakdown, lead_recommended_actions, created_at, updated_at)
                        VALUES (:id, :user_id, :name, :email, :phone, :linkedin, :company_name, :notes, :links, :source, :lead_score, :lead_temperature, :lead_score_reasoning, :lead_score_breakdown, :lead_recommended_actions, :created_at, :updated_at)
                        ON CONFLICT (id) DO NOTHING
                    """),
                    {
                        "id": contact_id,
                        "user_id": new_user_id,
                        "name": contact_name,
                        "email": contact_email,
                        "phone": meta.get("phone", "N/A"),
                        "linkedin": meta.get("linkedin", "N/A"),
                        "company_name": meta.get("company_name", "N/A"),
                        "notes": meta.get("notes", ""),
                        "links": json.dumps(meta.get("links", [])),
                        "source": meta.get("source", "manual"),
                        "lead_score": meta.get("lead_score"),
                        "lead_temperature": meta.get("lead_temperature"),
                        "lead_score_reasoning": meta.get("lead_score_reasoning", ""),
                        "lead_score_breakdown": json.dumps(meta.get("lead_score_breakdown", {})),
                        "lead_recommended_actions": json.dumps(meta.get("lead_recommended_actions", [])),
                        "created_at": datetime.now(timezone.utc),
                        "updated_at": datetime.now(timezone.utc),
                    }
                )

                # Also add to shared_contacts
                session.execute(
                    text("""
                        INSERT INTO shared_contacts (original_contact_id, user_id, name, email, phone, linkedin, company_name, notes, links, source, created_at)
                        VALUES (:original_contact_id, :user_id, :name, :email, :phone, :linkedin, :company_name, :notes, :links, :source, :created_at)
                    """),
                    {
                        "original_contact_id": contact_id,
                        "user_id": new_user_id,
                        "name": contact_name,
                        "email": contact_email,
                        "phone": meta.get("phone", "N/A"),
                        "linkedin": meta.get("linkedin", "N/A"),
                        "company_name": meta.get("company_name", "N/A"),
                        "notes": meta.get("notes", ""),
                        "links": json.dumps(meta.get("links", [])),
                        "source": meta.get("source", "manual"),
                        "created_at": datetime.now(timezone.utc),
                    }
                )

                total_migrated += 1

        session.commit()

    print(f"Contacts: {total_migrated} migrated, {total_skipped} skipped")


def migrate_profiles(engine, user_id_map: dict):
    """Migrate user profiles from JSON files to Postgres."""
    migrated = 0

    with Session(engine) as session:
        for old_user_id, new_user_id in user_id_map.items():
            profile_path = os.path.join(USERS_DIR, old_user_id, "profile.json")
            if not os.path.exists(profile_path):
                continue

            try:
                with open(profile_path, "r") as f:
                    profile_data = json.load(f)
            except Exception as e:
                print(f"  Error loading profile for {old_user_id}: {e}")
                continue

            if not profile_data:
                continue

            session.execute(
                text("""
                    INSERT INTO user_profiles (user_id, profile_data, updated_at)
                    VALUES (:user_id, :profile_data, :updated_at)
                    ON CONFLICT (user_id) DO UPDATE SET profile_data = :profile_data, updated_at = :updated_at
                """),
                {
                    "user_id": new_user_id,
                    "profile_data": json.dumps(profile_data),
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            migrated += 1
            print(f"  Migrated profile for user {old_user_id[:8]}...")

        session.commit()

    print(f"Profiles: {migrated} migrated")


def verify_migration(engine):
    """Print summary of migrated data."""
    with Session(engine) as session:
        users = session.execute(text("SELECT COUNT(*) FROM users")).scalar()
        contacts = session.execute(text("SELECT COUNT(*) FROM contacts")).scalar()
        shared = session.execute(text("SELECT COUNT(*) FROM shared_contacts")).scalar()
        profiles = session.execute(text("SELECT COUNT(*) FROM user_profiles")).scalar()

    print(f"\n=== Migration Summary ===")
    print(f"Users: {users}")
    print(f"Contacts: {contacts}")
    print(f"Shared contacts: {shared}")
    print(f"User profiles: {profiles}")


if __name__ == "__main__":
    print("=== Event Scout Data Migration ===\n")

    # Step 1: Create tables
    print("Step 1: Creating tables...")
    create_tables(engine)

    # Step 2: Load users.json
    users_data = {}
    if USERS_FILE and os.path.exists(USERS_FILE):
        print(f"\nStep 2: Loading users from {USERS_FILE}...")
        with open(USERS_FILE, "r") as f:
            users_data = json.load(f)
        print(f"  Found {len(users_data)} users")
    else:
        print("\nStep 2: No users.json found, skipping user migration")

    # Step 3: Migrate users
    if users_data:
        print("\nStep 3: Migrating users...")
        user_id_map = migrate_users(engine, users_data)
    else:
        user_id_map = {}

    # Step 4: Migrate contacts
    if user_id_map:
        print("\nStep 4: Migrating contacts...")
        migrate_contacts(engine, user_id_map)

        # Step 5: Migrate profiles
        print("\nStep 5: Migrating profiles...")
        migrate_profiles(engine, user_id_map)
    else:
        print("\nStep 4-5: No users to migrate contacts/profiles for")

    # Step 6: Verify
    verify_migration(engine)
    print("\nMigration complete!")
