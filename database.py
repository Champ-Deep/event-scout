"""
PostgreSQL database layer for Event Scout.
Uses SQLAlchemy async with asyncpg for all data persistence.
FAISS is kept in-memory only for vector search, rebuilt from Postgres on startup.
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    Column, String, Text, Boolean, Integer, Float, DateTime,
    ForeignKey, Index, JSON, LargeBinary, create_engine, text
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, relationship


# --- Database URL ---
DATABASE_URL = os.environ.get("DATABASE_URL", "")
BACKUP_DATABASE_URL = os.environ.get("BACKUP_DATABASE_URL", "")


def _to_async_url(url):
    """Convert a postgres:// URL to postgresql+asyncpg:// for SQLAlchemy async."""
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


ASYNC_DATABASE_URL = _to_async_url(DATABASE_URL)
ASYNC_BACKUP_URL = _to_async_url(BACKUP_DATABASE_URL)

# Sync URL for migrations
if DATABASE_URL.startswith("postgres://"):
    SYNC_DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
else:
    SYNC_DATABASE_URL = DATABASE_URL


# --- SQLAlchemy Base ---
class Base(DeclarativeBase):
    pass


# --- Models ---
class UserDB(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False, server_default="false")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    contacts = relationship("ContactDB", back_populates="user", cascade="all, delete-orphan")
    profile = relationship("UserProfileDB", back_populates="user", uselist=False, cascade="all, delete-orphan")


class ContactDB(Base):
    __tablename__ = "contacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    email = Column(String(255), default="N/A")
    phone = Column(String(100), default="N/A")
    linkedin = Column(String(500), default="N/A")
    linkedin_source = Column(String(20), default="card")  # 'card' or 'ai_detected'
    company_name = Column(String(255), default="N/A")
    notes = Column(Text, default="")
    links = Column(JSON, default=list)  # [{url, label, added_by}]
    source = Column(String(50), default="manual")  # 'manual', 'scan', 'qr'
    lead_score = Column(Integer, nullable=True)
    lead_temperature = Column(String(10), nullable=True)
    lead_score_reasoning = Column(Text, default="")
    lead_score_breakdown = Column(JSON, default=dict)
    lead_recommended_actions = Column(JSON, default=list)
    audio_notes = Column(JSON, default=list)  # [{audio_base64, transcript, timestamp}]
    admin_notes = Column(Text, default="")  # Admin-only annotations/intel
    photo_base64 = Column(Text, nullable=True)  # Compressed JPEG thumbnail from scanned card
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("UserDB", back_populates="contacts")

    __table_args__ = (
        Index("idx_contacts_user_id", "user_id"),
        Index("idx_contacts_email", "email"),
    )


class SharedContactDB(Base):
    __tablename__ = "shared_contacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_contact_id = Column(UUID(as_uuid=True), nullable=True)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    name = Column(String(255), default="N/A")
    email = Column(String(255), default="N/A")
    phone = Column(String(100), default="N/A")
    linkedin = Column(String(500), default="N/A")
    company_name = Column(String(255), default="N/A")
    notes = Column(Text, default="")
    links = Column(JSON, default=list)
    source = Column(String(50), default="manual")
    webhook_sent = Column(Boolean, default=False)
    webhook_sent_at = Column(DateTime(timezone=True), nullable=True)
    enriched = Column(Boolean, default=False)
    enriched_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_shared_contacts_email", "email"),
    )


class UserProfileDB(Base):
    __tablename__ = "user_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), unique=True, nullable=False)
    profile_data = Column(JSON, default=dict)  # Store full profile as JSON for flexibility
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    user = relationship("UserDB", back_populates="profile")


class ExhibitorDB(Base):
    """WHX exhibitor data for event intelligence."""
    __tablename__ = "exhibitors"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_name = Column(String(255), default="WHX Dubai 2026")
    name = Column(String(500), nullable=False)
    booth = Column(String(100), default="")
    hall = Column(String(100), default="")
    category = Column(String(500), default="")
    subcategory = Column(String(500), default="")
    country = Column(String(100), default="")
    website = Column(String(500), default="")
    description = Column(Text, default="")
    products = Column(JSON, default=list)
    tags = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_exhibitors_event_name", "event_name"),
        Index("idx_exhibitors_name", "name"),
        Index("idx_exhibitors_category", "category"),
    )


class ConversationDB(Base):
    """Store AI conversation history per user for persistence."""
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    title = Column(String(500), default="New Chat")
    messages = Column(JSON, default=list)  # [{role, content, timestamp}]
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_conversations_user_id", "user_id"),
    )


class UserCardDB(Base):
    """Store user's digital business card for QR/NFC sharing."""
    __tablename__ = "user_cards"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), unique=True, nullable=False)
    card_data = Column(JSON, default=dict)  # {name, title, company, email, phone, linkedin, zoom, photo_url, etc.}
    shareable_token = Column(String(100), unique=True, nullable=True, index=True)  # UUID for public access
    is_active = Column(Boolean, default=True)  # Allow users to temporarily disable card sharing
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_user_cards_token", "shareable_token"),
    )


class EventFileDB(Base):
    """Store event materials (PDFs, PPTs) uploaded by admins for team access."""
    __tablename__ = "event_files"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(500), nullable=False)
    original_filename = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)  # 'pdf', 'ppt', 'pptx'
    mime_type = Column(String(100), nullable=False)
    file_size = Column(Integer, nullable=False)  # bytes
    file_data = Column(LargeBinary, nullable=False)  # Actual file content (bytea)
    description = Column(Text, default="")
    event_name = Column(String(255), default="WHX Dubai 2026")
    category = Column(String(100), default="general")  # 'general', 'catalog', 'brochure', 'pricing'
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False, server_default="true")
    download_count = Column(Integer, default=0, nullable=False, server_default="0")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_event_files_event_name", "event_name"),
        Index("idx_event_files_active", "is_active"),
    )


class ContactFileDB(Base):
    """Documents attached to specific contacts by admins (research, pitch decks, briefs)."""
    __tablename__ = "contact_files"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    contact_id = Column(UUID(as_uuid=True), ForeignKey("contacts.id"), nullable=False)
    filename = Column(String(500), nullable=False)
    original_filename = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)
    mime_type = Column(String(100), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_data = Column(LargeBinary, nullable=False)
    description = Column(Text, default="")
    category = Column(String(100), default="research")  # 'research', 'pitch', 'brief', 'other'
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False, server_default="true")
    download_count = Column(Integer, default=0, nullable=False, server_default="0")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_contact_files_contact_id", "contact_id"),
        Index("idx_contact_files_active", "is_active"),
    )


class ContactPipelineDB(Base):
    """Automated intelligence pipeline state per contact (research → pitch → deck)."""
    __tablename__ = "contact_pipelines"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    contact_id = Column(UUID(as_uuid=True), ForeignKey("contacts.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # Pipeline state
    status = Column(String(50), default="pending", nullable=False)
    # pending → researching → scoring → pitching → generating_deck → attaching → complete | failed
    current_step = Column(String(100), default="")
    error_message = Column(Text, nullable=True)

    # Step 1: Research output
    research_summary = Column(Text, default="")
    research_data = Column(JSON, default=dict)

    # Step 2: Score (stored on ContactDB, just track completion)
    score_completed = Column(Boolean, default=False, nullable=False)

    # Step 3: Pitch output
    pitch_angle = Column(Text, default="")
    pitch_email_subject = Column(Text, default="")
    pitch_email_body = Column(Text, default="")
    pitch_slides_content = Column(JSON, default=list)

    # Step 4: Deck output
    deck_file_id = Column(UUID(as_uuid=True), nullable=True)
    presenton_presentation_id = Column(String(255), nullable=True)

    # Timestamps
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_pipelines_contact_id", "contact_id"),
        Index("idx_pipelines_user_id", "user_id"),
        Index("idx_pipelines_status", "status"),
    )


class AdminBroadcastDB(Base):
    """Admin broadcast messages to sales team."""
    __tablename__ = "admin_broadcasts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    admin_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    message = Column(Text, nullable=False)
    priority = Column(String(20), default="normal")  # "normal", "urgent"
    is_active = Column(Boolean, default=True, nullable=False, server_default="true")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


# --- Engine & Session (Primary) ---
engine = None
async_session_factory = None

# --- Engine & Session (Backup) ---
backup_engine = None
backup_session_factory = None


def _create_engine(url):
    """Create an async engine with resilience settings."""
    return create_async_engine(
        url,
        echo=False,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=1800,   # Recycle connections every 30 min
        pool_timeout=30,     # Fail after 30s if no connection available
        connect_args={
            "timeout": 3,           # TCP connection timeout (seconds)
            "command_timeout": 5,   # SQL command timeout (seconds)
        }
    )


def get_engine():
    global engine
    if engine is None:
        if not ASYNC_DATABASE_URL:
            print("[DB] DATABASE_URL environment variable not set")
            return None
        if not ASYNC_DATABASE_URL.startswith("postgresql+asyncpg://"):
            print(f"[DB] Invalid DATABASE_URL format (must start with postgresql+asyncpg://)")
            return None
        try:
            engine = _create_engine(ASYNC_DATABASE_URL)
            print(f"[DB] Engine created successfully")
        except Exception as e:
            print(f"[DB] Failed to create engine: {e}")
            engine = None
    return engine


def get_backup_engine():
    global backup_engine
    if backup_engine is None and ASYNC_BACKUP_URL:
        backup_engine = _create_engine(ASYNC_BACKUP_URL)
    return backup_engine


def get_session_factory():
    global async_session_factory
    if async_session_factory is None:
        eng = get_engine()
        if eng:
            try:
                async_session_factory = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False)
                print("[DB] Session factory created successfully")
            except Exception as e:
                print(f"[DB] Failed to create session factory: {e}")
                async_session_factory = None
        else:
            print("[DB] Cannot create session factory - engine is None (DATABASE_URL may not be set)")
    return async_session_factory


def get_backup_session_factory():
    global backup_session_factory
    if backup_session_factory is None:
        eng = get_backup_engine()
        if eng:
            backup_session_factory = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False)
    return backup_session_factory


async def _init_schema(eng, label="Primary"):
    """Create all tables and run migrations on the given engine."""
    try:
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

            migrations = [
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT FALSE NOT NULL",
                "ALTER TABLE contacts ADD COLUMN IF NOT EXISTS audio_notes JSONB DEFAULT '[]'::jsonb",
                "ALTER TABLE contacts ADD COLUMN IF NOT EXISTS admin_notes TEXT DEFAULT ''",
                "ALTER TABLE contacts ADD COLUMN IF NOT EXISTS photo_base64 TEXT",
            ]
            for sql in migrations:
                try:
                    await conn.execute(text(sql))
                except Exception as mig_err:
                    print(f"[DB-{label}] Migration note: {mig_err}")

        print(f"[DB-{label}] Tables created/verified successfully")
        return True
    except Exception as e:
        print(f"[DB-{label}] Error initializing: {e}")
        return False


async def init_db():
    """Create all tables on primary (and backup if configured)."""
    eng = get_engine()
    if eng is None:
        print("[DB] No DATABASE_URL configured - skipping database init")
        return False

    result = await _init_schema(eng, "Primary")

    # Initialize backup DB schema if configured
    beng = get_backup_engine()
    if beng:
        await _init_schema(beng, "Backup")
        print("[DB] Backup database configured and ready")
    else:
        print("[DB] No BACKUP_DATABASE_URL configured - backup disabled")

    return result


async def dispose_engines():
    """Gracefully dispose of all database engines."""
    global engine, backup_engine, async_session_factory, backup_session_factory
    if engine:
        await engine.dispose()
        engine = None
        async_session_factory = None
        print("[DB] Primary engine disposed")
    if backup_engine:
        await backup_engine.dispose()
        backup_engine = None
        backup_session_factory = None
        print("[DB] Backup engine disposed")


async def get_db() -> AsyncSession:
    """Get an async database session."""
    factory = get_session_factory()
    if factory is None:
        raise Exception("Database not configured")
    async with factory() as session:
        yield session
