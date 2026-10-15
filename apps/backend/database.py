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


# Production = ENVIRONMENT=production, or running on Railway. Turns missing config into a boot error
# instead of silently using dev fallbacks (local SQLite file, generated JWT key, weak API key).
IS_PRODUCTION = (
    os.environ.get("ENVIRONMENT", "").strip().lower() in ("production", "prod")
    or bool(os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("RAILWAY_ENVIRONMENT_NAME"))
)

# --- Database URL ---
DATABASE_URL = os.environ.get("DATABASE_URL", "")
DATABASE_PUBLIC_URL = os.environ.get("DATABASE_PUBLIC_URL", "")
BACKUP_DATABASE_URL = os.environ.get("BACKUP_DATABASE_URL", "")

# IMPORTANT: Railway internal networking can fail. Always prefer PUBLIC URL for reliability.
# The internal URL (*.railway.internal) often times out due to network issues.
if DATABASE_PUBLIC_URL:
    print(f"[DB] Using DATABASE_PUBLIC_URL for reliable external connection")
    DATABASE_URL = DATABASE_PUBLIC_URL
elif DATABASE_URL:
    print(f"[DB] Using DATABASE_URL (internal network)")
else:
    print(f"[DB] WARNING: No database URL configured")


def _to_async_url(url):
    """Convert a postgres:// URL to postgresql+asyncpg:// for SQLAlchemy async."""
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://") and not url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("sqlite://") and not url.startswith("sqlite+aiosqlite://"):
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
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
    # LinkedIn OpenID Connect identity (`sub` claim) connected to this account, if any
    linkedin_sub = Column(String(255), unique=True, nullable=True, index=True)
    linkedin_email = Column(String(255), nullable=True)
    linkedin_linked_at = Column(DateTime(timezone=True), nullable=True)
    # Set when LinkedIn is disconnected; blocks LinkedIn sign-in from silently re-linking by email
    linkedin_disconnected_at = Column(DateTime(timezone=True), nullable=True)
    # False for accounts created via LinkedIn (random password hash) until the user sets one
    password_set = Column(Boolean, default=True, nullable=False, server_default="true")
    # Bumped to sign out every existing session (e.g. after a password reset); JWTs carry it as "tv"
    token_version = Column(Integer, default=0, nullable=False, server_default="0")

    # Relationships
    contacts = relationship("ContactDB", back_populates="user", cascade="all, delete-orphan")
    profile = relationship("UserProfileDB", back_populates="user", uselist=False, cascade="all, delete-orphan")


class PasswordResetDB(Base):
    """One-time password reset codes. Only a keyed hash of the code is stored."""
    __tablename__ = "password_resets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    code_hash = Column(String(128), nullable=False)
    attempts = Column(Integer, default=0, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ContactDB(Base):
    __tablename__ = "contacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    # Missing = NULL. Older rows may hold the "N/A" placeholder; contact_view.present() reads both as missing.
    email = Column(String(255), nullable=True)
    phone = Column(String(100), nullable=True)
    linkedin = Column(String(500), nullable=True)
    linkedin_source = Column(String(20), default="card")  # 'card' or 'ai_detected'
    company_name = Column(String(255), nullable=True)
    job_title = Column(String(255), nullable=True)  # contact's role, e.g. "Head of Data"
    notes = Column(Text, default="")
    links = Column(JSON, default=list)  # [{url, label, added_by}]
    source = Column(String(50), default="manual")  # 'manual', 'scan', 'qr'
    event_name = Column(String(255), default="")  # Event at which contact was scanned/added
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


class AdminContactScoreDB(Base):
    """The admin console's own AI score for a contact, kept apart from the rep's score.

    Scoring from the admin console (single, bulk, or the admin-run pipeline) writes here and
    never to ContactDB.lead_* — so the rep's score, notes and data in the app stay exactly as
    the rep left them. One row per contact (latest admin score). No foreign key on purpose:
    contacts are moved to deleted_contacts and back, and an admin score must never block that.
    """
    __tablename__ = "admin_contact_scores"

    contact_id = Column(UUID(as_uuid=True), primary_key=True)
    owner_id = Column(UUID(as_uuid=True), nullable=True)      # the rep whose profile it was scored against
    score = Column(Integer, nullable=False)
    temperature = Column(String(10), nullable=True)
    reasoning = Column(Text, default="")
    breakdown = Column(JSON, default=dict)                    # same shape as lead_score_breakdown (incl. "_meta")
    recommended_actions = Column(JSON, default=list)
    scored_by = Column(UUID(as_uuid=True), nullable=True)     # the admin who asked (None for pipeline runs)
    scored_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class AdminSalesProfileDB(Base):
    """The admin console's own "what we sell / how we sell" (products, pitch style, value
    propositions, objections, case studies) — what admin scoring, the admin-run pipeline and the
    admin assistant pitch and score against. Kept apart from every rep's user_profiles row: editing
    it never changes a rep's profile, and reps' edits never change it. One row, shared by all admins.
    """
    __tablename__ = "admin_sales_profile"

    id = Column(String(32), primary_key=True, default="console")
    data = Column(JSON, default=dict)                         # same keys/shape as the app profile's sales fields
    updated_by = Column(UUID(as_uuid=True), nullable=True)    # the admin who last saved it
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class CompanyIndustryDB(Base):
    """The industry the admin console worked out for a company — automatically, by AI, from the
    company name, its work-email domain and the job titles met there. Keyed by the normalised
    company name (or "@domain" when a contact has no company), so each company is classified once.
    Never written onto the rep's contact. industry=None = the model couldn't tell (retried later).
    """
    __tablename__ = "company_industries"

    company_key = Column(String(255), primary_key=True)
    company_name = Column(String(255), nullable=True)
    industry = Column(String(80), nullable=True)
    source = Column(String(20), default="ai")
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class DeletedContactDB(Base):
    """"Recently deleted" contacts, kept 30 days so they can be restored, then purged.

    The contact row itself is removed from `contacts` (so no screen or AI index can show
    it); a full snapshot lives here until it's restored or `purge_after` passes.
    """
    __tablename__ = "deleted_contacts"

    id = Column(UUID(as_uuid=True), primary_key=True)  # the original contact id
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    company_name = Column(String(255), nullable=True)
    snapshot = Column(JSON, nullable=False)        # every ContactDB column, JSON-safe
    files = Column(JSON, default=list)             # attached contact_files rows (data base64)
    reason = Column(String(255), nullable=True)    # e.g. "Merged into Maya Patel"; None = deleted
    deleted_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    purge_after = Column(DateTime(timezone=True), nullable=False, index=True)


class DuplicateDismissalDB(Base):
    """A user said two of their contacts are NOT duplicates; stop flagging that pair."""
    __tablename__ = "duplicate_dismissals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    contact_a = Column(UUID(as_uuid=True), nullable=False)  # stored with contact_a < contact_b
    contact_b = Column(UUID(as_uuid=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (Index("uq_duplicate_dismissal_pair", "user_id", "contact_a", "contact_b", unique=True),)


class CrmLinkDB(Base):
    """Which CRM record a contact became when an admin sent it to a CRM (Exports → Send to CRM), so
    sending again updates that record instead of creating a duplicate. No FK on contact_id: deleting
    or restoring a contact must not depend on this table (a stale link is just dropped on next send)."""
    __tablename__ = "crm_links"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    contact_id = Column(UUID(as_uuid=True), nullable=False)
    crm = Column(String(20), nullable=False)          # hubspot | salesforce | zoho | pipedrive
    external_id = Column(String(64), nullable=False)  # the record's id in that CRM
    synced_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (Index("uq_crm_link_contact", "crm", "contact_id", unique=True),)


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


class ContactShareLinkDB(Base):
    """A view-only link to one contact, showing only the fields the owner ticked.

    Live, not a snapshot: edits show up, and the link stops working when the contact is
    deleted, the link is revoked, or it expires.
    """
    __tablename__ = "contact_share_links"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    token = Column(String(64), unique=True, nullable=False, index=True)
    contact_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    fields = Column(JSON, nullable=False, default=list)  # e.g. ["title", "company", "email"]
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)  # None = never
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    view_count = Column(Integer, default=0, nullable=False, server_default="0")
    last_viewed_at = Column(DateTime(timezone=True), nullable=True)


class UserProfileDB(Base):
    __tablename__ = "user_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), unique=True, nullable=False)
    profile_data = Column(JSON, default=dict)  # Store full profile as JSON for flexibility
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    user = relationship("UserDB", back_populates="profile")


class ExhibitorDB(Base):
    """Exhibitor (booth) directory per event, for event intelligence."""
    __tablename__ = "exhibitors"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_name = Column(String(255), default="")  # every writer sets the real event
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
    # None = from the admin's directory import; a user id = a booth a rep added from the app.
    added_by = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_exhibitors_event_name", "event_name"),
        Index("idx_exhibitors_name", "name"),
        Index("idx_exhibitors_category", "category"),
    )


class BoothVisitDB(Base):
    """A booth a user marked as visited (Event page → Exhibitors → Mark visited).

    Keyed by event + booth name rather than the exhibitor row, so visits survive the admin
    re-importing the event's directory (which replaces the rows).
    """
    __tablename__ = "booth_visits"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    event_key = Column(String(255), nullable=False)  # lower-cased, single-spaced event name
    exhibitor_key = Column(String(500), nullable=False)  # lower-cased, single-spaced booth name
    note = Column(Text, default="")
    visited_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("uq_booth_visits_user_event_exhibitor", "user_id", "event_key", "exhibitor_key", unique=True),
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
    event_name = Column(String(255), default="")  # every writer sets the real event
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
    gamma_deck_url = Column(Text, nullable=True)  # Gamma shareable presentation URL
    # Started from the admin console: research / pitch made for the ADMIN's sales profile. Never shown
    # in the rep's app or export (admin/rep isolation).
    admin_run = Column(Boolean, default=False, nullable=False, server_default="false")

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
    recipient_ids = Column(JSON, nullable=True)  # None = all team; list of user UUID strings = targeted
    is_active = Column(Boolean, default=True, nullable=False, server_default="true")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    ended_at = Column(DateTime(timezone=True), nullable=True)   # when an admin ended it (None = still live, or ended before this was recorded)
    event_id = Column(UUID(as_uuid=True), nullable=True)        # the planned event it's about (managed_events.id), if any


class ManagedEventDB(Base):
    """An event the admins track and plan (Admin console → Announcements → Events).

    `name` is the same event name used everywhere else (contacts' "met at", the booth directory,
    reps' event lists, event files — floor plans are event files with category "floorplan"), so a
    planned event joins up with all of that data. `source` is "manual" today; discovered events
    (a web scraper, once one is connected) will arrive as "discovery" with status "suggested".
    """
    __tablename__ = "managed_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    status = Column(String(20), nullable=False, default="planned")  # suggested · interested · planned · attending · done · dismissed
    start_date = Column(String(10), nullable=True)   # YYYY-MM-DD (the event's own local dates)
    end_date = Column(String(10), nullable=True)
    venue = Column(String(255), default="")
    address = Column(String(500), default="")
    city = Column(String(120), default="")
    country = Column(String(120), default="")
    website = Column(String(500), default="")
    organizer = Column(String(255), default="")
    industry = Column(String(255), default="")
    description = Column(Text, default="")
    # anything else worth knowing: {"hours", "registration", "tickets", "booth", "contactName", "contactEmail", "contactPhone", "notes"}
    details = Column(JSON, default=dict)
    source = Column(String(20), nullable=False, default="manual")
    source_url = Column(String(500), default="")
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    __table_args__ = (Index("idx_managed_events_status", "status"),)


class EventAssignmentDB(Base):
    """Who's going to a managed event (the team), and who leads it."""
    __tablename__ = "event_assignments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_id = Column(UUID(as_uuid=True), ForeignKey("managed_events.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False, default="member")  # "lead" | "member"
    assigned_by = Column(UUID(as_uuid=True), nullable=True)
    assigned_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (Index("idx_event_assignment_unique", "event_id", "user_id", unique=True),)


class EventNoteDB(Base):
    """A rep's own note / reminder for an event (app → Me → Calendar). Private to that rep.

    Keyed by event name like the rest of the event data; an admin rename of a planned event
    renames its notes too. `date` pins the note to a calendar day; `remind_at` makes it a reminder.
    """
    __tablename__ = "event_notes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    event_name = Column(String(255), nullable=False, default="")
    date = Column(String(10), nullable=True)  # YYYY-MM-DD
    text = Column(Text, nullable=False, default="")
    remind_at = Column(DateTime(timezone=True), nullable=True)
    done = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class EmailVerificationDB(Base):
    """Latest verification of an email address (email_verify.py — the check-if-email-exists checks).

    Keyed by the lower-cased address: whether a mailbox exists is a fact about the address, not
    about who asked, so one check serves every contact with that email.
    """
    __tablename__ = "email_verifications"

    email = Column(String(320), primary_key=True)
    is_reachable = Column(String(10), nullable=False, default="unknown")  # safe · risky · invalid · unknown
    result = Column(JSON, default=dict)
    checked_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ContactListDB(Base):
    """Named contact lists created by admins for grouping/targeting."""
    __tablename__ = "contact_lists"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    admin_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    members = relationship("ContactListMemberDB", back_populates="contact_list", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_contact_lists_admin_id", "admin_id"),
    )


class ContactListMemberDB(Base):
    """Members of a contact list."""
    __tablename__ = "contact_list_members"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    list_id = Column(UUID(as_uuid=True), ForeignKey("contact_lists.id"), nullable=False)
    contact_id = Column(UUID(as_uuid=True), ForeignKey("contacts.id"), nullable=False)
    added_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    contact_list = relationship("ContactListDB", back_populates="members")

    __table_args__ = (
        Index("idx_contact_list_members_list_id", "list_id"),
        Index("idx_contact_list_members_contact_id", "contact_id"),
    )


# --- Engine & Session (Primary) ---
engine = None
async_session_factory = None

# --- Engine & Session (Backup) ---
backup_engine = None
backup_session_factory = None


def _create_engine(url):
    """Create an async engine with resilience settings."""
    if "sqlite" in url:
        return create_async_engine(url, echo=False)
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
            # SQL command timeout (seconds). Backups, file uploads and the admin aggregates need more than 5 s.
            "command_timeout": int(os.environ.get("DB_COMMAND_TIMEOUT", "30")),
        }
    )


def get_engine():
    global engine
    if engine is None:
        if not ASYNC_DATABASE_URL:
            if IS_PRODUCTION:
                # A container's disk is wiped on redeploy: never put real data in a throwaway dev file.
                raise RuntimeError("DATABASE_URL is not set. Production needs a persistent database (Postgres).")
            local_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "eventscout_dev.db"))
            sqlite_url = f"sqlite+aiosqlite:///{local_db_path.replace(os.sep, '/')}"
            print(f"[DB] DATABASE_URL not set - using local SQLite fallback: {sqlite_url}")
            try:
                engine = _create_engine(sqlite_url)
                print(f"[DB] Local SQLite engine created successfully")
            except Exception as e:
                print(f"[DB] Failed to create SQLite engine: {e}")
                engine = None
            return engine
        if not (ASYNC_DATABASE_URL.startswith("postgresql+asyncpg://") or ASYNC_DATABASE_URL.startswith("sqlite+aiosqlite://")):
            print(f"[DB] Invalid DATABASE_URL format (must start with postgresql+asyncpg:// or sqlite+aiosqlite://)")
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
        is_sqlite = "sqlite" in str(eng.url)
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

            if not is_sqlite:
                migrations = [
                    "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT FALSE NOT NULL",
                    "ALTER TABLE contacts ADD COLUMN IF NOT EXISTS audio_notes JSONB DEFAULT '[]'::jsonb",
                    "ALTER TABLE contacts ADD COLUMN IF NOT EXISTS admin_notes TEXT DEFAULT ''",
                    "ALTER TABLE contacts ADD COLUMN IF NOT EXISTS photo_base64 TEXT",
                    "ALTER TABLE contacts ADD COLUMN IF NOT EXISTS event_name VARCHAR(255) DEFAULT ''",
                    "ALTER TABLE contact_pipelines ADD COLUMN IF NOT EXISTS gamma_deck_url TEXT",
                    "ALTER TABLE admin_broadcasts ADD COLUMN IF NOT EXISTS recipient_ids JSONB",
                    "ALTER TABLE users ADD COLUMN IF NOT EXISTS linkedin_sub VARCHAR(255)",
                    "ALTER TABLE users ADD COLUMN IF NOT EXISTS linkedin_email VARCHAR(255)",
                    "ALTER TABLE users ADD COLUMN IF NOT EXISTS linkedin_linked_at TIMESTAMP WITH TIME ZONE",
                    "CREATE UNIQUE INDEX IF NOT EXISTS ix_users_linkedin_sub ON users (linkedin_sub)",
                    "ALTER TABLE users ADD COLUMN IF NOT EXISTS linkedin_disconnected_at TIMESTAMP WITH TIME ZONE",
                    "ALTER TABLE users ADD COLUMN IF NOT EXISTS password_set BOOLEAN DEFAULT TRUE NOT NULL",
                    "ALTER TABLE users ADD COLUMN IF NOT EXISTS token_version INTEGER DEFAULT 0 NOT NULL",
                    "ALTER TABLE contacts ADD COLUMN IF NOT EXISTS job_title VARCHAR(255)",
                    "ALTER TABLE deleted_contacts ADD COLUMN IF NOT EXISTS reason VARCHAR(255)",
                    "ALTER TABLE exhibitors ADD COLUMN IF NOT EXISTS added_by UUID",
                    "ALTER TABLE admin_broadcasts ADD COLUMN IF NOT EXISTS ended_at TIMESTAMP WITH TIME ZONE",
                    "ALTER TABLE admin_broadcasts ADD COLUMN IF NOT EXISTS event_id UUID",
                    "ALTER TABLE contact_pipelines ADD COLUMN IF NOT EXISTS admin_run BOOLEAN DEFAULT FALSE NOT NULL",
                ]
                title_col = await conn.execute(text(
                    "SELECT COUNT(*) FROM information_schema.columns "
                    "WHERE table_name = 'contacts' AND column_name = 'job_title'"
                ))
                title_added = (title_col.scalar() or 0) == 0
                for sql in migrations:
                    try:
                        await conn.execute(text(sql))
                    except Exception as mig_err:
                        print(f"[DB-{label}] Migration note: {mig_err}")
            else:
                # SQLite has no ADD COLUMN IF NOT EXISTS; skip columns that already exist.
                existing = {row[1] for row in (await conn.execute(text("PRAGMA table_info(users)"))).fetchall()}
                sqlite_migrations = [
                    ("linkedin_sub", "ALTER TABLE users ADD COLUMN linkedin_sub VARCHAR(255)"),
                    ("linkedin_email", "ALTER TABLE users ADD COLUMN linkedin_email VARCHAR(255)"),
                    ("linkedin_linked_at", "ALTER TABLE users ADD COLUMN linkedin_linked_at DATETIME"),
                    ("linkedin_disconnected_at", "ALTER TABLE users ADD COLUMN linkedin_disconnected_at DATETIME"),
                    ("password_set", "ALTER TABLE users ADD COLUMN password_set BOOLEAN NOT NULL DEFAULT 1"),
                    ("token_version", "ALTER TABLE users ADD COLUMN token_version INTEGER NOT NULL DEFAULT 0"),
                ]
                for column, sql in sqlite_migrations:
                    if column not in existing:
                        await conn.execute(text(sql))
                await conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_linkedin_sub ON users (linkedin_sub)"))

                contact_cols = {row[1] for row in (await conn.execute(text("PRAGMA table_info(contacts)"))).fetchall()}
                title_added = bool(contact_cols) and "job_title" not in contact_cols
                if title_added:
                    await conn.execute(text("ALTER TABLE contacts ADD COLUMN job_title VARCHAR(255)"))

                archive_cols = {row[1] for row in (await conn.execute(text("PRAGMA table_info(deleted_contacts)"))).fetchall()}
                if archive_cols and "reason" not in archive_cols:
                    await conn.execute(text("ALTER TABLE deleted_contacts ADD COLUMN reason VARCHAR(255)"))

                exhibitor_cols = {row[1] for row in (await conn.execute(text("PRAGMA table_info(exhibitors)"))).fetchall()}
                if exhibitor_cols and "added_by" not in exhibitor_cols:
                    await conn.execute(text("ALTER TABLE exhibitors ADD COLUMN added_by CHAR(32)"))

                broadcast_cols = {row[1] for row in (await conn.execute(text("PRAGMA table_info(admin_broadcasts)"))).fetchall()}
                if broadcast_cols and "ended_at" not in broadcast_cols:
                    await conn.execute(text("ALTER TABLE admin_broadcasts ADD COLUMN ended_at DATETIME"))
                if broadcast_cols and "event_id" not in broadcast_cols:
                    await conn.execute(text("ALTER TABLE admin_broadcasts ADD COLUMN event_id CHAR(32)"))

                pipeline_cols = {row[1] for row in (await conn.execute(text("PRAGMA table_info(contact_pipelines)"))).fetchall()}
                if pipeline_cols and "admin_run" not in pipeline_cols:
                    await conn.execute(text("ALTER TABLE contact_pipelines ADD COLUMN admin_run BOOLEAN NOT NULL DEFAULT 0"))

            # One-time backfill, only when job_title was just added to an existing table:
            # titles used to be stored in admin_notes (which admins also use for notes), so copy
            # short single-line values over. admin_notes itself is left untouched.
            if title_added:
                newline_free = (
                    "instr(admin_notes, char(10)) = 0" if is_sqlite else "POSITION(CHR(10) IN admin_notes) = 0"
                )
                try:
                    async with conn.begin_nested():
                        await conn.execute(text(
                            "UPDATE contacts SET job_title = admin_notes "
                            "WHERE job_title IS NULL AND admin_notes IS NOT NULL AND admin_notes <> '' "
                            f"AND LENGTH(admin_notes) <= 120 AND {newline_free}"
                        ))
                except Exception as mig_err:
                    print(f"[DB-{label}] Title backfill note: {mig_err}")

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
