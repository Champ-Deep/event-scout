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
    ForeignKey, Index, JSON, create_engine, text
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, relationship


# --- Database URL ---
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Convert postgres:// to postgresql+asyncpg:// for SQLAlchemy async
if DATABASE_URL.startswith("postgres://"):
    ASYNC_DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
else:
    ASYNC_DATABASE_URL = DATABASE_URL

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


# --- Engine & Session ---
engine = None
async_session_factory = None


def get_engine():
    global engine
    if engine is None and ASYNC_DATABASE_URL:
        engine = create_async_engine(
            ASYNC_DATABASE_URL,
            echo=False,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    return engine


def get_session_factory():
    global async_session_factory
    if async_session_factory is None:
        eng = get_engine()
        if eng:
            async_session_factory = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False)
    return async_session_factory


async def init_db():
    """Create all tables if they don't exist."""
    eng = get_engine()
    if eng is None:
        print("[DB] No DATABASE_URL configured - skipping database init")
        return False

    try:
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

            # Migrations: add columns that may not exist on older databases
            migrations = [
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT FALSE NOT NULL",
                "ALTER TABLE contacts ADD COLUMN IF NOT EXISTS audio_notes JSONB DEFAULT '[]'::jsonb",
            ]
            for sql in migrations:
                try:
                    await conn.execute(text(sql))
                except Exception as mig_err:
                    print(f"[DB] Migration note: {mig_err}")

        print("[DB] Database tables created/verified successfully")
        return True
    except Exception as e:
        print(f"[DB] Error initializing database: {e}")
        return False


async def get_db() -> AsyncSession:
    """Get an async database session."""
    factory = get_session_factory()
    if factory is None:
        raise Exception("Database not configured")
    async with factory() as session:
        yield session
