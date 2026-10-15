"""Unit tests for the backend's deep modules, through their interfaces, on a throwaway SQLite database.

app.py is never imported (it loads the embedding model); the side-effect / AI adapters are fakes.
Run from apps/backend:  python -m pytest tests_unit -q
"""

import os
import sys
import uuid

# Before any backend module loads: a fixed signing key, no real database, no .env surprises.
os.environ.setdefault("JWT_SECRET", "unit-test-secret-" + "x" * 40)
os.environ["DATABASE_URL"] = ""
os.environ["DATABASE_PUBLIC_URL"] = ""
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import database  # noqa: E402
import contact_store  # noqa: E402


@pytest_asyncio.fixture
async def db():
    """A fresh in-memory SQLite database with every table; database.get_session_factory() returns it.
    (One shared connection, so every session sees the same database.)"""
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(database.Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    old = (database.engine, database.async_session_factory)
    database.engine, database.async_session_factory = engine, factory
    try:
        yield factory
    finally:
        database.engine, database.async_session_factory = old
        await engine.dispose()


class FakeEffects:
    """Records every side effect contact_store sends (search index, live feed, webhook)."""

    def __init__(self):
        self.indexed, self.unindexed, self.live, self.webhooks = [], [], [], []

    def index_update(self, user_id, doc):
        self.indexed.append((user_id, doc))

    def index_delete(self, user_id, contact_id):
        self.unindexed.append((user_id, contact_id))

    async def live_event(self, event_type, data):
        self.live.append((event_type, data))

    async def webhook(self, contact):
        self.webhooks.append(str(contact.id))

    def events(self):
        return [t for t, _ in self.live]


@pytest.fixture
def effects():
    fake = FakeEffects()
    e = contact_store.effects
    old = (e.index_update, e.index_delete, e.live, e.webhook)
    e.index_update, e.index_delete, e.live, e.webhook = fake.index_update, fake.index_delete, fake.live_event, fake.webhook
    try:
        yield fake
    finally:
        e.index_update, e.index_delete, e.live, e.webhook = old


async def make_user(factory, name="Rep", admin=False, token_version=0) -> uuid.UUID:
    async with factory() as s:
        uid = uuid.uuid4()
        s.add(database.UserDB(id=uid, name=name, email=f"{uid.hex[:8]}@example.com", password_hash="x",
                              is_admin=admin, token_version=token_version))
        await s.commit()
        return uid
