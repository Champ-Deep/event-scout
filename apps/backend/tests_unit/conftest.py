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
from sqlalchemy.pool import NullPool  # noqa: E402

import database  # noqa: E402
import contact_store  # noqa: E402


@pytest_asyncio.fixture
async def db(tmp_path):
    """A fresh SQLite database with every table; database.get_session_factory() returns it."""
    # Separate connections keep overlapping route and background sessions' transactions isolated.
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}", poolclass=NullPool)
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


def real_pages(site: str):
    """The pages event_scraper.crawl really fetched for `site` (fixtures/real/manifest.json has URL + time + sha256)."""
    import gzip
    import json
    import pathlib

    import event_scraper
    real = pathlib.Path(__file__).parent / "fixtures" / "real"
    return [event_scraper.FetchedPage(url=p["final_url"], final_url=p["final_url"], status=200, content_type="text/html",
                                      html=gzip.decompress((real / p["file"]).read_bytes()).decode("utf-8", "replace"))
            for p in json.loads((real / "manifest.json").read_text())[site]["pages"]]


def real_start_url(site: str) -> str:
    import json
    import pathlib
    return json.loads((pathlib.Path(__file__).parent / "fixtures" / "real" / "manifest.json").read_text())[site]["start_url"]
