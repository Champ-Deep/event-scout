"""
EventScout Unified REST & SSE API Router.
Connects Admin Ops Console (apps/admin) and Mobile Field Client (apps/mobile)
with the backend database, scoring engine, and live SSE event stream.
"""

import os
import re
from urllib.parse import urlparse
import uuid
import io
import json
import base64
import asyncio
from io import BytesIO
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

import qrcode
from qrcode.image.pil import PilImage
from fastapi import APIRouter, HTTPException, Query, Request, Body, Depends, Header, File, Form, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select, update, delete, func, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from database import (
    UserDB, ContactDB, SharedContactDB, UserProfileDB, ConversationDB, ExhibitorDB,
    UserCardDB, EventFileDB, ContactFileDB, ContactPipelineDB, AdminBroadcastDB, AdminContactScoreDB, CompanyIndustryDB,
    AdminSalesProfileDB, ContactListDB, ContactListMemberDB, DeletedContactDB, DuplicateDismissalDB, ContactShareLinkDB, BoothVisitDB,
    ManagedEventDB, EventAssignmentDB, EventNoteDB, EmailVerificationDB, get_session_factory
)

from sanitize import safe_csv_writer  # noqa: E402
import identity
import event_identity
import contact_view
import contact_store
import scoring
from contact_store import process_contact_photo, from_snapshot as _from_snapshot, sync_index as sync_search_index, RECENTLY_DELETED_DAYS
from contact_view import (format_contact, app_view, console_scores, console_contact, console_matches, admin_score_out,
                          present as _present, aware as _aware, iso as _iso, lite as _lite)
from event_identity import event_key as _key, event_filter, profile_events as _profile_events  # booth names compare like events
from score_levels import level_for, band_for, is_hot, is_hot_or_warm, level_case, level_condition, band_condition, WARM_MIN

api_router = APIRouter(prefix="/api", tags=["EventScout API"])

# --- SSE Event Stream Subscribers ---
sse_subscribers: List[asyncio.Queue] = []
SSE_QUEUE_MAX = 500  # per listener; a stalled client loses its oldest events instead of growing memory forever


def new_sse_queue() -> asyncio.Queue:
    return asyncio.Queue(maxsize=SSE_QUEUE_MAX)


# ---- Real implementations injected by app.py (it imports this module; importing back would be
# circular). Each admin route that needs one answers 503 with the reason when it isn't set.
pipeline_impl = None          # async (contact_id, user_id, admin=False) -> None   (research → score → pitch → deck)

# Files only the admin console sees (decks from admin-run pipelines). Never listed or served to the app.
ADMIN_FILE_CATEGORY = "admin_pitch"
pipeline_settings_impl = None  # () -> {autoPipelineEnabled, aiConfigured, gammaConfigured}
backup_impl = None            # async () -> dict   (copy to BACKUP_DATABASE_URL)
backup_configured_impl = None  # () -> bool
hash_password_impl = None     # (password) -> bcrypt hash
models_impl = None            # () -> [{id, name, provider, role, default}]
webhook_url_impl = None       # () -> str   (WEBHOOK_URL)
last_backup_impl = None       # () -> dict  (run_backup's last result: startup / periodic / console)
system_info_impl = None       # () -> {version, aiConfigured, webhookConfigured, backupIntervalHours}
delete_user_index_impl = None  # (user_id) -> None   (FAISS)
_STARTED_AT = datetime.now(timezone.utc)  # when this server process started (System page "up for")


def _need(impl, what: str):
    if impl is None:
        raise HTTPException(status_code=503, detail=f"{what} is not available on this server")
    return impl


async def broadcast_live_event(event_type: str, data: dict):
    """Broadcast an event to all connected admin SSE clients."""
    msg = {
        "type": event_type,
        "data": data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    for q in list(sse_subscribers):
        try:
            if q.full():
                q.get_nowait()  # drop the oldest (the console re-syncs everything when it reconnects)
            q.put_nowait(msg)
        except Exception:
            pass
    # a new or edited contact may bring a company nobody has classified yet: sort it into an industry
    # in the background right away (no-op when one is running, the AI is off, or nothing is new)
    if event_type in ("contact_scanned", "score_updated", "contact_restored"):
        try:
            import ai_client
            if ai_client.ai_configured:
                kick_industry_classification()
        except Exception as e:  # never let this break the request that sent the event
            print(f"[INDUSTRY] couldn't start classification: {e}")


contact_store.effects.live = broadcast_live_event  # every contact write announces itself here


async def get_db():
    factory = get_session_factory()
    if not factory:
        raise HTTPException(status_code=503, detail="Database session unavailable")
    async with factory() as session:
        yield session


# ==================== REQUEST IDENTITY & SEARCH INDEX HOOKS ====================
# Who is acting lives in identity.py; these are the FastAPI dependencies the routes use (thin shims).
# The search index hooks every contact write needs live in contact_store.effects (app.py wires them).


async def require_admin(authorization: Optional[str] = Header(None)) -> uuid.UUID:
    """The signed-in admin. Every admin-console-only route depends on this."""
    return await identity.admin_user(authorization)


async def require_admin_stream(
    authorization: Optional[str] = Header(None),
    ticket: Optional[str] = Query(None),
    access_token: Optional[str] = Query(None),
) -> uuid.UUID:
    """Same as require_admin for the SSE feed: a header, a `?ticket=` from POST /api/admin/live/ticket,
    or (older consoles) `?access_token=`."""
    return await identity.stream_admin(authorization, ticket, access_token)


async def request_user_id(
    authorization: Optional[str] = Header(None),
    x_admin_console: Optional[str] = Header(None),
) -> Optional[uuid.UUID]:
    """Whose data the request works on (identity.acting(...).owner): the signed-in rep's id, or None for
    the admin console (`X-Admin-Console: 1` with an admin's token — the cross-user view)."""
    return (await identity.acting(authorization, console=bool(x_admin_console))).owner


def ensure_owner(contact: "ContactDB", user_id: Optional[uuid.UUID]) -> None:
    # 404 rather than 403 so other users' contact IDs can't be probed.
    if user_id is not None and contact.user_id != user_id:
        raise HTTPException(status_code=404, detail="Contact not found")


async def load_contact(session: AsyncSession, contact_id: str, user_id: Optional[uuid.UUID]) -> "ContactDB":
    try:
        cid = uuid.UUID(contact_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid contact ID format")
    result = await session.execute(select(ContactDB).where(ContactDB.id == cid))
    contact = result.scalar_one_or_none()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")
    ensure_owner(contact, user_id)
    return contact


# Admin-console aggregates (dashboard, events, team, segments, exports) count by the ADMIN score — the
# same one the console's contacts grid shows, filters and sorts by — never the rep's ContactDB.lead_score.
ADMIN_SCORE = AdminContactScoreDB.score


def _with_admin_score(query):
    """Outer-join the admin score onto a query over contacts (contacts without one = unscored).
    select_from pins contacts as the left side (needed when the select starts with count() / the score)."""
    return query.select_from(ContactDB).outerjoin(AdminContactScoreDB, AdminContactScoreDB.contact_id == ContactDB.id)


# ==================== SEED INITIAL DATA ====================

async def seed_initial_data_if_empty():
    """Seed sample reps and contacts for local testing if the database is empty.

    Disabled by default. Set SEED_DEMO_DATA=true to opt in (local/demo only) — otherwise
    a fresh deployment starts with a genuinely empty database instead of fabricated leads.
    """
    if os.environ.get("SEED_DEMO_DATA", "false").strip().lower() not in ("true", "1", "yes"):
        print("[SEED] Demo seeding disabled (set SEED_DEMO_DATA=true to enable).")
        return
    from database import IS_PRODUCTION
    if IS_PRODUCTION:
        print("[SEED] SEED_DEMO_DATA ignored in production — fake leads never go into a real workspace.")
        return

    factory = get_session_factory()
    if not factory:
        return
    try:
        async with factory() as session:
            # Check users
            res = await session.execute(select(func.count(UserDB.id)))
            user_count = res.scalar() or 0
            admin_user_id = None

            if user_count == 0:
                admin_user_id = uuid.uuid4()
                admin_user = UserDB(
                    id=admin_user_id,
                    name="Sarah Connor",
                    email="lakeb2b.com",
                    password_hash="pbkdf2_sha256$local$hash",
                    is_admin=True,
                    created_at=datetime.now(timezone.utc),
                )
                session.add(admin_user)
                await session.commit()
            else:
                user_res = await session.execute(select(UserDB).limit(1))
                u = user_res.scalar_one_or_none()
                if u:
                    admin_user_id = u.id

            # Check contacts
            c_res = await session.execute(select(func.count(ContactDB.id)))
            contact_count = c_res.scalar() or 0

            if contact_count == 0 and admin_user_id:
                sample_leads = [
                    {
                        "name": "Alex Mercer",
                        "title": "VP of Engineering",
                        "company": "Stripe Inc.",
                        "email": "alex@stripe.com",
                        "phone": "+1 (555) 234-5678",
                        "score": 92,
                        "temp": "hot",
                        "notes": "VP of Engineering | High buying intent for enterprise data sync",
                    },
                    {
                        "name": "Elena Rostova",
                        "title": "Director of Product",
                        "company": "Datadog",
                        "email": "elena.r@datadog.com",
                        "phone": "+1 (555) 345-6789",
                        "score": 88,
                        "temp": "hot",
                        "notes": "Director of Product | Interested in booth lead scoring automation",
                    },
                    {
                        "name": "Marcus Vance",
                        "title": "Head of Infrastructure",
                        "company": "Snowflake",
                        "email": "mvance@snowflake.com",
                        "phone": "+1 (555) 456-7890",
                        "score": 78,
                        "temp": "warm",
                        "notes": "Head of Infrastructure | Evaluating field rep offline sync",
                    },
                    {
                        "name": "Chloe Bennett",
                        "title": "Sr Partnerships Manager",
                        "company": "Figma",
                        "email": "chloe@figma.com",
                        "phone": "+1 (555) 567-8901",
                        "score": 65,
                        "temp": "warm",
                        "notes": "Sr Partnerships Manager | Potential co-marketing integration",
                    },
                    {
                        "name": "David Kim",
                        "title": "Founder & CEO",
                        "company": "Nexus AI",
                        "email": "dkim@nexusai.io",
                        "phone": "+1 (555) 678-9012",
                        "score": 96,
                        "temp": "hot",
                        "notes": "Founder & CEO | Requested executive demo and custom pricing",
                    },
                    {
                        "name": "Samantha Reed",
                        "title": "Procurement Specialist",
                        "company": "Global Logistics Corp",
                        "email": "s.reed@globallogistics.com",
                        "phone": "+1 (555) 789-0123",
                        "score": 48,
                        "temp": "cold",
                        "notes": "Procurement Specialist | Looking for generic conference RFP",
                    },
                    {
                        "name": "Michael Thorne",
                        "title": "VP Sales Strategy",
                        "company": "Atlassian",
                        "email": "mthorne@atlassian.com",
                        "phone": "+1 (555) 890-1234",
                        "score": 84,
                        "temp": "hot",
                        "notes": "VP Sales Strategy | Wants team pilot for Q4 flagship event",
                    },
                    {
                        "name": "Linda Zhang",
                        "title": "Principal Architect",
                        "company": "Cloudflare",
                        "email": "lzhang@cloudflare.com",
                        "phone": "+1 (555) 901-2345",
                        "score": 91,
                        "temp": "hot",
                        "notes": "Principal Architect | Interested in sub-50ms offline OCR indexing",
                    },
                ]

                now = datetime.now(timezone.utc)
                for item in sample_leads:
                    contact = ContactDB(
                        id=uuid.uuid4(),
                        user_id=admin_user_id,
                        name=item["name"],
                        email=item["email"],
                        phone=item["phone"],
                        company_name=item["company"],
                        event_name="WHX 2026",
                        notes=item["notes"],
                        admin_notes=item["title"],
                        source="scan",
                        lead_score=item["score"],
                        lead_temperature=item["temp"],
                        created_at=now - timedelta(minutes=len(sample_leads) * 10),
                    )
                    session.add(contact)
                await session.commit()
                print(f"[API SEED] Seeded {len(sample_leads)} demo contacts.")

            # Check exhibitors
            ex_res = await session.execute(select(func.count(ExhibitorDB.id)))
            ex_count = ex_res.scalar() or 0
            if ex_count == 0:
                sample_exhibitors = [
                    {"name": "BioHealth AI Labs", "booth": "B-104", "hall": "Hall B", "category": "AI & Diagnostics", "country": "United States", "website": "https://biohealthai.example.com", "description": "Edge AI for real-time surgical computer vision and pathology analytics.", "tags": ["AI", "Pathology", "Enterprise"]},
                    {"name": "NeuroPulse Dynamics", "booth": "B-218", "hall": "Hall B", "category": "Medical Devices", "country": "Germany", "website": "https://neuropulse.example.de", "description": "Non-invasive neural telemetry and wireless ICU patient monitoring headsets.", "tags": ["Hardware", "ICU", "Neurology"]},
                    {"name": "Genomix Cloud", "booth": "A-012", "hall": "Hall A", "category": "Genomics & Biotech", "country": "Switzerland", "website": "https://genomixcloud.example.ch", "description": "Federated clinical sequencing infrastructure and variant annotation engine.", "tags": ["Sequencing", "Cloud", "HIPAA"]},
                    {"name": "CareSync Telehealth", "booth": "C-305", "hall": "Hall C", "category": "Digital Health", "country": "Singapore", "website": "https://caresync.example.sg", "description": "Omnichannel virtual clinic platform with automated EHR note generation.", "tags": ["Telehealth", "SaaS", "EHR"]},
                    {"name": "Apex Surgical Robotics", "booth": "B-410", "hall": "Hall B", "category": "Robotics", "country": "Japan", "website": "https://apexsurgical.example.jp", "description": "Sub-millimeter micro-manipulator robots for minimally invasive cardiovascular procedures.", "tags": ["Robotics", "Hardware", "Cardiology"]},
                    {"name": "PharmaTrack IoT", "booth": "C-112", "hall": "Hall C", "category": "Supply Chain & Logistics", "country": "United Kingdom", "website": "https://pharmatrack.example.co.uk", "description": "Cold-chain continuous temperature sensors and blockchain custody tracking.", "tags": ["IoT", "ColdChain", "Logistics"]},
                ]
                for ex in sample_exhibitors:
                    session.add(ExhibitorDB(
                        id=uuid.uuid4(),
                        name=ex["name"],
                        booth=ex["booth"],
                        hall=ex["hall"],
                        category=ex["category"],
                        country=ex["country"],
                        website=ex["website"],
                        description=ex["description"],
                        tags=ex["tags"],
                        event_name="WHX 2026",
                        created_at=datetime.now(timezone.utc),
                    ))
                await session.commit()
                print(f"[API SEED] Seeded {len(sample_exhibitors)} demo exhibitors.")

            # Check broadcasts
            bc_res = await session.execute(select(func.count(AdminBroadcastDB.id)))
            bc_count = bc_res.scalar() or 0
            if bc_count == 0 and admin_user_id:
                sample_broadcasts = [
                    {"message": "🚨 High Priority: VIP Delegation from Mayo Clinic arriving at Booth B-104 at 2:30 PM. All reps in Hall B prep Tier A collateral.", "priority": "urgent"},
                    {"message": "📢 Keynote stage session starts in 20 minutes in Auditorium 1. Ensure all leads scanned in Morning Session are synced.", "priority": "normal"},
                    {"message": "⚡ Lead velocity update: Team has crossed 1,200 verified scans! Top rep leaderboard will be announced at 5 PM.", "priority": "normal"},
                ]
                for bc in sample_broadcasts:
                    session.add(AdminBroadcastDB(
                        id=uuid.uuid4(),
                        admin_id=admin_user_id,
                        message=bc["message"],
                        priority=bc["priority"],
                        is_active=True,
                        created_at=datetime.now(timezone.utc),
                    ))
                await session.commit()
                print(f"[API SEED] Seeded {len(sample_broadcasts)} demo broadcasts.")

            # Check files
            ef_res = await session.execute(select(func.count(EventFileDB.id)))
            ef_count = ef_res.scalar() or 0
            if ef_count == 0 and admin_user_id:
                sample_files = [
                    {"filename": "WHX_2026_Official_Floor_Plan.pdf", "file_type": "pdf", "category": "general", "size": 3480000, "desc": "Official Hall A, B, C floor plan and booth layout map."},
                    {"filename": "EventScout_Enterprise_Overview_2026.pdf", "file_type": "pdf", "category": "brochure", "size": 1820000, "desc": "Executive 2-page sales collateral on offline AI lead capture."},
                    {"filename": "Q4_Event_Pricing_Rate_Card.xlsx", "file_type": "other", "category": "pricing", "size": 420000, "desc": "Tiered event team seat licensing and pilot package pricing."},
                    {"filename": "Technical_Architecture_Offline_Sync.pptx", "file_type": "pptx", "category": "catalog", "size": 5600000, "desc": "Deep-dive presentation on sub-50ms local CRDT sync engine."},
                ]
                for f in sample_files:
                    session.add(EventFileDB(
                        id=uuid.uuid4(),
                        filename=f["filename"],
                        original_filename=f["filename"],
                        file_type=f["file_type"],
                        mime_type="application/octet-stream",
                        file_size=f["size"],
                        file_data=b"Mock file binary payload for testing",
                        description=f["desc"],
                        event_name="WHX 2026",
                        category=f["category"],
                        uploaded_by=admin_user_id,
                        download_count=12,
                        created_at=datetime.now(timezone.utc),
                    ))
                await session.commit()
                print(f"[API SEED] Seeded {len(sample_files)} demo files.")

            # Check contact lists
            cl_res = await session.execute(select(func.count(ContactListDB.id)))
            cl_count = cl_res.scalar() or 0
            if cl_count == 0 and admin_user_id:
                sample_lists = [
                    {"name": "VIP Enterprise Accounts"},
                    {"name": "Day 1 Follow-ups (Hot Intent)"},
                    {"name": "Digital Health & Telehealth Targets"},
                ]
                for lst in sample_lists:
                    session.add(ContactListDB(
                        id=uuid.uuid4(),
                        name=lst["name"],
                        admin_id=admin_user_id,
                        created_at=datetime.now(timezone.utc),
                    ))
                await session.commit()
                print(f"[API SEED] Seeded {len(sample_lists)} demo contact lists.")
    except Exception as e:
        print(f"[API SEED] Error seeding data: {e}")


# ==================== ENDPOINTS ====================

@api_router.get("/health")
async def api_health():
    """Backend and database status. The database is really queried (SELECT 1, 3 s limit), so a
    database that is down shows as down; status is "degraded" then."""
    factory = get_session_factory()
    db_ok, db_ms, db_error = False, None, None
    if factory is not None:
        import time as _t
        from sqlalchemy import text as _sql
        t0 = _t.perf_counter()
        try:
            async def _ping():
                async with factory() as s:
                    await s.execute(_sql("SELECT 1"))
            await asyncio.wait_for(_ping(), timeout=3)
            db_ok, db_ms = True, round((_t.perf_counter() - t0) * 1000)
        except asyncio.TimeoutError:
            db_error = "The database didn't answer within 3 s"
        except Exception as e:
            db_error = f"{e.__class__.__name__}: {str(e)[:160]}"
    else:
        db_error = "No database is configured (DATABASE_URL)"
    from database import DATABASE_URL as _db_url
    info = system_info_impl() if system_info_impl else {}
    now = datetime.now(timezone.utc)
    return {
        "status": "healthy" if db_ok else "degraded",
        "api": "EventScout Backend API",
        "databaseReady": db_ok,
        "databaseLatencyMs": db_ms,
        "databaseError": db_error,
        "databaseKind": "SQLite" if _db_url and "sqlite" in _db_url else ("PostgreSQL" if _db_url else None),
        "version": info.get("version"),
        "aiConfigured": info.get("aiConfigured"),
        "webhookConfigured": info.get("webhookConfigured"),
        "backupIntervalHours": info.get("backupIntervalHours"),
        "startedAt": _STARTED_AT.isoformat(),
        "uptimeSeconds": int((now - _STARTED_AT).total_seconds()),
        "timestamp": now.isoformat(),
    }


def _parse_since(value: Optional[str], label: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid {label} (use an ISO date-time)")
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


async def _contact_filters(session: AsyncSession, me: Optional[uuid.UUID], search: Optional[str], eventId: Optional[str],
                           owner: Optional[str], createdSince: Optional[str], createdBefore: Optional[str]) -> list:
    """WHERE conditions shared by the contacts list and its stats (everything but the score band)."""
    conditions = []
    if me is not None:
        conditions.append(ContactDB.user_id == me)
    elif owner:
        try:
            conditions.append(ContactDB.user_id == uuid.UUID(owner))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid userId")

    if search:
        s = f"%{search.strip()}%"
        conditions.append(or_(
            ContactDB.name.ilike(s),
            ContactDB.email.ilike(s),
            ContactDB.company_name.ilike(s),
            ContactDB.notes.ilike(s)
        ))

    if eventId:
        name = " ".join(eventId.split())
        if name.startswith("event-"):
            # legacy slug ids ("event-whx-2026"): loose match as before
            clean_eid = name.replace("event-", "").replace("-", " ")
            conditions.append(or_(ContactDB.event_name == name, ContactDB.event_name.ilike(f"%{clean_eid}%")))
        else:
            # an event NAME: the same event in any case / spacing — "GITEX" used to also match
            # "GITEX 2026", mixing events on the event page
            conditions.append(await event_filter(session, ContactDB.event_name, name))

    since = _parse_since(createdSince, "createdSince")
    if since:
        conditions.append(ContactDB.created_at >= since)
    before = _parse_since(createdBefore, "createdBefore")
    if before:
        conditions.append(ContactDB.created_at < before)
    return conditions


@api_router.get("/contacts/stats")
async def contact_stats(
    search: Optional[str] = Query(None),
    eventId: Optional[str] = Query(None),
    userId: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    createdSince: Optional[str] = Query(None),
    createdBefore: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """How many contacts match the list filters, per level (hot ≥ 85, warm 70–84, cool 50–69, cold < 50,
    unscored) — counted in the database, so the console needn't download every row to show the bands.
    The app counts the rep's own score; the console its own (admin) score."""
    conditions = await _contact_filters(session, me, search, eventId, userId or user_id, createdSince, createdBefore)
    score_col = ContactDB.lead_score if me is not None else AdminContactScoreDB.score
    level = level_case(score_col)
    query = select(level.label("level"), func.count(ContactDB.id)).select_from(ContactDB)
    if me is None:
        query = query.outerjoin(AdminContactScoreDB, AdminContactScoreDB.contact_id == ContactDB.id)
    if conditions:
        query = query.where(and_(*conditions))
    levels = {"hot": 0, "warm": 0, "cool": 0, "cold": 0, "unscored": 0}
    for name, n in (await session.execute(query.group_by(level))).all():
        levels[name] = int(n or 0)
    return {"total": sum(levels.values()), "levels": levels}


@api_router.get("/contacts")
async def list_contacts(
    search: Optional[str] = Query(None),
    stage: Optional[str] = Query(None),
    scoreBand: Optional[str] = Query(None),
    eventId: Optional[str] = Query(None),
    userId: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    limit: int = Query(100),
    cursor: Optional[str] = Query(None),
    # recent = newest first (default); score = best lead score first, unscored last, then newest.
    # The Scout feed ranks with sort=score so its "follow up first" order covers ALL contacts,
    # not just the newest page it downloaded.
    sort: str = Query("recent"),
    createdSince: Optional[str] = Query(None),
    createdBefore: Optional[str] = Query(None),
    # lite=1: list rows without the photo / avatar / score write-up (the console's big tables)
    lite: bool = Query(False),
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """List contacts matching filters for Admin Table Grid & Mobile Feed.

    With a login token (mobile) the list is always the caller's own contacts; a userId
    query can't widen it. Without one (admin console) userId optionally narrows it.
    """
    conditions = await _contact_filters(session, me, search, eventId, userId or user_id, createdSince, createdBefore)

    # The app ranks by the rep's own score; the admin console by its own (admin_contact_scores).
    score_col = ContactDB.lead_score if me is not None else AdminContactScoreDB.score
    band = band_condition(score_col, scoreBand)
    if band is not None:
        conditions.append(band)

    limit = max(1, min(limit, 1000))
    # `cursor` = opaque offset from the previous page ("" / None = first page); id breaks ties so
    # pages never repeat or skip a contact.
    try:
        offset = max(0, int(cursor)) if cursor else 0
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid cursor")
    if (sort or "").lower() == "score":
        order = (score_col.is_(None), score_col.desc(), ContactDB.created_at.desc(), ContactDB.id)
    else:
        order = (ContactDB.created_at.desc(), ContactDB.id)
    query = select(ContactDB)
    if me is None:
        query = query.outerjoin(AdminContactScoreDB, AdminContactScoreDB.contact_id == ContactDB.id)
    query = query.order_by(*order).offset(offset).limit(limit + 1)
    if conditions:
        query = query.where(and_(*conditions))

    result = await session.execute(query)
    rows = result.scalars().all()
    has_more = len(rows) > limit
    rows = rows[:limit]

    contacts = await console_scores(session, me, [format_contact(r) for r in rows])
    if lite:
        contacts = [_lite(c) for c in contacts]
    return {
        "contacts": contacts,
        "cursor": str(offset + limit) if has_more else None,
    }


# ==================== RECENTLY DELETED (30-DAY ARCHIVE) ====================
# Declared before /contacts/{contact_id} so "deleted"/"duplicates" aren't parsed as IDs.

async def reassign_user_archive(session: AsyncSession, source: uuid.UUID, target: uuid.UUID) -> None:
    """User merge: the source account's Recently deleted items and duplicate decisions move too."""
    await session.execute(update(DeletedContactDB).where(DeletedContactDB.user_id == source).values(user_id=target))
    await session.execute(update(DuplicateDismissalDB).where(DuplicateDismissalDB.user_id == source).values(user_id=target))


async def purge_user_contact_data(session: AsyncSession, user_id: uuid.UUID) -> None:
    """Account deletion: permanently remove the user's contacts, their archive and dependent rows.
    (No one is left to restore to, so these skip Recently deleted.)"""
    ids = select(ContactDB.id).where(ContactDB.user_id == user_id)
    for model in (ContactFileDB, ContactPipelineDB, ContactListMemberDB):
        await session.execute(delete(model).where(model.contact_id.in_(ids)))
    await session.execute(delete(ContactDB).where(ContactDB.user_id == user_id))
    await session.execute(delete(DeletedContactDB).where(DeletedContactDB.user_id == user_id))
    await session.execute(delete(DuplicateDismissalDB).where(DuplicateDismissalDB.user_id == user_id))
    await session.execute(delete(ContactShareLinkDB).where(ContactShareLinkDB.user_id == user_id))


async def purge_expired_deleted(session: AsyncSession, user_id: Optional[uuid.UUID] = None) -> int:
    """Permanently remove archived contacts past their 30 days."""
    query = delete(DeletedContactDB).where(DeletedContactDB.purge_after < datetime.now(timezone.utc))
    if user_id is not None:
        query = query.where(DeletedContactDB.user_id == user_id)
    result = await session.execute(query)
    await session.commit()
    return result.rowcount or 0


async def purge_expired_deleted_contacts() -> int:
    """Entry point for app.py's startup / periodic job."""
    factory = get_session_factory()
    if not factory:
        return 0
    async with factory() as session:
        purged = await purge_expired_deleted(session)
    if purged:
        print(f"[CONTACTS] Purged {purged} contact(s) deleted more than {RECENTLY_DELETED_DAYS} days ago")
    return purged


def _format_deleted(d: "DeletedContactDB") -> dict:
    now = datetime.now(timezone.utc)
    purge_after = _aware(d.purge_after)
    photo = (d.snapshot or {}).get("photo_base64")
    return {
        "id": str(d.id),
        "name": d.name,
        "company": _present(d.company_name),
        "title": _present((d.snapshot or {}).get("job_title")),
        "photoUrl": (photo if str(photo).startswith(("data:", "http")) else f"data:image/jpeg;base64,{photo}") if photo else None,
        "reason": d.reason,
        "deletedAt": _aware(d.deleted_at).isoformat() if d.deleted_at else None,
        "purgeAfter": purge_after.isoformat() if purge_after else None,
        "daysLeft": max(0, (purge_after - now).days + (1 if (purge_after - now).seconds > 0 else 0)) if purge_after else 0,
    }


async def _owner_or_401(me: Optional[uuid.UUID], user_id: Optional[str]) -> uuid.UUID:
    return await identity.data_owner(me, user_id)


@api_router.get("/contacts/deleted")
async def list_deleted_contacts(
    userId: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """Your Recently deleted contacts (newest first). Expired ones are purged first."""
    owner = await _owner_or_401(me, userId)
    await purge_expired_deleted(session, owner)
    rows = (await session.execute(
        select(DeletedContactDB).where(DeletedContactDB.user_id == owner).order_by(DeletedContactDB.deleted_at.desc())
    )).scalars().all()
    return {"contacts": [_format_deleted(d) for d in rows], "retentionDays": RECENTLY_DELETED_DAYS}


@api_router.delete("/contacts/deleted")
async def empty_deleted_contacts(
    userId: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """Permanently delete everything in your Recently deleted."""
    owner = await _owner_or_401(me, userId)
    ids = [str(i) for i in (await session.execute(
        select(DeletedContactDB.id).where(DeletedContactDB.user_id == owner))).scalars().all()]
    result = await session.execute(delete(DeletedContactDB).where(DeletedContactDB.user_id == owner))
    await session.commit()
    if ids:
        await broadcast_live_event("contact_purged", {"ids": ids})
    return {"status": "emptied", "deleted": result.rowcount or 0}


async def _load_deleted(session: AsyncSession, contact_id: str, owner: uuid.UUID) -> "DeletedContactDB":
    try:
        cid = uuid.UUID(contact_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid contact ID format")
    d = (await session.execute(select(DeletedContactDB).where(DeletedContactDB.id == cid))).scalar_one_or_none()
    if d is None or d.user_id != owner or _aware(d.purge_after) < datetime.now(timezone.utc):
        raise HTTPException(status_code=404, detail="That contact isn't in Recently deleted (it may have expired)")
    return d


@api_router.post("/contacts/deleted/{contact_id}/restore")
async def restore_deleted_contact(
    contact_id: str,
    userId: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """Put a Recently deleted contact back (same ID, fields, photo and attached files)."""
    owner = await _owner_or_401(me, userId)
    d = await _load_deleted(session, contact_id, owner)
    contact = await contact_store.restore_contact(session, d, owner)
    return await console_contact(session, me, contact)


@api_router.delete("/contacts/deleted/{contact_id}")
async def purge_deleted_contact(
    contact_id: str,
    userId: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """Permanently delete one contact from Recently deleted (can't be undone)."""
    owner = await _owner_or_401(me, userId)
    d = await _load_deleted(session, contact_id, owner)
    await session.delete(d)
    await session.commit()
    await broadcast_live_event("contact_purged", {"ids": [contact_id]})
    return {"status": "purged", "id": contact_id}


# ==================== DUPLICATE DETECTION ====================

# Matching rules (shared by the app's per-rep checks and the admin console's workspace-wide ones):
#   strong evidence — a shared email, a shared phone number, or the same name + company
#                     (or the same 3+ word name when there's no company) → flagged on its own;
#   weak evidence   — a team inbox (info@, sales@…) or a number / address that 3+ different people
#                     share (an office switchboard) → flagged only when the person's name matches too.
# Repeated copies of ONE person sharing a phone stay strong: sharing is counted in distinct people.

# One field can hold several values ("+91 98…, +91 99…", "a@x.com; b@y.com"): the same separators
# the admin console splits on (, ; | newline and " / ").
_MULTI_SPLIT = re.compile(r"\s*[,;|\n]\s*|\s+/\s+")
_EMAIL_RE = re.compile(r"[a-z0-9._%+'-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+")
# "+1 555 123 4567 ext 89" / "x89" / "#89": the extension isn't part of the number
_PHONE_EXT = re.compile(r"(?:\s*(?:ext|extn|extension|x)\.?\s*|\s*#\s*)\d{1,6}\s*$", re.IGNORECASE)
# Words that differ between two captures of the same person / company.
_NAME_TITLES = {"mr", "mrs", "ms", "miss", "mx", "dr", "prof", "sir", "shri", "smt", "sri"}
_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "phd", "md", "mba", "esq", "cpa"}
_COMPANY_SUFFIXES = {"inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "pvt", "private", "plc", "corp",
                     "corporation", "co", "company", "gmbh", "ag", "sa", "bv", "nv", "pte", "pty", "srl", "oy", "ab", "the"}
# "Companies" that say nothing about who someone is: two John Smiths who are both "Freelance" aren't one person
_NO_COMPANY = {"self employed", "self", "freelance", "freelancer", "independent", "student", "consultant", "retired",
               "unemployed", "none", "na", "n a", "nil", "personal", "individual", "not applicable", "home", "homemaker", "unknown"}
# Team / role inboxes several people use
_GENERIC_INBOXES = {"info", "sales", "contact", "contactus", "hello", "hi", "admin", "administrator", "support", "help", "helpdesk",
                    "office", "enquiry", "enquiries", "inquiry", "inquiries", "team", "marketing", "hr", "careers", "jobs",
                    "recruitment", "noreply", "no-reply", "donotreply", "mail", "email", "billing", "accounts", "accounting",
                    "finance", "service", "services", "customerservice", "customercare", "care", "reception", "frontdesk",
                    "general", "events", "media", "press", "pr", "webmaster", "postmaster", "feedback", "booking", "bookings", "orders"}
SHARED_BY_PEOPLE = 3  # a phone / email used by this many different people is a shared line, not one person's


def _words(value: Optional[str]) -> List[str]:
    """Lower-case words with punctuation dropped ("Acme, Inc." → ["acme", "inc"])."""
    return re.sub(r"[^\w\s]", " ", (value or "").lower()).split()


def _person_words(name: Optional[str]) -> List[str]:
    """The name without courtesy titles or suffixes ("Dr. Jane Doe, PhD" → ["jane", "doe"])."""
    words = _words(name)
    while words and words[0] in _NAME_TITLES:
        words = words[1:]
    while words and words[-1] in _NAME_SUFFIXES:
        words = words[:-1]
    return words


def _dup_company_key(company: Optional[str]) -> str:
    """"Acme Pvt. Ltd." = "ACME" = "The Acme Company"; "" for none / a placeholder like "Freelance"."""
    c = " ".join(w for w in _words(_present(company)) if w not in _COMPANY_SUFFIXES)
    return "" if c in _NO_COMPANY else c


def _phone_keys(phone: Optional[str]) -> set:
    """Each number in the field as its last 10 digits (the local number, country code ignored)."""
    out = set()
    for part in _MULTI_SPLIT.split(_present(phone) or ""):
        digits = re.sub(r"\D", "", _PHONE_EXT.sub("", part))
        if len(digits) >= 7:
            out.add(digits[-10:])
    return out


def _email_keys(email: Optional[str]) -> set:
    """Every email address in the field, lower-cased."""
    return set(_EMAIL_RE.findall((_present(email) or "").lower()))


def _generic_inbox(email: str) -> bool:
    return email.split("@", 1)[0] in _GENERIC_INBOXES


def _name_key(name: Optional[str], company: Optional[str]) -> Optional[str]:
    words = _person_words(name)
    if not words:
        return None
    n, c = " ".join(words), _dup_company_key(company)
    # Name alone is too weak ("John Smith"); require the company too, or a 3+ word name.
    if c:
        return f"{n}|{c}"
    return n if len(words) >= 3 else None


def _keys_for(name, company, email, phone) -> dict:
    """kind → set of match keys (a field can hold several emails / phone numbers). `person` (a 2+ word
    name) only backs up weak evidence; `who` identifies the person when counting shared lines."""
    words = _person_words(name)
    nk = _name_key(name, company)
    return {"email": _email_keys(email), "phone": _phone_keys(phone), "name": {nk} if nk else set(),
            "person": {" ".join(words)} if len(words) >= 2 else set(), "who": " ".join(words)}


def _shared_lines(keys_list: List[dict]) -> set:
    """"email:…" / "phone:…" values that SHARED_BY_PEOPLE or more different people use."""
    people: Dict[str, set] = {}
    for k in keys_list:
        who = k["who"] or f"#{id(k)}"  # no name: count the contact as its own person
        for kind in ("email", "phone"):
            for v in k[kind]:
                people.setdefault(f"{kind}:{v}", set()).add(who)
    return {key for key, who in people.items() if len(who) >= SHARED_BY_PEOPLE}


def _match(a: dict, b: dict, shared: set = frozenset()) -> tuple:
    """(reasons, strong). reasons = [] when the two aren't flagged."""
    common = {kind: a[kind] & b[kind] for kind in ("email", "phone", "name")}
    strong = bool(common["name"]) or any(
        f"{kind}:{v}" not in shared and not (kind == "email" and _generic_inbox(v))
        for kind in ("email", "phone") for v in common[kind])
    backed = bool(a["person"] & b["person"]) and bool(common["email"] or common["phone"])
    if not (strong or backed):
        return [], False
    return [kind for kind in ("email", "phone", "name") if common[kind]], strong


def duplicate_reasons(a: dict, b: dict, shared: set = frozenset()) -> List[str]:
    """Why two contacts look like the same person ([] = they don't): see the rules above."""
    return _match(a, b, shared)[0]


def _match_confidence(reasons: List[str], strong: bool) -> int:
    # weak evidence backed only by a matching name ranks below every strong match
    return max(DUPLICATE_CONFIDENCE[r] for r in reasons) if strong else 75


def _duplicate_pairs(keyed: List[tuple], dismissed: set) -> List[tuple]:
    """[(a, b, reasons, strong)] for every flagged pair among `keyed` [(contact, keys)], each pair once,
    minus dismissed (lo, hi) pairs. Only contacts that share a strong value or a person name are compared."""
    shared = _shared_lines([k for _, k in keyed])
    buckets: Dict[str, List[int]] = {}
    for idx, (_, k) in enumerate(keyed):
        for kind in ("email", "phone"):
            for v in k[kind]:
                if f"{kind}:{v}" not in shared:
                    buckets.setdefault(f"{kind}:{v}", []).append(idx)
        for kind in ("name", "person"):
            for v in k[kind]:
                buckets.setdefault(f"{kind}:{v}", []).append(idx)
    seen, out = set(), []
    for members in buckets.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                (a, ka), (b, kb) = keyed[members[i]], keyed[members[j]]
                pair = _pair(a.id, b.id)
                if pair in seen or pair in dismissed:
                    continue
                seen.add(pair)
                reasons, strong = _match(ka, kb, shared)
                if reasons:
                    out.append((a, b, reasons, strong))
    return out


async def _dismissed_pairs(session: AsyncSession, owner: uuid.UUID) -> set:
    rows = (await session.execute(
        select(DuplicateDismissalDB.contact_a, DuplicateDismissalDB.contact_b).where(DuplicateDismissalDB.user_id == owner)
    )).all()
    return {(a, b) for a, b in rows}


def _pair(a: uuid.UUID, b: uuid.UUID):
    return (a, b) if str(a) < str(b) else (b, a)


async def find_duplicates(session: AsyncSession, owner: uuid.UUID, keys: dict,
                          exclude_id: Optional[uuid.UUID] = None) -> List[tuple]:
    """Owner's contacts that look like the same person as `keys` (see the matching rules above)."""
    if not (keys["email"] or keys["phone"] or keys["name"] or keys["person"]):
        return []
    rows = (await session.execute(select(ContactDB).where(ContactDB.user_id == owner))).scalars().all()
    dismissed = await _dismissed_pairs(session, owner) if exclude_id else set()
    others = [(c, _keys_for(c.name, c.company_name, c.email, c.phone)) for c in rows if c.id != exclude_id]
    shared = _shared_lines([keys] + [k for _, k in others])
    matches = []
    for c, k in others:
        reasons = duplicate_reasons(keys, k, shared)
        if reasons and not (exclude_id is not None and _pair(exclude_id, c.id) in dismissed):
            matches.append((c, reasons))
    return matches


@api_router.get("/contacts/duplicates")
async def list_duplicate_pairs(
    userId: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """Every pair of your contacts that look like the same person (minus ones you dismissed)."""
    owner = await _owner_or_401(me, userId)
    rows = (await session.execute(select(ContactDB).where(ContactDB.user_id == owner))).scalars().all()
    dismissed = await _dismissed_pairs(session, owner)
    keyed = [(c, _keys_for(c.name, c.company_name, c.email, c.phone)) for c in rows]
    found = _duplicate_pairs(keyed, dismissed)
    views = await console_scores(session, me, [format_contact(x) for a, b, _, _ in found for x in (a, b)])
    pairs = [{"a": views[2 * i], "b": views[2 * i + 1], "reasons": reasons} for i, (_, _, reasons, _) in enumerate(found)]
    return {"pairs": pairs, "count": len(pairs)}


class CheckDuplicatesPayload(BaseModel):
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    company: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    excludeId: Optional[str] = None  # the contact being edited
    userId: Optional[str] = None


@api_router.post("/contacts/check-duplicates")
async def check_duplicates(
    payload: CheckDuplicatesPayload,
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """Would these details duplicate one of your contacts? Used before saving an edit."""
    owner = await _owner_or_401(me, payload.userId)
    name = f"{payload.firstName or ''} {payload.lastName or ''}".strip()
    exclude = None
    if payload.excludeId:
        try:
            exclude = uuid.UUID(payload.excludeId)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid excludeId")
    matches = await find_duplicates(session, owner, _keys_for(name, payload.company, payload.email, payload.phone), exclude)
    # An edit shouldn't be blocked by a pair the user already said is fine.
    if exclude is not None:
        dismissed = await _dismissed_pairs(session, owner)
        matches = [(c, r) for c, r in matches if _pair(exclude, c.id) not in dismissed]
    return {"matches": await console_matches(session, me, matches)}


@api_router.get("/contacts/{contact_id}/duplicates")
async def get_contact_duplicates(
    contact_id: str,
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """Other contacts of the same owner that look like this one."""
    contact = await load_contact(session, contact_id, me)
    keys = _keys_for(contact.name, contact.company_name, contact.email, contact.phone)
    matches = await find_duplicates(session, contact.user_id, keys, exclude_id=contact.id)
    return {"matches": await console_matches(session, me, matches)}


@api_router.post("/contacts/{contact_id}/duplicates/{other_id}/dismiss")
async def dismiss_duplicate(
    contact_id: str,
    other_id: str,
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """Mark two contacts as different people so they stop being flagged."""
    a = await load_contact(session, contact_id, me)
    b = await load_contact(session, other_id, me)
    if a.user_id != b.user_id:
        raise HTTPException(status_code=404, detail="Contact not found")
    first, second = _pair(a.id, b.id)
    exists = (await session.execute(select(DuplicateDismissalDB.id).where(
        DuplicateDismissalDB.user_id == a.user_id,
        DuplicateDismissalDB.contact_a == first,
        DuplicateDismissalDB.contact_b == second,
    ))).scalar_one_or_none()
    if not exists:
        session.add(DuplicateDismissalDB(user_id=a.user_id, contact_a=first, contact_b=second))
        await session.commit()
    return {"status": "dismissed", "contactId": contact_id, "otherId": other_id}


@api_router.get("/contacts/{contact_id}")
async def get_contact(
    contact_id: str,
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """Get single contact detail (only your own when signed in)."""
    contact = await load_contact(session, contact_id, me)
    return await console_contact(session, me, contact)


class CreateContactPayload(BaseModel):
    eventId: Optional[str] = None
    firstName: str
    lastName: Optional[str] = None
    company: Optional[str] = None
    title: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    notes: Optional[str] = None
    score: Optional[int] = None  # unscored until the AI (or the user) scores it
    scoreBand: Optional[str] = None
    stage: Optional[str] = "new"
    source: Optional[str] = "manual"
    photo: Optional[str] = None  # base64 / data URI image
    # Owner when called without a login token (admin console). Ignored for signed-in callers.
    userId: Optional[str] = None
    # Phone-made UUID: a create retried after a timeout returns the saved contact instead of a copy.
    id: Optional[str] = None


async def resolve_owner(session: AsyncSession, me: Optional[uuid.UUID], explicit: Optional[str]) -> uuid.UUID:
    """Who a new contact belongs to: the signed-in user, else an explicit, existing userId."""
    return await identity.data_owner(me, explicit, session=session, missing_status=400,
                                     missing_detail="Sign in, or pass userId, to choose who owns this contact")


@api_router.post("/contacts")
async def create_contact(
    payload: CreateContactPayload,
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """Create a contact for the signed-in user (or an explicit userId), persist, broadcast SSE."""
    user_id = await resolve_owner(session, me, payload.userId)
    if not f"{payload.firstName} {payload.lastName or ''}".strip():
        raise HTTPException(status_code=400, detail="A contact needs a name")
    contact_id = None
    if payload.id:
        try:
            contact_id = uuid.UUID(payload.id)
        except ValueError:
            raise HTTPException(status_code=400, detail="id must be a UUID")
        existing, archived = await contact_store.find_by_id(session, contact_id)
        if existing is not None:
            if existing.user_id != user_id:
                raise HTTPException(status_code=409, detail="That contact id is already used")
            return await console_contact(session, me, existing)  # the retry of a create that went through
        if archived is not None:
            raise HTTPException(status_code=409, detail="This contact was deleted")
    fields = payload.model_dump(exclude_unset=True, exclude={"id", "userId", "scoreBand", "stage", "source"})
    saved = await contact_store.create_contact(session, user_id, fields, strict=True, contact_id=contact_id,
                                               source=payload.source or "manual")
    return await console_contact(session, me, saved.contact)


class PatchContactPayload(BaseModel):
    baseVersion: Optional[int] = 1
    patch: Dict[str, Any]
    # The edited fields as they were when the edit started (API names). When the server's value has
    # changed since then (someone else edited it), the save is refused with 409 + the conflicts,
    # unless force is set ("keep mine").
    base: Optional[Dict[str, Any]] = None
    force: bool = False


@api_router.patch("/contacts/{contact_id}")
async def patch_contact(
    contact_id: str,
    payload: PatchContactPayload,
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """Patch contact attributes (incl. photo) and broadcast SSE update.

    `photo`: base64 / data-URI image to set, or null to remove the picture.
    """
    contact = await load_contact(session, contact_id, me)
    try:
        await contact_store.update_contact(session, contact, payload.patch, strict=True, base=payload.base,
                                           force=payload.force, console=me is None)
    except contact_store.EditConflict as clash:
        raise HTTPException(status_code=409, detail={
            "message": "This contact was changed somewhere else",
            "conflicts": clash.conflicts,
            "contact": await console_contact(session, me, contact),
        })
    return await console_contact(session, me, contact)


# ==================== FIND ON LINKEDIN ====================
# Search: live web search for the contact's LinkedIn profile; returns candidates, saves nothing.
# Attach: the user picked one — store the link and (if found) the profile photo.

LINKEDIN_SEARCH_MESSAGES = {
    "not_configured": "LinkedIn search isn't set up on this server yet.",
    "bad_key": "LinkedIn search isn't working right now (server setup). Please tell your admin.",
    "no_credits": "LinkedIn search is paused: the AI account is out of credits.",
    "rate_limited": "Too many searches right now. Please try again in a minute.",
    "timeout": "The search took too long. Please try again.",
    "unavailable": "LinkedIn search isn't available right now. Please try again.",
}


def _person_for_search(c: ContactDB) -> Dict[str, str]:
    def present(v):
        v = (v or "").strip()
        return "" if v.upper() == "N/A" else v
    parts = present(c.name).split()
    return {
        "first": parts[0] if parts else "",
        "last": " ".join(parts[1:]),
        "company": present(c.company_name),
        "title": present(c.job_title),
        "email": present(c.email),
    }


@api_router.post("/contacts/{contact_id}/linkedin/search")
async def linkedin_search(
    contact_id: str,
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """Possible LinkedIn profiles for this contact, best match first (nothing is saved).

    Each candidate: {url, name, headline, confidence 0-100, matched: ['name','company',…], photoUrl}.
    Only profiles that appear in real web search results are returned.
    """
    import ai_client
    import linkedin_finder

    contact = await load_contact(session, contact_id, me)
    person = _person_for_search(contact)
    if not person["first"]:
        raise HTTPException(status_code=400, detail="Add the contact's name first, then search LinkedIn.")
    if not (person["last"] or person["company"] or person["title"]):
        raise HTTPException(status_code=400, detail="Add a last name, company or job title so the search can tell people apart.")
    try:
        # The contact's own Gravatar (from their email) is looked up alongside the search: it shows
        # the real person, so the user can compare it with the candidates.
        candidates, gravatar = await asyncio.gather(
            linkedin_finder.find_profiles(person),
            linkedin_finder.gravatar_url(person["email"]),
        )
    except ai_client.AIError as e:
        print(f"[LINKEDIN] search failed for {contact_id}: {e.reason} — {e}")
        raise HTTPException(status_code=503, detail=LINKEDIN_SEARCH_MESSAGES.get(e.reason, LINKEDIN_SEARCH_MESSAGES["unavailable"]))
    return {
        "candidates": candidates,
        "searchedFor": {k: v for k, v in {"name": f"{person['first']} {person['last']}".strip(), "company": person["company"], "title": person["title"]}.items() if v},
        "linkedinSearchUrl": linkedin_finder.people_search_url(person),
        # LinkedIn is refusing automated page loads right now, so LinkedIn photos may be missing.
        "photosLimited": linkedin_finder.photos_limited(),
        "contactPhoto": {"url": gravatar, "source": "gravatar"} if gravatar else None,
        "imageSearchConfigured": linkedin_finder.brave_configured(),
    }


class AttachLinkedInPayload(BaseModel):
    url: str
    photoUrl: Optional[str] = None
    usePhoto: bool = True


@api_router.post("/contacts/{contact_id}/linkedin/attach")
async def linkedin_attach(
    contact_id: str,
    payload: AttachLinkedInPayload,
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """Save the LinkedIn profile the user picked, plus its photo when there is one.

    The photo is only ever downloaded from LinkedIn's image servers (*.licdn.com). Returns
    {contact, photoAdded}.
    """
    import linkedin_finder

    contact = await load_contact(session, contact_id, me)
    url = linkedin_finder.canonical_profile_url(payload.url)
    if not url:
        raise HTTPException(status_code=400, detail="That isn't a LinkedIn profile link (linkedin.com/in/…)")
    contact.linkedin = url  # canonical_profile_url: already a clean linkedin.com/in/… link
    contact.linkedin_source = "search"

    # Photo, first that works: the one the search showed → LinkedIn's (looked up now) → Brave image
    # search → the contact's Gravatar.
    photo_added, photo_source = False, None
    if payload.usePhoto:
        tries = []
        if payload.photoUrl:
            tries.append(("search", lambda: _const(payload.photoUrl)))
        tries.append(("linkedin", lambda: linkedin_finder.find_profile_photo(url)))
        tries.append(("brave", lambda: _brave_one(linkedin_finder, contact, url)))
        tries.append(("gravatar", lambda: linkedin_finder.gravatar_url(contact.email)))
        for source, get in tries:
            photo_url = await get()
            data = await linkedin_finder.download_photo(photo_url) if photo_url else None
            if not data:
                continue
            try:
                contact.photo_base64 = process_contact_photo(base64.b64encode(data).decode("ascii"))
                photo_added, photo_source = True, source
                break
            except HTTPException as e:
                print(f"[LINKEDIN] {source} photo for {contact_id} not usable: {e.detail}")
    await contact_store.save(session, contact, "contact_updated")
    # photoStatus: added | limited (LinkedIn is refusing lookups for now) | none (no public photo)
    photo_status = "added" if photo_added else ("limited" if payload.usePhoto and linkedin_finder.photos_limited() else "none")
    return {"contact": await console_contact(session, me, contact), "photoAdded": photo_added, "photoStatus": photo_status, "photoSource": photo_source}


async def _const(value):
    return value


async def _brave_one(linkedin_finder, contact: ContactDB, profile_url: str) -> Optional[str]:
    return (await linkedin_finder.brave_profile_photos(_person_for_search(contact), [profile_url])).get(profile_url)


@api_router.delete("/contacts/{contact_id}")
async def delete_contact(
    contact_id: str,
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """Delete a contact (your own when signed in): it moves to Recently deleted for 30 days,
    where it can be restored, and disappears from every list, search and the AI index."""
    contact = await load_contact(session, contact_id, me)
    archived = await contact_store.delete_contact(session, contact)
    return {"status": "deleted", "id": contact_id, **{k: v for k, v in _format_deleted(archived).items() if k in ("purgeAfter", "daysLeft")}}


# ==================== EVENTS & SEATS ====================

@api_router.get("/events", dependencies=[Depends(request_user_id)])
async def list_events(session: AsyncSession = Depends(get_db)):
    """Returns the events that have an imported exhibitor directory.

    There is no dedicated events table: an event exists once exhibitors have been
    imported for it (ExhibitorDB.event_name), so the list is derived from that data.
    """
    res = await session.execute(
        select(ExhibitorDB.event_name, func.count(ExhibitorDB.id))
        .group_by(ExhibitorDB.event_name)
        .order_by(ExhibitorDB.event_name)
    )

    events = []
    for event_name, exhibitor_count in res.all():
        if not event_name:
            continue
        slug = event_identity.event_slug(event_name)
        events.append({
            "id": slug,
            "name": event_name,
            "slug": slug,
            "exhibitorCount": exhibitor_count,
        })
    return events


@api_router.get("/events/{event_id}/seats")
async def get_event_seats(event_id: str, session: AsyncSession = Depends(get_db),
                          me: Optional[uuid.UUID] = Depends(request_user_id)):
    """People who scanned contacts at this event, with their real scan counts.

    `event_id` is the event name or its slug. (Used to list every user with a made-up
    "42 × n" scan count, and a fake Sarah Connor / Marcus Wright roster when empty.)
    """
    names = (await session.execute(select(ContactDB.event_name).distinct())).scalars().all()
    wanted = event_id.strip().lower()
    event_names = [n for n in names if n and (event_identity.same_event(n, event_id) or event_identity.event_slug(n) == wanted)]
    if not event_names:
        return []

    rows = (await session.execute(
        select(ContactDB.user_id, func.count(ContactDB.id))
        .where(ContactDB.event_name.in_(event_names))
        .group_by(ContactDB.user_id)
    )).all()
    counts = {uid: n for uid, n in rows}
    users = (await session.execute(select(UserDB).where(UserDB.id.in_(list(counts))))).scalars().all() if counts else []
    seats = [{
        "id": f"seat-{u.id}",
        "eventId": event_id,
        "userId": str(u.id),
        "name": u.name,
        # colleagues' emails only for the console (and your own); reps see names and counts
        "email": u.email if me is None or u.id == me else None,
        "role": "admin" if u.is_admin else "rep",
        "status": "active",
        "scansCount": counts.get(u.id, 0),
    } for u in users]
    return sorted(seats, key=lambda s: -s["scansCount"])


# ==================== ADMIN · EVENTS ====================
# There's no events table: an event is a name. It shows up in the booth directory (exhibitors), on
# contacts ("met at"), in reps' event lists (profile.events / current_event_name) and on event files.
# The admin console lists the union of all four, matched case- and spacing-insensitively (_key).

_FLOOR_PLAN_RE = re.compile(r"floor|plan|map|layout", re.IGNORECASE)


def _is_floor_plan(category: Optional[str], name: Optional[str], description: Optional[str]) -> bool:
    """An event file that's the venue's floor plan: the Files page category "floorplan", or (like the
    app) a name / description that mentions floor, plan, map or layout."""
    return (category or "").strip().lower() == "floorplan" or bool(_FLOOR_PLAN_RE.search(f"{name or ''} {description or ''}"))


@api_router.get("/admin/events", dependencies=[Depends(require_admin)])
async def admin_list_events(session: AsyncSession = Depends(get_db)):
    """Every event the workspace knows — from the booth directory, contacts' "met at", reps' event
    lists and event files — with real counts. Newest activity first."""
    agg: Dict[str, Dict[str, Any]] = {}

    def bucket(name: Optional[str], prefer: bool = False) -> Optional[Dict[str, Any]]:
        k = _key(name)
        if not k:
            return None
        e = agg.setdefault(k, {"key": k, "spellings": {}, "directory": None, "booths": 0, "contacts": 0, "hot": 0,
                               "reps": set(), "following": set(), "activeFor": set(), "files": 0, "floorPlans": 0, "lastActivity": None})
        clean_name = " ".join(str(name).split())
        e["spellings"][clean_name] = e["spellings"].get(clean_name, 0) + 1
        if prefer and not e["directory"]:
            e["directory"] = clean_name  # the booth directory's spelling wins (it's what the app shows)
        return e

    def seen(e: Dict[str, Any], when: Optional[datetime]) -> None:
        when = _aware(when)
        if when and (e["lastActivity"] is None or when > e["lastActivity"]):
            e["lastActivity"] = when

    for name, n, last in (await session.execute(select(ExhibitorDB.event_name, func.count(ExhibitorDB.id), func.max(ExhibitorDB.created_at))
                                                .group_by(ExhibitorDB.event_name))).all():
        e = bucket(name, prefer=True)
        if e:
            e["booths"] += n
            seen(e, last)
    for name, uid, score, created in (await session.execute(_with_admin_score(
            select(ContactDB.event_name, ContactDB.user_id, ADMIN_SCORE, ContactDB.created_at)))).all():
        e = bucket(name)
        if e:
            e["contacts"] += 1
            e["hot"] += is_hot(score)
            e["reps"].add(uid)
            seen(e, created)
    for uid, pd in (await session.execute(select(UserProfileDB.user_id, UserProfileDB.profile_data))).all():
        listed, current = _profile_events(pd)
        for name in listed + ([current] if current else []):
            e = bucket(name)
            if e:
                e["following"].add(uid)
        if current:
            e = bucket(current)
            if e:
                e["activeFor"].add(uid)
    for name, cat, fname, desc, size, stored, created in (await session.execute(
            select(EventFileDB.event_name, EventFileDB.category, EventFileDB.original_filename, EventFileDB.description,
                   EventFileDB.file_size, func.length(EventFileDB.file_data), EventFileDB.created_at)
            .where(EventFileDB.is_active == True))).all():  # noqa: E712
        e = bucket(name)
        if not e:
            continue
        real = _stored_len(stored) > 0 and _stored_len(stored) == int(size or 0)  # demo placeholders don't count
        if real:
            e["files"] += 1
            e["floorPlans"] += 1 if _is_floor_plan(cat, fname, desc) else 0
            seen(e, created)

    out = []
    for e in agg.values():
        name = e["directory"] or max(e["spellings"].items(), key=lambda kv: (kv[1], kv[0]))[0]
        out.append({
            "name": name, "key": e["key"], "booths": e["booths"], "contacts": e["contacts"], "hot": e["hot"],
            "reps": len(e["reps"]), "following": len(e["following"]), "activeFor": len(e["activeFor"]),
            "files": e["files"], "floorPlans": e["floorPlans"], "hasDirectory": e["booths"] > 0,
            "lastActivity": _iso(e["lastActivity"]),
        })
    out.sort(key=lambda x: (x["lastActivity"] or "", x["contacts"] + x["booths"]), reverse=True)
    return out


@api_router.get("/admin/events/detail", dependencies=[Depends(require_admin)])
async def admin_event_detail(event: str = Query(..., min_length=1), session: AsyncSession = Depends(get_db)):
    """One event in full: its booths (with how many reps visited each and how many contacts were met from
    that company), the reps there, contacts captured per day, and its files — floor plans flagged."""
    k = _key(event)
    if not k:
        raise HTTPException(status_code=400, detail="Pick an event.")
    users = {u.id: u for u in (await session.execute(select(UserDB))).scalars().all()}
    name_of = lambda uid: users[uid].name if uid in users else "Unknown"

    booths = (await session.execute(select(ExhibitorDB).where(await event_filter(session, ExhibitorDB.event_name, k))
                                    .order_by(ExhibitorDB.hall, ExhibitorDB.booth, ExhibitorDB.name))).scalars().all()
    contacts = (await session.execute(_with_admin_score(
        select(ContactDB.id, ContactDB.user_id, ContactDB.company_name, ADMIN_SCORE.label("score"), ContactDB.created_at))
                                      .where(await event_filter(session, ContactDB.event_name, k)))).all()
    visits = (await session.execute(select(BoothVisitDB.user_id, BoothVisitDB.exhibitor_key, BoothVisitDB.visited_at)
                                    .where(BoothVisitDB.event_key == k))).all()

    # contacts met per company (same company matching as duplicate detection: suffixes ignored)
    met_by_company: Dict[str, int] = {}
    for c in contacts:
        ck = _dup_company_key(c.company_name)
        if ck:
            met_by_company[ck] = met_by_company.get(ck, 0) + 1
    visitors: Dict[str, List[str]] = {}
    for v in visits:
        visitors.setdefault(v.exhibitor_key, []).append(name_of(v.user_id))

    booth_out = []
    for b in booths:
        who = visitors.get(_key(b.name), [])
        booth_out.append({**_exhibitor_out(b, None, None), "visitCount": len(who), "visitors": sorted(who),
                          "metContacts": met_by_company.get(_dup_company_key(b.name), 0) if _dup_company_key(b.name) else 0,
                          "addedBy": name_of(b.added_by) if b.added_by else None})

    # reps: who captured here, who visited booths, who has it on their list / as current event
    reps: Dict[uuid.UUID, Dict[str, Any]] = {}
    rep = lambda uid: reps.setdefault(uid, {"userId": str(uid), "name": name_of(uid), "contacts": 0, "hot": 0, "boothsVisited": 0,
                                            "following": False, "active": False, "lastCapture": None})
    for c in contacts:
        r = rep(c.user_id)
        r["contacts"] += 1
        r["hot"] += is_hot(c.score)
        when = _iso(c.created_at)
        if when and (not r["lastCapture"] or when > r["lastCapture"]):
            r["lastCapture"] = when
    for v in visits:
        rep(v.user_id)["boothsVisited"] += 1
    for uid, pd in (await session.execute(select(UserProfileDB.user_id, UserProfileDB.profile_data))).all():
        listed, current = _profile_events(pd)
        if any(_key(x) == k for x in listed) or _key(current) == k:
            r = rep(uid)
            r["following"] = True
            r["active"] = _key(current) == k

    # contacts captured per day (UTC date), oldest first
    per_day: Dict[str, int] = {}
    for c in contacts:
        d = _aware(c.created_at)
        if d:
            per_day[d.date().isoformat()] = per_day.get(d.date().isoformat(), 0) + 1

    files = []
    for f in (await session.execute(
            select(EventFileDB.id, EventFileDB.original_filename, EventFileDB.filename, EventFileDB.file_type, EventFileDB.mime_type,
                   EventFileDB.category, EventFileDB.file_size, EventFileDB.description, EventFileDB.event_name, EventFileDB.created_at,
                   func.length(EventFileDB.file_data).label("stored"))
            .where(EventFileDB.is_active == True,  # noqa: E712
                   or_(await event_filter(session, EventFileDB.event_name, k), EventFileDB.event_name.is_(None), EventFileDB.event_name == ""))
            .order_by(EventFileDB.created_at.desc()))).all():
        if _stored_len(f.stored) == 0 or _stored_len(f.stored) != int(f.file_size or 0):
            continue  # demo placeholder without the real file
        files.append({"id": str(f.id), "filename": f.original_filename or f.filename, "fileType": f.file_type, "mimeType": f.mime_type,
                      "category": f.category, "fileSize": _stored_len(f.stored), "description": f.description or "",
                      "forAllEvents": not (f.event_name or "").strip(),
                      "floorPlan": _is_floor_plan(f.category, f.original_filename or f.filename, f.description),
                      "createdAt": _iso(f.created_at)})

    halls = {(b.hall or "").strip().lower() for b in booths if (b.hall or "").strip()}
    return {
        "name": booths[0].event_name if booths else event.strip(),
        "key": k,
        "stats": {
            "booths": len(booths), "halls": len(halls),
            "categories": len({(b.category or "").strip().lower() for b in booths if (b.category or "").strip()}),
            "contacts": len(contacts), "hot": sum(1 for c in contacts if is_hot(c.score)),
            "reps": len(reps),  # the people listed below: captured here, visited booths or follow the event
            "repsCapturing": len({c.user_id for c in contacts}),
            "boothsVisited": sum(1 for b in booth_out if b["visitCount"]),
            "boothsMet": sum(1 for b in booth_out if b["metContacts"]),
        },
        "booths": booth_out,
        "reps": sorted(reps.values(), key=lambda r: (-r["contacts"], -r["boothsVisited"], r["name"])),
        "perDay": [{"date": d, "count": n} for d, n in sorted(per_day.items())],
        "files": files,
    }


# ==================== ADMIN DASHBOARD ====================

@api_router.get("/admin/dashboard", dependencies=[Depends(require_admin)])
async def get_admin_dashboard(
    days: int = Query(7, ge=1, le=90, description="Length of the period for `range` (charts, deltas)"),
    tz_offset: int = Query(0, ge=-840, le=840, description="Browser Date.getTimezoneOffset() — days are the admin's local days"),
    tzOffset: Optional[int] = Query(None, ge=-840, le=840, description="Same as tz_offset (camelCase)"),
    session: AsyncSession = Depends(get_db),
):
    """Workspace metrics for the admin dashboard — every number is counted from the database.

    Levels match the app: hot ≥ 85, warm 70–84, cool 50–69, cold < 50 (score), unscored = no score.
    "Active reps today" = people who added a contact in the last 24 h.
    `range` = the last `days` local days vs the `days` before them (series, deltas, distribution, top reps).
    """
    if tzOffset is not None:
        tz_offset = tzOffset
    period = await _dashboard_range(session, days, tz_offset)
    now = datetime.now(timezone.utc)
    day, week = now - timedelta(hours=24), now - timedelta(days=7)
    count = lambda *where: select(func.count(ContactDB.id)).where(*where) if where else select(func.count(ContactDB.id))
    scalar = lambda q: session.execute(q)
    total = (await scalar(count())).scalar() or 0
    # levels + average by the admin score (what the contacts grid filters on: ?band=A etc.)
    level_count = lambda *where: _with_admin_score(select(func.count(ContactDB.id))).where(*where)
    hot, warm, cool, cold, unscored = [(await scalar(level_count(level_condition(ADMIN_SCORE, lvl)))).scalar() or 0
                                       for lvl in ("hot", "warm", "cool", "cold", "unscored")]
    today = (await scalar(count(ContactDB.created_at >= day))).scalar() or 0
    this_week = (await scalar(count(ContactDB.created_at >= week))).scalar() or 0
    avg = (await session.execute(_with_admin_score(select(func.avg(ADMIN_SCORE))).where(ADMIN_SCORE.is_not(None)))).scalar()
    users = (await session.execute(select(UserDB.is_admin, func.count(UserDB.id)).group_by(UserDB.is_admin))).all()
    admins = sum(n for a, n in users if a)
    reps = sum(n for a, n in users if not a)
    # reps = non-admin accounts (same as `reps` above), so "active x of y" compares like with like
    active_today = (await session.execute(select(func.count(func.distinct(ContactDB.user_id))).join(UserDB, UserDB.id == ContactDB.user_id)
                                          .where(ContactDB.created_at >= day, UserDB.is_admin == False))).scalar() or 0  # noqa: E712
    by_event = (await session.execute(
        select(ContactDB.event_name, func.count(ContactDB.id))
        .where(ContactDB.event_name.is_not(None), func.trim(ContactDB.event_name) != "")
        .group_by(ContactDB.event_name).order_by(func.count(ContactDB.id).desc()).limit(6))).all()
    by_source = (await session.execute(select(ContactDB.source, func.count(ContactDB.id)).group_by(ContactDB.source))).all()
    days = []
    for i in range(6, -1, -1):  # last 7 days, oldest first (UTC days)
        start = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
        n = (await scalar(count(ContactDB.created_at >= start, ContactDB.created_at < start + timedelta(days=1)))).scalar() or 0
        days.append({"date": start.date().isoformat(), "count": n})
    pipes = dict((await session.execute(select(ContactPipelineDB.status, func.count(ContactPipelineDB.id)).group_by(ContactPipelineDB.status))).all())
    running = sum(n for s, n in pipes.items() if s in PIPELINE_RUNNING)
    return {
        "totalContacts": total,
        "scansToday": today,
        "scansThisWeek": this_week,
        "levels": {"hot": hot, "warm": warm, "cool": cool, "cold": cold, "unscored": unscored},
        "averageScore": round(float(avg), 1) if avg is not None else None,
        "reps": reps,
        "admins": admins,
        "activeRepsToday": active_today,
        "activeReps": active_today,
        "totalReps": reps,
        "todayStart": period["todayStart"],
        "todayByHour": period["todayByHour"],
        "repsToday": period["repsToday"],
        "byEvent": [{"event": e, "count": n} for e, n in by_event],
        "bySource": [{"source": s or "unknown", "count": n} for s, n in by_source],
        "last7Days": days,
        "pipelines": {"running": running, "complete": pipes.get("complete", 0) + pipes.get("complete_no_deck", 0), "failed": pipes.get("failed", 0)},
        "activeBroadcasts": (await session.execute(select(func.count(AdminBroadcastDB.id)).where(AdminBroadcastDB.is_active == True))).scalar() or 0,  # noqa: E712
        "files": (await session.execute(select(func.count(EventFileDB.id)).where(EventFileDB.is_active == True))).scalar() or 0,  # noqa: E712
        "lists": (await session.execute(select(func.count(ContactListDB.id)))).scalar() or 0,
        # still restorable (expired rows wait for the purge job, they're not "recently deleted")
        "recentlyDeleted": (await session.execute(select(func.count(DeletedContactDB.id)).where(DeletedContactDB.purge_after >= now))).scalar() or 0,
        "range": period,
        # kept for older clients (same real numbers)
        "totalScans": total, "tierALeads": hot, "tierBLeads": warm, "tierCLeads": cool + cold,
        "timestamp": now.isoformat(),
    }


# ---- Audience mix: who the team is meeting, read from each contact's job title, company and the
# industry found by AI research. Rules run in order; the first match wins.

# (key, label, pattern) — most senior first; student/intern is checked before everything else
SENIORITY_RULES = [
    ("student", "Student / intern", r"\b(intern|internship|student|trainee|apprentice)\b"),
    ("owner", "Founder / owner", r"\b(co-?founder|founder|proprietor)\b|(?<!product )(?<!process )(?<!account )\bowner\b"
                                 r"|(?<!business )\bpartner\b(?!\s*(manager|success|marketing|sales|development|lead|executive|programs?|relations))"),
    ("clevel", "C-suite", r"\b(chief|ceo|cto|cfo|coo|cmo|cio|cro|cpo|cso|ciso|cdo|cco|chro|cxo)\b|\bmanaging director\b|(?<!vice[\s-])\bpresident\b"),
    ("vp", "VP", r"\b(vp|svp|evp|avp)\b|vice[\s-]president"),
    ("director", "Director / head", r"\b(director|head|gm)\b|general manager"),
    ("manager", "Manager / lead", r"\b(manager|mgr|lead|supervisor|team leader)\b"),
]
SENIORITY_ORDER = [k for k, _, _ in SENIORITY_RULES[1:]] + ["ic", "student", "unknown"]
SENIORITY_LABEL = {**{k: l for k, l, _ in SENIORITY_RULES}, "ic": "Individual contributor", "unknown": "No job title"}

# a title names its area before its level ("VP Sales" → Sales); product marketing is marketing,
# a data engineer is data, a sales engineer is sales — hence the order
FUNCTION_RULES = [
    ("sales", "Sales & BD", r"\b(sales|business development|bd|bdm|bdr|sdr|account executive|account manager|account owner|key account|revenue|commercial|partnerships?|channel)\b|\bcro\b"),
    ("marketing", "Marketing", r"\b(marketing|growth|brand|demand gen|content|communications?|pr|seo|social media|events?)\b|\bcmo\b"),
    ("data", "Data & AI", r"\b(data|analytics|analyst|ai|ml|machine learning|bi|business intelligence|scientist)\b|\bcdo\b"),
    ("product", "Product & design", r"\b(product|design|designer|ux|ui)\b|\bcpo\b"),
    ("engineering", "Engineering & IT", r"\b(engineer|engineering|developer|software|it|technology|technical|tech|architect|devops|infrastructure|security|cloud|systems?|chief information)\b|\b(cto|cio|ciso)\b"),
    ("operations", "Operations", r"\b(operations|operating|ops|supply chain|logistics|procurement|purchasing|sourcing|manufacturing|production|quality)\b|\bcoo\b"),
    ("finance", "Finance", r"\b(finance|financial|accounting|accountant|accounts|controller|treasury|audit|tax)\b|\bcfo\b"),
    ("people", "HR & people", r"\b(hr|human resources|people|talent|recruit(er|ing|ment)?|hiring)\b|\bchro\b"),
    ("success", "Customer success", r"\b(customer success|customer service|support|client services|customer experience|cx)\b"),
    ("leadership", "Leadership", r"\b(ceo|chief executive|founder|co-?founder|owner|partner|president|managing director|executive director|general manager|gm|chairman|chairperson|proprietor)\b"),
]
FUNCTION_LABEL = {**{k: l for k, l, _ in FUNCTION_RULES}, "other": "Other roles", "unknown": "No job title"}

# industry text from research is free-form ("B2B SaaS", "Information Technology & Services"),
# so the common ones are folded into one name; anything else keeps its own (tidied) name.
# Verticals come before the horizontal tech names, so "healthcare SaaS" counts as healthcare.
INDUSTRY_RULES = [
    ("Financial services", r"financ|bank|fintech|insur|invest|venture capital|private equity|payments?|wealth|accounting|lending"),
    ("Healthcare & life sciences", r"health|medic|pharma|biotech|life science|hospital|clinic|diagnos"),
    ("Education", r"educat|edtech|e-?learning|universit|school"),
    ("Retail & e-commerce", r"retail|e-?commerce|consumer goods|fmcg|cpg|fashion|apparel|food|beverage"),
    ("Marketing & media", r"marketing|advertis|\bmedia\b|publish|entertainment|agency|public relations"),
    ("Telecom", r"telecom|telco|wireless"),
    ("Energy & utilities", r"energy|\boil\b|\bgas\b|utilit|renewable|solar"),
    ("Logistics & transport", r"logistic|supply chain|transport|shipping|freight|aviation|airline|mobility"),
    ("Automotive", r"automotive|\bauto\b|vehicle"),
    ("Real estate & construction", r"real estate|property|construction|built environment"),
    ("Government & non-profit", r"government|public sector|non-?profit|\bngo\b|charity"),
    ("Travel & hospitality", r"travel|hospitality|hotel|tourism|restaurant"),
    ("Software & SaaS", r"saas|software|cloud|cyber ?security|internet|platform"),
    ("Data & AI", r"\bai\b|artificial intelligence|machine learning|analytics|\bdata\b"),
    ("IT services", r"\bit\b|information technology|technology services|systems integrat|outsourc|\btech"),
    ("Manufacturing & industrial", r"manufactur|industrial|engineering|chemical|machinery|electronics|semiconductor|aerospace|defen[cs]e"),
    ("Professional services", r"consult|professional services|legal|\blaw\b|staffing|recruit|hr services"),
]
_COMPANY_SUFFIX = re.compile(r"[\s,.]+(inc|incorporated|llc|ltd|limited|pvt|private|corp|corporation|co|company|gmbh|plc|sa|ag|bv|pte|llp)\.?$", re.I)
_BLANK = {"", "n/a", "na", "none", "null", "unknown", "-", "--"}


def _classify_title(title: Optional[str]) -> tuple:
    """(function key, seniority key) for a job title; ("unknown", "unknown") when there isn't one."""
    t = (title or "").strip().lower()
    if t in _BLANK:
        return "unknown", "unknown"
    seniority = next((k for k, _, p in SENIORITY_RULES if re.search(p, t)), "ic")
    function = next((k for k, _, p in FUNCTION_RULES if re.search(p, t)), "other")
    return function, seniority


def _industry_of(research: Any) -> Optional[str]:
    """The industry an AI research run found for the contact's company, folded into a common name."""
    if not isinstance(research, dict) or research.get("parse_error"):
        return None
    company = research.get("company") if isinstance(research.get("company"), dict) else {}
    raw = str(company.get("industry") or research.get("industry") or "").strip()
    if raw.lower() in _BLANK:
        return None
    low = raw.lower()
    for name, pattern in INDUSTRY_RULES:
        if re.search(pattern, low):
            return name
    # unrecognised: keep the first part of the description ("Agritech / farming" → "Agritech")
    first = re.split(r"\s*[/,;(|]\s*", raw)[0].strip()[:40]
    return first[:1].upper() + first[1:] if first else None


def _company_key(name: Optional[str]) -> Optional[str]:
    n = (name or "").strip()
    if n.lower() in _BLANK:
        return None
    n = _COMPANY_SUFFIX.sub("", _COMPANY_SUFFIX.sub("", n)).strip(" ,.")  # "Acme Pvt. Ltd." → "acme"
    return re.sub(r"\s+", " ", n).lower() or None


def _segment_rows(groups: Dict[str, Dict[str, Any]], limit: Optional[int] = 7, order: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Groups → [{key, label, count, hot, hotWarm, avgScore}]: biggest first (or in `order`), the rest as "other".
    hot = scoring ≥ 85, hotWarm = scoring ≥ 70 (score_levels)."""
    row = lambda k, g: {"key": k, "label": g["label"], "count": g["count"], "hot": g["hot"], "hotWarm": g["hotWarm"],
                        "avgScore": round(g["scoreSum"] / g["scoreN"], 1) if g["scoreN"] else None}
    if order:
        return [row(k, groups[k]) for k in order if k in groups]
    ranked = sorted(groups.items(), key=lambda kv: (-kv[1]["count"], -kv[1]["hot"], kv[1]["label"]))
    out = [row(k, g) for k, g in ranked[:limit]]
    rest = [g for _, g in ranked[limit:]] if limit else []
    if rest:
        merged = {"label": f"{len(rest)} more", "count": 0, "hot": 0, "hotWarm": 0, "scoreSum": 0, "scoreN": 0}
        for g in rest:
            for f in ("count", "hot", "hotWarm", "scoreSum", "scoreN"):
                merged[f] += g[f]
        out.append({**row("_other", merged), "other": True})
    return out


# ---- Industry, worked out automatically: every company the team meets is classified by AI (name +
# work-email domain + the titles met there), once per company, in the background. Nobody has to run
# research for the dashboard to know a contact's industry; a research run, when there is one, wins.
INDUSTRY_NAMES = [name for name, _ in INDUSTRY_RULES]
INDUSTRY_BATCH = 40              # companies per AI call
INDUSTRY_RETRY_AFTER = timedelta(days=7)  # "couldn't tell" answers are asked again after this
_FREE_MAIL = {"gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.in", "ymail.com", "hotmail.com", "outlook.com", "live.com",
              "msn.com", "icloud.com", "me.com", "aol.com", "proton.me", "protonmail.com", "zoho.com", "gmx.com", "mail.com",
              "rediffmail.com", "yandex.com", "qq.com", "163.com"}
_industry_lock = asyncio.Lock()
_industry_tasks: set = set()
_industry_error: Dict[str, Any] = {"reason": None, "at": None}  # the last run's failure (cleared on success)


def _work_domain(email: Optional[str]) -> Optional[str]:
    e = (email or "").strip().lower()
    if "@" not in e or e in _BLANK:
        return None
    d = e.rsplit("@", 1)[1].strip(". ")
    return d if d and "." in d and d not in _FREE_MAIL else None


def _industry_key(company: Optional[str], email: Optional[str]) -> Optional[str]:
    """What a contact's industry is looked up by: its company, else its work-email domain."""
    ck = _company_key(company)
    if ck:
        return ck
    d = _work_domain(email)
    return f"@{d}" if d else None


def _canonical_industry(answer: Any) -> Optional[str]:
    a = str(answer or "").strip()
    if not a or a.lower() in _BLANK or a.lower().startswith("unknown"):
        return None
    for name in INDUSTRY_NAMES:
        if a.lower() == name.lower():
            return name
    return _industry_of({"industry": re.sub(r"^other\s*[:\-–]\s*", "", a, flags=re.I)})


async def classify_company_industries(max_companies: int = 1000) -> int:
    """Classify every company (contacts' company or work domain) that has no industry yet.
    One run at a time; returns how many companies got an answer. Silently does nothing without AI."""
    import ai_client
    if not ai_client.ai_configured or _industry_lock.locked():
        return 0
    factory = get_session_factory()
    if not factory:
        return 0
    async with _industry_lock:
        async with factory() as session:
            now = datetime.now(timezone.utc)
            known = {k: (ind, at) for k, ind, at in (await session.execute(
                select(CompanyIndustryDB.company_key, CompanyIndustryDB.industry, CompanyIndustryDB.updated_at))).all()}
            groups: Dict[str, Dict[str, Any]] = {}
            for company, email, title in (await session.execute(select(ContactDB.company_name, ContactDB.email, ContactDB.job_title))).all():
                key = _industry_key(company, email)
                if not key:
                    continue
                g = groups.setdefault(key, {"name": (company or "").strip() if _company_key(company) else key[1:], "domains": set(), "titles": set()})
                if (d := _work_domain(email)):
                    g["domains"].add(d)
                if title and title.strip().lower() not in _BLANK and len(g["titles"]) < 3:
                    g["titles"].add(title.strip()[:60])

            def due(k: str) -> bool:
                if k not in known:
                    return True
                ind, at = known[k]
                at = (at if at is None or at.tzinfo else at.replace(tzinfo=timezone.utc))
                return ind is None and (at is None or now - at > INDUSTRY_RETRY_AFTER)

            todo = [k for k in groups if due(k)][:max_companies]
            done = 0
            for i in range(0, len(todo), INDUSTRY_BATCH):
                batch = todo[i:i + INDUSTRY_BATCH]
                lines = "\n".join(
                    f"{n}. {groups[k]['name'][:80]}"
                    + (f" — web: {', '.join(sorted(groups[k]['domains'])[:2])}" if groups[k]["domains"] else "")
                    + (f" — people met: {', '.join(sorted(groups[k]['titles']))}" if groups[k]["titles"] else "")
                    for n, k in enumerate(batch, 1))
                prompt = (
                    "Classify each company into the ONE industry it mainly operates in. Use what is commonly known about the "
                    "company and the hints given (web domain, job titles). Pick exactly one of:\n"
                    + "\n".join(f"- {name}" for name in INDUSTRY_NAMES)
                    + "\nIf none fits, answer \"Other: <2-3 word industry>\". If you truly cannot tell, answer \"Unknown\".\n\n"
                    f"Companies:\n{lines}\n\nReturn ONLY JSON mapping each number to its answer, e.g. {{\"1\": \"Software & SaaS\", \"2\": \"Unknown\"}}"
                )
                try:
                    text, _ = await ai_client.chat([{"role": "user", "content": prompt}], max_tokens=60 + 24 * len(batch),
                                                   temperature=0, json_mode=True, timeout=45, title="Event Scout Industries")
                    answers = ai_client.parse_json(text)
                except Exception as e:  # AIError, bad JSON — leave the rest for the next run
                    print(f"[INDUSTRY] classification stopped: {e}")
                    _industry_error.update(reason=getattr(e, "reason", None) or "unavailable", at=datetime.now(timezone.utc))
                    break
                _industry_error.update(reason=None, at=None)
                for n, k in enumerate(batch, 1):
                    # a company the model skipped is saved as "couldn't tell" too, so it's retried in a
                    # week rather than asked again on every dashboard refresh
                    await session.merge(CompanyIndustryDB(company_key=k[:255], company_name=groups[k]["name"][:255],
                                                          industry=_canonical_industry(answers.get(str(n))), source="ai", updated_at=now))
                    done += 1
                await session.commit()
            if done:
                print(f"[INDUSTRY] classified {done} companies")
                # consoles re-sort their industry tiles / audience mix straight away
                await broadcast_live_event("industries_updated", {"classified": done})
            return done


def industry_stalled() -> Optional[str]:
    """The reason the last run failed, if it failed in the last 10 minutes (then don't retry yet)."""
    at = _industry_error["at"]
    return _industry_error["reason"] if at and datetime.now(timezone.utc) - at < timedelta(minutes=10) else None


def kick_industry_classification() -> None:
    """Start a background classification run unless one is going or the last one just failed."""
    if _industry_lock.locked() or industry_stalled():
        return
    t = asyncio.create_task(classify_company_industries())
    _industry_tasks.add(t)
    t.add_done_callback(_industry_tasks.discard)


async def _dashboard_range(session: AsyncSession, days: int, tz_offset: int) -> Dict[str, Any]:
    """Per-day series and period-over-period numbers for the dashboard charts (admin's local days)."""
    shift = timedelta(minutes=tz_offset)  # local = utc - offset (JS getTimezoneOffset convention)
    now = datetime.now(timezone.utc)
    aware = lambda dt: (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)) if dt else None
    local_day = lambda dt: (aware(dt) - shift).date()
    today = local_day(now)
    start = today - timedelta(days=days - 1)
    prev_start = start - timedelta(days=days)

    # scores / "scored" counts are the admin console's own (admin_contact_scores), like the contacts grid
    rows = (await session.execute(_with_admin_score(
        select(ContactDB.user_id, ContactDB.created_at, ADMIN_SCORE, ContactDB.source,
               ContactDB.event_name, AdminContactScoreDB.scored_at,
               ContactDB.id, ContactDB.job_title, ContactDB.company_name, ContactDB.email,
               ContactDB.updated_at)))).all()
    all_users = (await session.execute(select(UserDB))).scalars().all()
    names = {u.id: u.name for u in all_users}
    admin_ids = {u.id for u in all_users if u.is_admin}  # "reps" = non-admin accounts everywhere on the dashboard
    # industry: a research run on the contact (newest wins), else the company's automatic AI industry
    industry_of: Dict[Any, str] = {}
    for cid, research in (await session.execute(
            select(ContactPipelineDB.contact_id, ContactPipelineDB.research_data).order_by(ContactPipelineDB.updated_at.asc().nulls_first()))).all():
        ind = _industry_of(research)
        if ind:
            industry_of[cid] = ind
    company_industry: Dict[str, Optional[str]] = dict((await session.execute(
        select(CompanyIndustryDB.company_key, CompanyIndustryDB.industry))).all())
    pending_keys: set = set()  # companies met in the period the classifier hasn't answered yet

    series = {start + timedelta(days=i): {"added": 0, "hot": 0, "scored": 0, "scoreSum": 0, "scoreN": 0} for i in range(days)}
    cur = {"added": 0, "hot": 0, "scored": 0, "scoreSum": 0, "scoreN": 0}
    prev = {"added": 0, "hot": 0, "scored": 0, "scoreSum": 0, "scoreN": 0}
    reps: Dict[Any, Dict[str, Any]] = {}
    sources: Dict[str, int] = {}
    events: Dict[str, int] = {}
    histogram = [0] * 10
    active_today = set()
    today_by_hour = [0] * 24  # contacts added today, by the admin's local hour
    reps_today: Dict[Any, int] = {}
    seg: Dict[str, Dict[str, Dict[str, Any]]] = {"industry": {}, "function": {}, "seniority": {}, "company": {}}

    def tally(dim: str, key: str, label: str, score: Optional[int]) -> None:
        g = seg[dim].setdefault(key, {"label": label, "count": 0, "hot": 0, "hotWarm": 0, "scoreSum": 0, "scoreN": 0})
        g["count"] += 1
        if score is not None:
            g["scoreSum"] += score
            g["scoreN"] += 1
            g["hot"] += is_hot(score)
            g["hotWarm"] += is_hot_or_warm(score)

    for uid, created, score, source, event, scored_at, cid, title, company, email, updated in rows:
        if score is not None:
            histogram[min(9, max(0, int(score) // 10))] += 1
        d = local_day(created) if created else None
        sd = local_day(scored_at) if score is not None and scored_at else None
        if sd is None and score is not None:
            # a score row without scored_at: the contact's last change is when it was scored
            sd = local_day(updated or created) if (updated or created) else None
        if sd is not None:
            if sd in series:
                series[sd]["scored"] += 1
            if start <= sd <= today:
                cur["scored"] += 1
            elif prev_start <= sd < start:
                prev["scored"] += 1
        if d is None:
            continue
        if d == today:
            today_by_hour[(aware(created) - shift).hour] += 1
            if uid not in admin_ids:
                active_today.add(uid)
                reps_today[uid] = reps_today.get(uid, 0) + 1
        bucket = cur if start <= d <= today else prev if prev_start <= d < start else None
        if bucket is None:
            continue
        bucket["added"] += 1
        if score is not None:
            bucket["scoreSum"] += score
            bucket["scoreN"] += 1
            if is_hot(score):
                bucket["hot"] += 1
        if bucket is cur:
            s = series[d]
            s["added"] += 1
            if score is not None:
                s["scoreSum"] += score
                s["scoreN"] += 1
                if is_hot(score):
                    s["hot"] += 1
            r = reps.setdefault(uid, {"userId": str(uid), "name": names.get(uid, "Unknown"), "added": 0, "hot": 0, "scoreSum": 0, "scoreN": 0})
            r["added"] += 1
            if score is not None:
                r["scoreSum"] += score
                r["scoreN"] += 1
                r["hot"] += is_hot(score)
            sources[source or "unknown"] = sources.get(source or "unknown", 0) + 1
            if event and event.strip():
                events[event.strip()] = events.get(event.strip(), 0) + 1
            fn, level = _classify_title(title)
            tally("function", fn, FUNCTION_LABEL[fn], score)
            tally("seniority", level, SENIORITY_LABEL[level], score)
            ik = _industry_key(company, email)
            ind = industry_of.get(cid) or (company_industry.get(ik) if ik else None)
            if ind:
                tally("industry", ind.lower(), ind, score)
            elif ik and ik not in company_industry:
                pending_keys.add(ik)
            ck = _company_key(company)
            if ck:
                tally("company", ck, company.strip(), score)  # label = the first spelling seen

    import ai_client
    if pending_keys and ai_client.ai_configured:
        kick_industry_classification()  # the dashboard refreshes while `pending` > 0
    avg = lambda b: round(b["scoreSum"] / b["scoreN"], 1) if b["scoreN"] else None
    known = lambda dim: sum(g["count"] for k, g in seg[dim].items() if k != "unknown")
    companies = seg["company"]
    delta = lambda a, b: None if not b else round((a - b) / b * 100)
    top_reps = sorted(reps.values(), key=lambda r: (-r["added"], -r["hot"]))[:6]
    today_profiles = dict((await session.execute(select(UserProfileDB.user_id, UserProfileDB.profile_data)
                                                 .where(UserProfileDB.user_id.in_(list(reps_today))))).all()) if reps_today else {}
    today_start = datetime(today.year, today.month, today.day, tzinfo=timezone.utc) + shift  # local midnight, in UTC
    return {
        "todayStart": today_start.isoformat(),
        "todayByHour": today_by_hour,
        "repsToday": [{"userId": str(uid), "name": names.get(uid, "Unknown"),
                       "photoUrl": _user_photo(today_profiles.get(uid)),
                       "count": n} for uid, n in sorted(reps_today.items(), key=lambda kv: -kv[1])],
        "days": days,
        "from": start.isoformat(),
        "to": today.isoformat(),
        "series": [{"date": d.isoformat(), "added": s["added"], "hot": s["hot"], "scored": s["scored"], "avgScore": avg(s)} for d, s in series.items()],
        "totals": {"added": cur["added"], "hot": cur["hot"], "scored": cur["scored"], "avgScore": avg(cur)},
        "previous": {"added": prev["added"], "hot": prev["hot"], "scored": prev["scored"], "avgScore": avg(prev)},
        "change": {"added": delta(cur["added"], prev["added"]), "hot": delta(cur["hot"], prev["hot"]), "scored": delta(cur["scored"], prev["scored"]),
                   "avgScore": None if avg(cur) is None or avg(prev) is None else round(avg(cur) - avg(prev), 1)},
        "activeRepsToday": len(active_today),
        "scoreHistogram": [{"from": i * 10, "to": i * 10 + 9 if i < 9 else 100, "count": n} for i, n in enumerate(histogram)],
        "topReps": [{"userId": r["userId"], "name": r["name"], "added": r["added"], "hot": r["hot"],
                     "avgScore": round(r["scoreSum"] / r["scoreN"], 1) if r["scoreN"] else None} for r in top_reps],
        "sources": [{"source": k, "count": v} for k, v in sorted(sources.items(), key=lambda kv: -kv[1])],
        "events": [{"event": k, "count": v} for k, v in sorted(events.items(), key=lambda kv: -kv[1])[:6]],
        # contacts added in the period, split by industry / function / seniority / company.
        # `known` = how many of them the split could be worked out for (title on file, researched, …)
        "audience": {
            "total": cur["added"],
            # six groups + "N more" (the chart has six colour slots and a gray)
            # `pending` = companies the AI is still classifying; `ai` = whether it can (key configured)
            "industry": {"known": known("industry"), "rows": _segment_rows(seg["industry"], limit=6),
                         "pending": len(pending_keys) if ai_client.ai_configured else 0, "ai": bool(ai_client.ai_configured),
                         "error": industry_stalled()},
            "function": {"known": known("function"), "rows": _segment_rows(seg["function"], limit=6)},
            "seniority": {"known": known("seniority"), "rows": _segment_rows(seg["seniority"], order=SENIORITY_ORDER)},
            "company": {"known": known("company"), "rows": _segment_rows(companies, limit=6), "distinct": len(companies),
                        "repeat": sum(1 for g in companies.values() if g["count"] > 1)},
        },
    }


# ---- Smart lists (Admin console → Lists): every contact grouped by industry, seniority, role and
# company with the same rules as the dashboard's audience mix — but over ALL contacts, not a period.

SEGMENT_DIMS = ("industry", "seniority", "function", "company")


async def _contact_segments(session: AsyncSession) -> tuple:
    """({contact_id: {industry, function, seniority, company}}, {contact_id: score}, pending industry keys)."""
    industry_of: Dict[Any, str] = {}
    for cid, research in (await session.execute(
            select(ContactPipelineDB.contact_id, ContactPipelineDB.research_data).order_by(ContactPipelineDB.updated_at.asc().nulls_first()))).all():
        ind = _industry_of(research)
        if ind:
            industry_of[cid] = ind
    company_industry: Dict[str, Optional[str]] = dict((await session.execute(
        select(CompanyIndustryDB.company_key, CompanyIndustryDB.industry))).all())
    out: Dict[Any, Dict[str, Any]] = {}
    scores: Dict[Any, Optional[int]] = {}
    pending: set = set()
    for cid, title, company, email, score in (await session.execute(_with_admin_score(
            select(ContactDB.id, ContactDB.job_title, ContactDB.company_name, ContactDB.email, ADMIN_SCORE)))).all():
        fn, level = _classify_title(title)
        ik = _industry_key(company, email)
        ind = industry_of.get(cid) or (company_industry.get(ik) if ik else None)
        if not ind and ik and ik not in company_industry:
            pending.add(ik)
        ck = _company_key(company)
        out[cid] = {
            "industry": (ind.lower(), ind) if ind else None,
            "function": (fn, FUNCTION_LABEL[fn]),
            "seniority": (level, SENIORITY_LABEL[level]),
            "company": (ck, (company or "").strip()) if ck else None,
        }
        scores[cid] = score
    return out, scores, pending


@api_router.get("/admin/segments", dependencies=[Depends(require_admin)])
async def get_segments(session: AsyncSession = Depends(get_db)):
    """Smart lists: how many contacts fall in each industry / seniority / role / company, with hot
    counts and average score. Industry comes from AI research or the automatic company classifier
    (`pending` = companies still being classified; they're picked up on the next refresh)."""
    segs, scores, pending = await _contact_segments(session)
    groups: Dict[str, Dict[str, Dict[str, Any]]] = {d: {} for d in SEGMENT_DIMS}
    for cid, s in segs.items():
        score = scores.get(cid)
        for dim in SEGMENT_DIMS:
            kv = s[dim]
            if not kv:
                continue
            g = groups[dim].setdefault(kv[0], {"label": kv[1], "count": 0, "hot": 0, "hotWarm": 0, "scoreSum": 0, "scoreN": 0})
            g["count"] += 1
            if score is not None:
                g["scoreSum"] += score
                g["scoreN"] += 1
                # hot used to count ≥ 70 here (hot + warm) while every other page counts ≥ 85
                g["hot"] += is_hot(score)
                g["hotWarm"] += is_hot_or_warm(score)
    import ai_client
    if pending and ai_client.ai_configured:
        kick_industry_classification()
    known = lambda dim: sum(g["count"] for k, g in groups[dim].items() if k != "unknown")
    return {
        "total": len(segs),
        "industry": {"rows": _segment_rows(groups["industry"], limit=None), "known": known("industry"),
                     "pending": len(pending) if ai_client.ai_configured else 0, "ai": bool(ai_client.ai_configured), "error": industry_stalled()},
        "seniority": {"rows": _segment_rows(groups["seniority"], order=SENIORITY_ORDER), "known": known("seniority")},
        "function": {"rows": _segment_rows(groups["function"], limit=None), "known": known("function")},
        "company": {"rows": _segment_rows(groups["company"], limit=None), "known": known("company"), "distinct": len(groups["company"])},
    }


@api_router.get("/admin/segments/{dim}/contacts", dependencies=[Depends(require_admin)])
async def get_segment_contacts(dim: str, key: str = Query(..., min_length=1), session: AsyncSession = Depends(get_db)):
    """The contacts in one smart list (dim = industry | seniority | function | company, key from /admin/segments)."""
    if dim not in SEGMENT_DIMS:
        raise HTTPException(status_code=400, detail=f"dim must be one of: {', '.join(SEGMENT_DIMS)}")
    segs, _, _ = await _contact_segments(session)
    ids = [cid for cid, s in segs.items() if s[dim] and s[dim][0] == key]
    if not ids:
        return []
    rows = (await session.execute(select(ContactDB).where(ContactDB.id.in_(ids)).order_by(ContactDB.name))).scalars().all()
    return await console_scores(session, None, [format_contact(c) for c in rows])


# ==================== SSE LIVE EVENT STREAM ====================

@api_router.post("/admin/live/ticket")
async def live_stream_ticket(admin_id: uuid.UUID = Depends(require_admin)):
    """A one-use ticket (60 s) to open GET /api/live?ticket=… without putting the token in the URL."""
    return {"ticket": identity.issue_stream_ticket(admin_id), "expiresIn": identity.STREAM_TICKET_SECONDS}


@api_router.get("/live", dependencies=[Depends(require_admin_stream)])
async def live_event_stream(request: Request):
    """
    Server-Sent Events (SSE) feed for the Admin Ops Console header marquee.
    Yields real-time events when reps scan cards or leads are scored on mobile.
    """
    async def event_generator():
        queue: asyncio.Queue = new_sse_queue()
        sse_subscribers.append(queue)
        try:
            # Initial connect payload
            initial_msg = {
                "type": "connected",
                "data": {
                    "status": "online",
                    "channel": "eventscout:live_telemetry",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            }
            yield f"data: {json.dumps(initial_msg)}\n\n"

            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=12.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Heartbeat comment to keep HTTP connection warm
                    yield f": heartbeat\n\n"
        finally:
            if queue in sse_subscribers:
                sse_subscribers.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ==================== SYNC (OFFLINE-FIRST MOBILE ENGINE) ====================

class OutboxItemModel(BaseModel):
    id: str
    entityType: str
    entityId: str
    action: str
    payload: str
    createdAt: Optional[str] = None


class SyncPushBatchModel(BaseModel):
    items: List[OutboxItemModel]


def _require_signed_in(me: Optional[uuid.UUID]) -> uuid.UUID:
    return identity.require_rep(me)


def _outbox_time(value: Optional[str]) -> Optional[datetime]:
    """When the phone queued a change: ISO 8601 or epoch milliseconds. Future / unreadable → None."""
    if not value:
        return None
    try:
        if str(value).strip().isdigit():
            dt = datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)
        else:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None
    return dt if dt <= datetime.now(timezone.utc) + timedelta(minutes=5) else None


@api_router.post("/sync/push")
async def sync_push(
    batch: SyncPushBatchModel,
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """
    Receives batched outbox items from the offline mobile sync engine and applies them to the
    signed-in user's contacts (create, update or delete), broadcasting SSE live events.
    """
    owner = _require_signed_in(me)
    acknowledged_ids = []
    conflicts = []  # applied-or-skipped for good (e.g. someone else's contact): acknowledged
    failed = []     # couldn't be applied this time: NOT acknowledged, so the device keeps and retries it
    blocked = set()  # entities whose earlier change failed in this batch: later edits must wait for it

    for item in batch.items:
        if item.entityType != "contact":
            acknowledged_ids.append(item.id)
            continue
        if item.entityId in blocked:
            failed.append({"id": item.id, "reason": "waiting on an earlier change"})
            continue
        try:
            try:
                data = json.loads(item.payload) if item.payload else {}
            except ValueError:
                data = None
            if not isinstance(data, dict):
                raise HTTPException(status_code=400, detail="The change is not readable")
            try:
                cid = uuid.UUID(item.entityId)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid contact id")
            existing, archived = await contact_store.find_by_id(session, cid)

            if existing is not None and existing.user_id != owner:
                # Someone else's contact ID: never touch it.
                conflicts.append({"id": item.id, "reason": "not_owner"})
                acknowledged_ids.append(item.id)
                continue

            action = (item.action or "").lower()
            if action == "update" and existing is None:
                # Deleted on the web / another phone (or never uploaded): an edit must not bring it
                # back as a near-empty contact. Acknowledged so the phone stops retrying.
                conflicts.append({"id": item.id, "reason": "not_found", "entityId": item.entityId})
                acknowledged_ids.append(item.id)
                continue
            if action == "delete":
                if existing is not None:
                    await contact_store.delete_contact(session, existing)  # restorable for 30 days
                acknowledged_ids.append(item.id)
                continue
            if existing is None and archived is not None:
                # A create retried after the contact was already uploaded and then deleted (e.g. by an
                # admin): don't bring it back. It's in Recently deleted if someone wants it.
                conflicts.append({"id": item.id, "reason": "deleted", "entityId": item.entityId})
                acknowledged_ids.append(item.id)
                continue

            # Tolerant writes (contact_store): a bad photo / LinkedIn / score is dropped, notes over the
            # limit are shortened, a nameless capture is "Unnamed contact"; an offline edit vs. a newer
            # edit made elsewhere keeps the newer value (notes merged). Anything changed is reported.
            if existing is None:
                saved = await contact_store.create_contact(
                    session, owner, data, strict=False, contact_id=cid, source=data.get("source") or "scan",
                    # when it was captured on the phone, not when the phone got signal again
                    created_at=_outbox_time(item.createdAt) or datetime.now(timezone.utc))
            else:
                base = data.get("_base") if isinstance(data.get("_base"), dict) else None
                saved = await contact_store.update_contact(session, existing, data, strict=False, base=base)
            acknowledged_ids.append(item.id)
            if saved.clashes:
                conflicts.append({"id": item.id, "entityId": item.entityId, "reason": "changed_elsewhere",
                                  "fields": [c["field"] for c in saved.clashes], "message": saved.message})
            elif saved.notes_shortened:
                # applied (acknowledged) — only the notes were cut to the limit
                conflicts.append({"id": item.id, "entityId": item.entityId, "reason": "notes_shortened",
                                  "applied": True, "fields": ["notes"], "message": saved.message})
        except HTTPException as err:
            await session.rollback()
            if 400 <= err.status_code < 500:
                # The change itself is invalid: retrying can never succeed, so acknowledge it with the
                # reason (shown on the phone) instead of leaving it stuck in the queue forever.
                print(f"[SYNC PUSH] Item {item.id} rejected: {err.detail}")
                conflicts.append({"id": item.id, "reason": str(err.detail)[:200], "entityId": item.entityId})
                acknowledged_ids.append(item.id)
            else:
                blocked.add(item.entityId)
                failed.append({"id": item.id, "reason": str(err.detail)[:200]})
        except Exception as err:
            await session.rollback()  # keep the session usable for the rest of the batch
            print(f"[SYNC PUSH ERROR] Item {item.id}: {err}")
            # Used to be acknowledged anyway, which silently threw the change away.
            blocked.add(item.entityId)
            failed.append({"id": item.id, "reason": "The server couldn't apply this change"})

    return {
        "acknowledgedIds": acknowledged_ids,
        "conflicts": conflicts,
        "failed": failed,
    }


@api_router.get("/sync/pull")
async def sync_pull(
    cursor: Optional[str] = Query(None),
    eventId: Optional[str] = Query(None),
    limit: int = Query(100),
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """Lets a signed-in device pull its own latest server contacts for the local cache."""
    owner = _require_signed_in(me)
    limit = max(1, min(limit, 500))
    # cursor = how many rows were already sent (it used to be always None, so only the newest
    # 100 contacts ever reached the device).
    try:
        offset = max(0, int(cursor)) if cursor else 0
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid cursor")
    base = select(ContactDB).where(ContactDB.user_id == owner)
    if eventId:
        base = base.where(await event_filter(session, ContactDB.event_name, eventId))
    total = (await session.execute(select(func.count()).select_from(base.subquery()))).scalar() or 0
    rows = (await session.execute(
        base.order_by(ContactDB.created_at.desc(), ContactDB.id).offset(offset).limit(limit)
    )).scalars().all()
    next_offset = offset + len(rows)

    return {
        "contacts": [app_view(format_contact(r)) for r in rows],
        "cursor": str(next_offset) if next_offset < total else None,
        "total": total,
        "serverTimestamp": datetime.now(timezone.utc).isoformat(),
    }


# ==================== ACTIVITY & AUDIT TIMELINE ====================

def _parse_when(value: Optional[str], name: str) -> Optional[datetime]:
    """An ISO 8601 instant from the query string (naive = UTC), or None."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid {name}: use an ISO 8601 date-time")
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def _when(value) -> Optional[datetime]:
    """Any stored timestamp (datetime or ISO string, naive = UTC) as an aware UTC datetime."""
    if not value:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


async def _collect_activity(session: AsyncSession, limit: int, start: Optional[datetime] = None, end: Optional[datetime] = None) -> list:
    """The workspace timeline, newest first, from the real tables: contacts added, AI scores, pipelines,
    announcements, files, deletions / merges and new accounts. `start` / `end` (UTC, end exclusive) keep
    only what happened in that window; `limit` caps each source and the merged result."""
    names = {u.id: u.name for u in (await session.execute(select(UserDB))).scalars().all()}
    who = lambda uid: names.get(uid) or "Someone"

    def window(col):
        # SQL-side narrowing on the source's own time column (the exact check happens below)
        conds = []
        if start is not None:
            conds.append(col >= start)
        if end is not None:
            conds.append(col < end)
        return conds

    items = []
    # the photo (a base64 JPEG per row) isn't needed for the timeline
    contacts = (await session.execute(select(ContactDB).options(defer(ContactDB.photo_base64)).where(*window(ContactDB.created_at))
                                      .order_by(ContactDB.created_at.desc()).limit(limit))).scalars().all()
    for c in contacts:
        items.append({"id": f"contact-{c.id}", "type": "contact_added", "title": f"{who(c.user_id)} added {c.name}",
                      "subtitle": " · ".join(x for x in [_present(c.company_name), c.event_name or None, c.source or None] if x),
                      "actor": who(c.user_id), "timestamp": _iso(c.created_at),
                      "meta": {"contactId": str(c.id), "userId": str(c.user_id) if c.user_id else None}})
    # a score's time lives in its breakdown; it's never later than the contact's last update
    scored_q = select(ContactDB).options(defer(ContactDB.photo_base64)).where(ContactDB.lead_score.is_not(None))
    if start is not None:
        scored_q = scored_q.where(or_(ContactDB.updated_at >= start, ContactDB.updated_at.is_(None)))
    scored = (await session.execute(scored_q.order_by(ContactDB.updated_at.desc()).limit(limit))).scalars().all()
    for c in scored:
        meta = (c.lead_score_breakdown or {}).get("_meta") or {}
        at = meta.get("scored_at")
        if not at:
            continue  # scored before score times were recorded
        # the rep's own score (made in the app against the rep's profile) — labelled so it isn't read as
        # the console's score, which the contact page shows
        items.append({"id": f"score-{c.id}-{at}", "type": "contact_scored", "title": f"{c.name}: rep score {c.lead_score}",
                      "subtitle": f"{(level_for(c.lead_score) or '').capitalize()} lead · scored in {who(c.user_id)}'s app",
                      "actor": "AI scoring", "timestamp": at, "meta": {"contactId": str(c.id), "score": c.lead_score, "scoreKind": "rep"}})
    # the admin console's own scores (admin_contact_scores) — what the console ranks and filters by
    admin_q = (select(AdminContactScoreDB, ContactDB).join(ContactDB, ContactDB.id == AdminContactScoreDB.contact_id)
               .where(*window(AdminContactScoreDB.scored_at)))
    for s, c in (await session.execute(admin_q.order_by(AdminContactScoreDB.scored_at.desc()).limit(limit))).all():
        at = _iso(s.scored_at)
        if not at:
            continue
        items.append({"id": f"admin-score-{c.id}-{at}", "type": "contact_scored", "title": f"{c.name} scored {s.score}",
                      "subtitle": f"{(level_for(s.score) or '').capitalize()} lead · admin score · {who(c.user_id)}'s contact",
                      "actor": who(s.scored_by) if s.scored_by else "AI scoring", "timestamp": at,
                      "meta": {"contactId": str(c.id), "score": s.score, "scoreKind": "admin"}})
    runs_q = select(ContactPipelineDB)
    if end is not None:
        runs_q = runs_q.where(ContactPipelineDB.started_at < end)
    runs = (await session.execute(runs_q.order_by(ContactPipelineDB.started_at.desc()).limit(limit))).scalars().all()
    run_names = {c.id: c.name for c in (await session.execute(select(ContactDB).where(ContactDB.id.in_({p.contact_id for p in runs})))).scalars().all()} if runs else {}
    for p in runs:
        exists = p.contact_id in run_names
        nm = run_names.get(p.contact_id, "a deleted contact")
        verb = {"complete": "finished for", "complete_no_deck": "finished for", "failed": "failed for"}.get(p.status, "running for")
        sub = "Research, score and pitch ready" + ("" if p.status == "complete" else " (no deck)") if p.status.startswith("complete") \
            else (p.error_message or p.current_step or "")
        items.append({"id": f"pipeline-{p.id}", "type": "pipeline", "title": f"Pipeline {verb} {nm}",
                      "subtitle": sub, "actor": who(p.user_id), "timestamp": _iso(p.completed_at or p.started_at),
                      # only link to a contact that still exists
                      "meta": {"contactId": str(p.contact_id) if exists else None, "status": p.status}})
    for b in (await session.execute(select(AdminBroadcastDB).where(*window(AdminBroadcastDB.created_at))
                                    .order_by(AdminBroadcastDB.created_at.desc()).limit(limit))).scalars().all():
        items.append({"id": f"broadcast-{b.id}", "type": "broadcast", "title": f"{who(b.admin_id)} sent an announcement",
                      "subtitle": b.message, "actor": who(b.admin_id), "timestamp": _iso(b.created_at),
                      "meta": {"priority": b.priority, "active": b.is_active}})
    for f in (await session.execute(select(EventFileDB.id, EventFileDB.original_filename, EventFileDB.uploaded_by, EventFileDB.created_at)
                                    .where(EventFileDB.is_active == True, *window(EventFileDB.created_at))  # noqa: E712
                                    .order_by(EventFileDB.created_at.desc()).limit(limit))).all():
        items.append({"id": f"file-{f.id}", "type": "file", "title": f"{who(f.uploaded_by)} uploaded {f.original_filename}",
                      "subtitle": "Team file", "actor": who(f.uploaded_by), "timestamp": _iso(f.created_at), "meta": {"fileId": str(f.id)}})
    now = datetime.now(timezone.utc)
    for d in (await session.execute(select(DeletedContactDB).where(*window(DeletedContactDB.deleted_at))
                                    .order_by(DeletedContactDB.deleted_at.desc()).limit(limit))).scalars().all():
        restorable = _aware(d.purge_after) >= now if d.purge_after else False
        items.append({"id": f"deleted-{d.id}", "type": "contact_merged" if d.reason else "contact_deleted",
                      "title": f"{d.name} {'merged' if d.reason else 'deleted'}",
                      "subtitle": d.reason or f"{who(d.user_id)}'s contact · restorable for {RECENTLY_DELETED_DAYS} days",
                      "actor": who(d.user_id), "timestamp": _iso(d.deleted_at),
                      # still in Recently deleted → the console can open it there
                      "meta": {"deletedId": str(d.id) if restorable else None}})
    for u in (await session.execute(select(UserDB).where(*window(UserDB.created_at))
                                    .order_by(UserDB.created_at.desc()).limit(limit))).scalars().all():
        items.append({"id": f"user-{u.id}", "type": "user_joined", "title": f"{u.name} joined", "subtitle": "Admin" if u.is_admin else "Rep",
                      "actor": u.name, "timestamp": _iso(u.created_at), "meta": {"userId": str(u.id), "email": u.email}})

    out = []
    for i in items:
        t = _when(i["timestamp"])
        if t is None or (start is not None and t < start) or (end is not None and t >= end):
            continue
        out.append(i)
    out.sort(key=lambda i: _when(i["timestamp"]), reverse=True)
    return out[:limit]


@api_router.get("/admin/activity", dependencies=[Depends(require_admin)])
async def get_admin_activity(
    limit: int = Query(40, ge=1, le=5000),  # exports ask for a whole period
    start: Optional[str] = Query(None, description="ISO 8601; only activity at or after this instant"),
    end: Optional[str] = Query(None, description="ISO 8601; only activity before this instant"),
    session: AsyncSession = Depends(get_db),
):
    """Real workspace timeline, newest first: contacts added, AI scores, pipelines, announcements,
    files, deletions / merges and new accounts — each with the real person's name. Pass start / end
    (e.g. a day in the admin's own time zone) to see what happened then."""
    s, e = _parse_when(start, "start"), _parse_when(end, "end")
    if s and e and e <= s:
        raise HTTPException(status_code=400, detail="end must be after start")
    return await _collect_activity(session, limit, s, e)


@api_router.get("/admin/activity/calendar", dependencies=[Depends(require_admin)])
async def get_admin_activity_calendar(
    start: str = Query(..., description="ISO 8601 instant: the first day shown, at the admin's local midnight"),
    end: str = Query(..., description="ISO 8601 instant: the day after the last one shown"),
    tzOffset: int = Query(0, ge=-840, le=840, description="the browser's Date.getTimezoneOffset() in minutes"),
    session: AsyncSession = Depends(get_db),
):
    """How much happened on each day of a range (up to ~6 weeks), bucketed by the admin's local day:
    {"days": {"2026-10-08": {"total": 12, "types": {"contact_added": 9, ...}}}}."""
    s, e = _parse_when(start, "start"), _parse_when(end, "end")
    if e <= s or (e - s) > timedelta(days=45):
        raise HTTPException(status_code=400, detail="Pick a range of up to 45 days")
    days: dict = {}
    for i in await _collect_activity(session, 5000, s, e):
        local = _when(i["timestamp"]) - timedelta(minutes=tzOffset)
        d = days.setdefault(local.date().isoformat(), {"total": 0, "types": {}})
        d["total"] += 1
        d["types"][i["type"]] = d["types"].get(i["type"], 0) + 1
    return {"days": days}


@api_router.get("/admin/contacts/deleted", dependencies=[Depends(require_admin)])
async def list_admin_deleted_contacts(session: AsyncSession = Depends(get_db)):
    """Everyone's Recently deleted (deleted or merged away, restorable for 30 days), newest first.
    Restore with POST /api/contacts/deleted/{id}/restore?userId=<ownerId>."""
    await purge_expired_deleted(session)
    names = {u.id: u.name for u in (await session.execute(select(UserDB))).scalars().all()}
    rows = (await session.execute(select(DeletedContactDB).order_by(DeletedContactDB.deleted_at.desc()))).scalars().all()
    return {
        "contacts": [{**_format_deleted(d), "ownerId": str(d.user_id), "owner": names.get(d.user_id) or "Unknown"} for d in rows],
        "retentionDays": RECENTLY_DELETED_DAYS,
    }


@api_router.get("/admin/contacts/deleted/{contact_id}", dependencies=[Depends(require_admin)])
async def get_admin_deleted_contact(contact_id: str, session: AsyncSession = Depends(get_db)):
    """Read-only full view of one Recently deleted contact, rebuilt from its snapshot (nothing is
    written: the contact stays deleted until it's restored). Attached files are listed without data."""
    try:
        cid = uuid.UUID(contact_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid contact ID format")
    d = (await session.execute(select(DeletedContactDB).where(DeletedContactDB.id == cid))).scalar_one_or_none()
    if d is None or _aware(d.purge_after) < datetime.now(timezone.utc):
        raise HTTPException(status_code=404, detail="That contact isn't in Recently deleted (it may have expired)")
    contact = _from_snapshot(ContactDB, d.snapshot or {})
    contact.user_id = d.user_id
    owner = (await session.execute(select(UserDB.name).where(UserDB.id == d.user_id))).scalar_one_or_none()
    return {
        "contact": await console_contact(session, None, contact),
        **_format_deleted(d),
        "ownerId": str(d.user_id),
        "owner": owner or "Unknown",
        "files": [{
            "id": str(f.get("id") or ""),
            "filename": f.get("original_filename") or f.get("filename") or "file",
            "fileType": f.get("file_type") or "",
            "category": f.get("category") or "other",
            "fileSize": f.get("file_size") or 0,
            "description": f.get("description") or "",
            "createdAt": f.get("created_at"),
        } for f in (d.files or [])],
    }


# ==================== FIELD REPS & USER MANAGEMENT ====================

class CreateUserPayload(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    email: str = Field(..., max_length=255)
    # the person signs in with this (they can change it with "Forgot password")
    password: str = Field(..., min_length=8, max_length=128)
    isAdmin: Optional[bool] = False
    role: Optional[str] = None  # "admin" | "rep" (older clients)


class UpdateUserPayload(BaseModel):
    name: Optional[str] = Field(None, max_length=255)
    isAdmin: Optional[bool] = None
    # a new password set by the admin (the person is signed out everywhere and signs in with it)
    password: Optional[str] = Field(None, min_length=8, max_length=128)


def _user_photo(profile: Optional[dict]) -> Optional[str]:
    """The picture the person set on their own profile in the app (a data: or http URL), if any."""
    v = event_identity.read_profile(profile).get("avatar")
    if not isinstance(v, str) or not v.strip():
        return None
    v = v.strip()
    return v if v.startswith(("data:", "http://", "https://")) else f"data:image/jpeg;base64,{v}"


def _user_out(u: "UserDB", contacts: int = 0, today: int = 0, last: Optional[datetime] = None, profile: Optional[dict] = None,
              hot: int = 0) -> Dict[str, Any]:
    listed, current = _profile_events(profile)
    return {
        "hotCount": hot,
        # from their app profile: the event they're working now, and every event they listed
        "currentEvent": current or None,
        "events": listed,
        "company": (event_identity.read_profile(profile).get("company_name") or None),
        "id": str(u.id), "name": u.name, "email": u.email,
        "role": "admin" if u.is_admin else "rep", "isAdmin": bool(u.is_admin),
        "contactsCount": contacts, "scansCount": contacts, "scansToday": today,
        # last time they added a contact (None = never)
        "lastActive": _iso(last),
        "linkedin": bool(u.linkedin_sub),
        "createdAt": _iso(u.created_at),
        # from their own profile in the app (None when not set)
        "photoUrl": _user_photo(profile),
        "title": (event_identity.read_profile(profile).get("job_title") or None),
    }


@api_router.get("/admin/users", dependencies=[Depends(require_admin)])
async def list_admin_users(session: AsyncSession = Depends(get_db)):
    """Every account with real numbers: contacts, contacts added today, last activity."""
    day = datetime.now(timezone.utc) - timedelta(hours=24)
    stats = {uid: (n, last) for uid, n, last in (await session.execute(
        select(ContactDB.user_id, func.count(ContactDB.id), func.max(ContactDB.created_at)).group_by(ContactDB.user_id))).all()}
    today = dict((await session.execute(
        select(ContactDB.user_id, func.count(ContactDB.id)).where(ContactDB.created_at >= day).group_by(ContactDB.user_id))).all())
    hot = dict((await session.execute(_with_admin_score(
        select(ContactDB.user_id, func.count(ContactDB.id))).where(level_condition(ADMIN_SCORE, "hot")).group_by(ContactDB.user_id))).all())
    users = (await session.execute(select(UserDB).order_by(UserDB.created_at.desc()))).scalars().all()
    profiles = dict((await session.execute(select(UserProfileDB.user_id, UserProfileDB.profile_data))).all())
    return [_user_out(u, stats.get(u.id, (0, None))[0], today.get(u.id, 0), stats.get(u.id, (0, None))[1], profiles.get(u.id), hot.get(u.id, 0))
            for u in users]


@api_router.post("/admin/users", dependencies=[Depends(require_admin)])
async def create_admin_user(payload: CreateUserPayload, session: AsyncSession = Depends(get_db)):
    """Create an account (rep or admin) with a real password the person signs in with."""
    hasher = _need(hash_password_impl, "Creating accounts")
    email = payload.email.strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    if (await session.execute(select(UserDB.id).where(func.lower(UserDB.email) == email))).scalar_one_or_none():
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    is_admin = bool(payload.isAdmin or (payload.role or "").lower() == "admin")
    user = UserDB(id=uuid.uuid4(), name=" ".join(payload.name.split()), email=email,
                  password_hash=await asyncio.to_thread(hasher, payload.password), is_admin=is_admin, password_set=True,
                  created_at=datetime.now(timezone.utc))
    session.add(user)
    await session.commit()
    await broadcast_live_event("user_created", {"userId": str(user.id), "name": user.name})
    return _user_out(user)


@api_router.patch("/admin/users/{user_id}")
async def update_admin_user(user_id: str, payload: UpdateUserPayload, session: AsyncSession = Depends(get_db),
                            admin_id: uuid.UUID = Depends(require_admin)):
    """Rename an account or change its role. You can't remove your own admin rights."""
    try:
        uid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID")
    user = (await session.execute(select(UserDB).where(UserDB.id == uid))).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if payload.name is not None:
        name = " ".join(payload.name.split())
        if not name:
            raise HTTPException(status_code=400, detail="The name can't be empty.")
        user.name = name
        # the app shows the profile's full name first: keep the two the same
        prof = (await session.execute(select(UserProfileDB).where(UserProfileDB.user_id == uid))).scalar_one_or_none()
        prof_data = event_identity.profile_dict(prof) if prof else None
        if prof_data is not None and prof_data.get("full_name") and prof_data["full_name"] != name:
            prof_data["full_name"] = name
            event_identity.write_profile(session, prof, uid, prof_data)
    if payload.isAdmin is not None and payload.isAdmin != user.is_admin:
        if uid == admin_id and not payload.isAdmin:
            raise HTTPException(status_code=400, detail="You can't remove your own admin rights.")
        user.is_admin = payload.isAdmin
        user.token_version = (user.token_version or 0) + 1  # their current sign-in reflects the old role → sign in again
    if payload.password is not None:
        hasher = _need(hash_password_impl, "Setting passwords")
        user.password_hash = await asyncio.to_thread(hasher, payload.password)
        user.password_set = True
        user.token_version = (user.token_version or 0) + 1  # signed out everywhere; they sign in with the new one
    await session.commit()
    # other open consoles refresh their Team page (name / role changed)
    await broadcast_live_event("user_updated", {"userId": str(uid), "name": user.name})
    # answer with the same real numbers the list shows
    n, last = (await session.execute(select(func.count(ContactDB.id), func.max(ContactDB.created_at)).where(ContactDB.user_id == uid))).one()
    day = datetime.now(timezone.utc) - timedelta(hours=24)
    today = (await session.execute(select(func.count(ContactDB.id)).where(ContactDB.user_id == uid, ContactDB.created_at >= day))).scalar() or 0
    hot = (await session.execute(_with_admin_score(select(func.count(ContactDB.id))).where(ContactDB.user_id == uid, level_condition(ADMIN_SCORE, "hot")))).scalar() or 0
    profile = (await session.execute(select(UserProfileDB.profile_data).where(UserProfileDB.user_id == uid))).scalar_one_or_none()
    return _user_out(user, n or 0, today, last, profile, hot)


# ==================== MY ACCOUNT (the signed-in admin's own profile & password) ====================
# The same profile the app's Me screen edits (UserProfileDB.profile_data: full_name, job_title, company_name,
# company_website, linkedin_url, phone, avatar, email_signature), plus the account itself (UserDB).
verify_password_impl = None    # (plain, hash) -> bool
normalize_profile_impl = None  # (update_data) -> None   (the app's profile validation: full_name etc.)

MY_PROFILE_FIELDS = ("full_name", "job_title", "company_name", "company_website", "linkedin_url", "phone", "email_signature")


class MyProfilePayload(BaseModel):
    full_name: Optional[str] = Field(None, max_length=120)
    job_title: Optional[str] = Field(None, max_length=120)
    company_name: Optional[str] = Field(None, max_length=160)
    company_website: Optional[str] = Field(None, max_length=300)
    linkedin_url: Optional[str] = Field(None, max_length=300)
    phone: Optional[str] = Field(None, max_length=40)
    email_signature: Optional[str] = Field(None, max_length=2000)
    # a data URI / base64 image to set, "" to remove it (shrunk to a ≤512px JPEG, like the app)
    avatar: Optional[str] = None


class MyPasswordPayload(BaseModel):
    currentPassword: Optional[str] = Field(None, max_length=128)
    newPassword: str = Field(..., min_length=8, max_length=128)


def _me_out(user: "UserDB", profile: Optional[dict]) -> Dict[str, Any]:
    p = event_identity.read_profile(profile)
    return {
        "id": str(user.id), "name": user.name, "email": user.email, "isAdmin": bool(user.is_admin),
        "createdAt": _iso(user.created_at),
        # LinkedIn-created accounts have no password of their own until they set one
        "passwordSet": bool(user.password_set),
        "linkedin": {"connected": bool(user.linkedin_sub), "email": user.linkedin_email or None},
        "profile": {**{k: (p.get(k) or "") for k in MY_PROFILE_FIELDS}, "full_name": p.get("full_name") or user.name or ""},
        "photoUrl": _user_photo(p),
    }


async def _load_me(session: AsyncSession, admin_id: uuid.UUID):
    user = (await session.execute(select(UserDB).where(UserDB.id == admin_id))).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="Your account wasn't found. Sign in again.")
    row = (await session.execute(select(UserProfileDB).where(UserProfileDB.user_id == admin_id))).scalar_one_or_none()
    return user, row


@api_router.get("/admin/me")
async def get_my_account(session: AsyncSession = Depends(get_db), admin_id: uuid.UUID = Depends(require_admin)):
    """The signed-in admin's own account and profile."""
    user, row = await _load_me(session, admin_id)
    return _me_out(user, row)


@api_router.patch("/admin/me")
async def update_my_account(payload: MyProfilePayload, session: AsyncSession = Depends(get_db),
                            admin_id: uuid.UUID = Depends(require_admin)):
    """Edit your own profile (the same fields as the app's Me screen). Your name is also the account
    name the team sees (people list, announcements)."""
    user, row = await _load_me(session, admin_id)
    data = payload.model_dump(exclude_unset=True)
    if not data:
        raise HTTPException(status_code=400, detail="Nothing to save")
    for k in MY_PROFILE_FIELDS:
        if data.get(k) is not None:
            data[k] = data[k].strip() if k == "email_signature" else " ".join(data[k].split())
    if normalize_profile_impl:
        normalize_profile_impl(data)  # e.g. "Your name can't be empty"
    elif data.get("full_name") == "":
        raise HTTPException(status_code=400, detail="Your name can't be empty")
    for k in ("company_website", "linkedin_url"):
        v = data.get(k)
        if v and not re.match(r"^(https?://)?[\w.-]+\.[a-z]{2,}(/\S*)?$", v, re.I):
            raise HTTPException(status_code=400, detail=f"Enter a valid {'website' if k == 'company_website' else 'LinkedIn'} address.")
    remove_avatar = False
    if "avatar" in data:
        raw = data.pop("avatar")
        if raw:
            data["avatar"] = "data:image/jpeg;base64," + process_contact_photo(raw)
        else:
            remove_avatar = True
    merged = event_identity.read_profile(row)
    for k, v in data.items():
        if v is not None:
            merged[k] = v
    if remove_avatar:
        merged.pop("avatar", None)
    row = event_identity.write_profile(session, row, admin_id, merged)
    row.updated_at = datetime.now(timezone.utc)
    if data.get("full_name"):
        user.name = data["full_name"]
    await session.commit()
    await broadcast_live_event("user_updated", {"userId": str(admin_id), "name": user.name})
    return _me_out(user, merged)


@api_router.post("/admin/me/password")
async def change_my_password(payload: MyPasswordPayload, session: AsyncSession = Depends(get_db),
                             admin_id: uuid.UUID = Depends(require_admin)):
    """Change your password (your current one is required). Every other device is signed out; this
    console gets fresh tokens so it stays signed in. A LinkedIn-created account sets its first one."""
    hasher = _need(hash_password_impl, "Changing passwords")
    verify = _need(verify_password_impl, "Changing passwords")
    user, _ = await _load_me(session, admin_id)
    if user.password_set:
        if not payload.currentPassword:
            raise HTTPException(status_code=400, detail="Enter your current password.")
        if not await asyncio.to_thread(verify, payload.currentPassword, user.password_hash):
            raise HTTPException(status_code=400, detail="Your current password is incorrect.")
        if payload.currentPassword == payload.newPassword:
            raise HTTPException(status_code=400, detail="Choose a password different from your current one.")
    user.password_hash = await asyncio.to_thread(hasher, payload.newPassword)
    user.password_set = True
    user.token_version = int(user.token_version or 0) + 1  # tokens issued before this stop working
    await session.commit()
    tokens = identity.issue_auth_tokens(str(user.id), user.email, bool(user.is_admin), user.token_version)
    return {"status": "success", "message": "Password changed. You've been signed out on every other device.",
            "token": tokens["token"], "refreshToken": tokens["refresh_token"]}


# ==================== ADMIN SALES PROFILE (the admin console's own AI profile) ====================
# Its own table (admin_sales_profile), never a rep's user_profiles row. Everything admin-side AI does —
# scoring and ranking all contacts, research, pitches, the assistant — runs on THIS profile only; the
# app and the reps' own scores never read it, and it never reads theirs.
ADMIN_SALES_ID = scoring.ADMIN_SALES_ID


class AdminSalesPayload(BaseModel):
    # who is pitching
    company_name: Optional[str] = Field(None, max_length=160)
    job_title: Optional[str] = Field(None, max_length=120)
    # what we sell
    products: Optional[List[Dict[str, str]]] = Field(None, max_length=50)
    # how we sell
    pitch_style: Optional[str] = Field(None, max_length=40)
    value_propositions: Optional[List[str]] = Field(None, max_length=50)
    common_objections: Optional[List[Dict[str, str]]] = Field(None, max_length=50)
    case_studies: Optional[List[Dict[str, str]]] = Field(None, max_length=50)
    # who we target
    target_industries: Optional[List[str]] = Field(None, max_length=50)
    target_roles: Optional[List[str]] = Field(None, max_length=50)
    target_company_sizes: Optional[List[str]] = Field(None, max_length=20)
    target_geographies: Optional[List[str]] = Field(None, max_length=50)


async def _admin_sales_out(session: AsyncSession, row: Optional["AdminSalesProfileDB"]) -> Dict[str, Any]:
    d = (row.data if row else None) or {}
    pairs = lambda key, a, b: [{a: x.get(a) or "", b: x.get(b) or ""} for x in d.get(key) or [] if isinstance(x, dict)]  # noqa: E731
    strs = lambda key: [str(v) for v in d.get(key) or []]  # noqa: E731
    by = None
    if row and row.updated_by:
        by = (await session.execute(select(UserDB.name).where(UserDB.id == row.updated_by))).scalar_one_or_none()
    return {
        "company_name": d.get("company_name") or "",
        "job_title": d.get("job_title") or "",
        "products": [{"name": p.get("name") or "", "description": p.get("description") or "", "ideal_customer": p.get("ideal_customer") or ""}
                     for p in d.get("products") or [] if isinstance(p, dict)],
        "pitch_style": d.get("pitch_style") or "",
        "value_propositions": strs("value_propositions"),
        "common_objections": pairs("common_objections", "objection", "response"),
        "case_studies": pairs("case_studies", "title", "result"),
        "target_industries": strs("target_industries"),
        "target_roles": strs("target_roles"),
        "target_company_sizes": strs("target_company_sizes"),
        "target_geographies": strs("target_geographies"),
        "updatedAt": _iso(row.updated_at) if row else None,
        "updatedBy": by,
    }


@api_router.get("/admin/sales-profile")
async def get_admin_sales_profile(session: AsyncSession = Depends(get_db), admin_id: uuid.UUID = Depends(require_admin)):
    """The admin console's own AI profile (shared by every admin): who's pitching, what we sell,
    how we sell, who we target."""
    row = (await session.execute(select(AdminSalesProfileDB).where(AdminSalesProfileDB.id == ADMIN_SALES_ID))).scalar_one_or_none()
    return await _admin_sales_out(session, row)


@api_router.patch("/admin/sales-profile")
async def update_admin_sales_profile(payload: AdminSalesPayload, session: AsyncSession = Depends(get_db),
                                     admin_id: uuid.UUID = Depends(require_admin)):
    """Save part of the console's AI profile (only the fields sent change). Same rules as the app's Me
    screen (12 products max, a known pitch style…). No rep's profile, score or data is touched."""
    data = payload.model_dump(exclude_unset=True)
    if not data:
        raise HTTPException(status_code=400, detail="Nothing to save")
    for k in ("company_name", "job_title"):
        if data.get(k) is not None:
            data[k] = " ".join(data[k].split())
    for k, cap in (("target_industries", 20), ("target_roles", 20)):  # the app's normalizer doesn't cover these
        if data.get(k) is not None:
            seen, out = set(), []
            for v in data[k]:
                v = " ".join(str(v or "").split())[:80]
                if v and v.lower() not in seen:
                    seen.add(v.lower())
                    out.append(v)
            data[k] = out[:cap]
    _need(normalize_profile_impl, "Saving the sales profile")(data)  # products, pitch style, lists: trims, caps, validates
    row = (await session.execute(select(AdminSalesProfileDB).where(AdminSalesProfileDB.id == ADMIN_SALES_ID))).scalar_one_or_none()
    merged = {**((row.data if row else None) or {}), **{k: v for k, v in data.items() if v is not None}}
    if row is None:
        row = AdminSalesProfileDB(id=ADMIN_SALES_ID)
        session.add(row)
    row.data = merged  # a new dict, so the JSON column is saved
    row.updated_by = admin_id
    row.updated_at = datetime.now(timezone.utc)
    await session.commit()
    return await _admin_sales_out(session, row)


@api_router.delete("/admin/users/{user_id}")
async def delete_admin_user(user_id: str, session: AsyncSession = Depends(get_db), admin_id: uuid.UUID = Depends(require_admin)):
    """Delete a rep's account and all their data for good (contacts, archive, profile, card…).
    Admin accounts must be made reps first; you can't delete yourself."""
    try:
        uid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID")
    if uid == admin_id:
        raise HTTPException(status_code=400, detail="You can't delete your own account here.")
    user = (await session.execute(select(UserDB).where(UserDB.id == uid))).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.is_admin:
        raise HTTPException(status_code=400, detail="Make this person a rep first, then delete the account.")
    name = user.name
    await purge_user_contact_data(session, uid)
    await _detach_user_refs(session, uid, admin_id)
    for model in (ConversationDB, UserProfileDB, UserCardDB, BoothVisitDB, ContactShareLinkDB, EventNoteDB):
        await session.execute(delete(model).where(model.user_id == uid))
    try:
        from database import PasswordResetDB
        await session.execute(delete(PasswordResetDB).where(PasswordResetDB.user_id == uid))
    except ImportError:
        pass
    await session.delete(user)
    await session.commit()
    if delete_user_index_impl:
        try:
            delete_user_index_impl(str(uid))
        except Exception as e:  # the search index is rebuilt at startup anyway
            print(f"[ADMIN] search index cleanup for {uid} failed: {e}")
    await broadcast_live_event("user_deleted", {"userId": user_id, "name": name})
    return {"status": "deleted", "id": user_id, "message": f"{name}'s account and data were deleted."}


class MergeUsersPayload(BaseModel):
    sourceUserId: str
    targetUserId: str


def _keep_updated_at(model) -> dict:
    """For a bulk ownership move: leave updated_at as it is. The column's onupdate would otherwise
    stamp every moved row "updated just now" although nobody edited it."""
    return {"updated_at": model.updated_at} if hasattr(model, "updated_at") else {}


async def _move_user_refs(session: AsyncSession, src: uuid.UUID, tgt: uuid.UUID) -> None:
    """Account merge: everything that belongs to `src` moves to `tgt`. Rows that are one-per-person
    (a booth visit, an event assignment) move unless `tgt` already has its own, which is kept."""
    await session.execute(update(ContactDB).where(ContactDB.user_id == src).values(user_id=tgt, **_keep_updated_at(ContactDB)))
    await reassign_user_archive(session, src, tgt)  # Recently deleted + duplicate decisions
    for model, col in ((ConversationDB, "user_id"), (ContactShareLinkDB, "user_id"), (ContactPipelineDB, "user_id"), (EventNoteDB, "user_id"),
                       (SharedContactDB, "user_id"), (ContactFileDB, "uploaded_by"), (EventFileDB, "uploaded_by"),
                       (AdminBroadcastDB, "admin_id"), (ContactListDB, "admin_id"), (ManagedEventDB, "created_by"),
                       (ExhibitorDB, "added_by"), (AdminContactScoreDB, "owner_id"), (AdminContactScoreDB, "scored_by"),
                       (EventAssignmentDB, "assigned_by"), (AdminSalesProfileDB, "updated_by")):
        c = getattr(model, col)
        await session.execute(update(model).where(c == src).values({col: tgt, **_keep_updated_at(model)}))
    # booth visits: one per person + event + booth
    have = {(v.event_key, v.exhibitor_key) for v in (await session.execute(select(BoothVisitDB).where(BoothVisitDB.user_id == tgt))).scalars().all()}
    for v in (await session.execute(select(BoothVisitDB).where(BoothVisitDB.user_id == src))).scalars().all():
        if (v.event_key, v.exhibitor_key) in have:
            await session.delete(v)
        else:
            v.user_id = tgt
    # event team: one assignment per person per event (a "lead" role survives the merge)
    mine = {a.event_id: a for a in (await session.execute(select(EventAssignmentDB).where(EventAssignmentDB.user_id == tgt))).scalars().all()}
    for a in (await session.execute(select(EventAssignmentDB).where(EventAssignmentDB.user_id == src))).scalars().all():
        if a.event_id in mine:
            if a.role == "lead":
                mine[a.event_id].role = "lead"
            await session.delete(a)
        else:
            a.user_id = tgt
    # targeted announcements addressed to the merged-away account reach the kept one
    for b in (await session.execute(select(AdminBroadcastDB).where(AdminBroadcastDB.recipient_ids.isnot(None)))).scalars().all():
        ids = [str(x) for x in (b.recipient_ids or [])]
        if str(src) in ids:
            b.recipient_ids = list(dict.fromkeys(str(tgt) if x == str(src) else x for x in ids))
    # profile and business card: the kept account's win; the other's are used only if it has none
    for model in (UserProfileDB, UserCardDB):
        if (await session.execute(select(model.id).where(model.user_id == tgt))).scalar_one_or_none():
            await session.execute(delete(model).where(model.user_id == src))
        else:
            await session.execute(update(model).where(model.user_id == src).values(user_id=tgt, **_keep_updated_at(model)))
    try:
        from database import PasswordResetDB
        await session.execute(delete(PasswordResetDB).where(PasswordResetDB.user_id == src))
    except ImportError:
        pass


async def _detach_user_refs(session: AsyncSession, uid: uuid.UUID, admin_id: uuid.UUID) -> None:
    """Account deletion: rows elsewhere that point at the person and must outlive them. Things that
    need an owner (files they uploaded to other contacts, lists, announcements, pipeline runs on
    others' contacts) go to the admin deleting the account; plain markers are cleared."""
    for model, col in ((ContactFileDB, "uploaded_by"), (EventFileDB, "uploaded_by"), (AdminBroadcastDB, "admin_id"),
                       (ContactListDB, "admin_id"), (ContactPipelineDB, "user_id")):
        c = getattr(model, col)
        await session.execute(update(model).where(c == uid).values({col: admin_id, **_keep_updated_at(model)}))
    for model, col in ((ExhibitorDB, "added_by"), (ManagedEventDB, "created_by"), (EventAssignmentDB, "assigned_by"),
                       (AdminContactScoreDB, "owner_id"), (AdminContactScoreDB, "scored_by"), (AdminSalesProfileDB, "updated_by")):
        c = getattr(model, col)
        await session.execute(update(model).where(c == uid).values({col: None, **_keep_updated_at(model)}))
    await session.execute(delete(EventAssignmentDB).where(EventAssignmentDB.user_id == uid))
    await session.execute(delete(SharedContactDB).where(SharedContactDB.user_id == uid))


@api_router.post("/admin/users/merge")
async def merge_users(payload: MergeUsersPayload, session: AsyncSession = Depends(get_db), admin_id: uuid.UUID = Depends(require_admin)):
    """Merge two accounts of the same person (e.g. LinkedIn and email): everything the first one has —
    contacts, chats, Recently deleted, booth visits, share links, event team spots, files, lists,
    announcements — moves to the second, its LinkedIn sign-in too (if the second has none), then the
    first account is removed."""
    try:
        src_id = uuid.UUID(payload.sourceUserId)
        tgt_id = uuid.UUID(payload.targetUserId)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    if src_id == tgt_id:
        raise HTTPException(status_code=400, detail="Pick two different accounts.")
    if src_id == admin_id:
        raise HTTPException(status_code=400, detail="You can't merge away the account you're signed in with.")
    src = (await session.execute(select(UserDB).where(UserDB.id == src_id))).scalar_one_or_none()
    tgt = (await session.execute(select(UserDB).where(UserDB.id == tgt_id))).scalar_one_or_none()
    if not src:
        raise HTTPException(status_code=404, detail="The account to merge away no longer exists.")
    if not tgt:
        raise HTTPException(status_code=404, detail="The account to keep no longer exists.")
    moved = (await session.execute(select(func.count(ContactDB.id)).where(ContactDB.user_id == src_id))).scalar() or 0
    src_name, tgt_name = src.name, tgt.name

    await _move_user_refs(session, src_id, tgt_id)
    # keep their LinkedIn sign-in working: the link moves when the kept account has none
    linkedin_moved = False
    if src.linkedin_sub and not tgt.linkedin_sub:
        sub, li_email, li_at = src.linkedin_sub, src.linkedin_email, src.linkedin_linked_at
        src.linkedin_sub = None  # unique column: free it before the kept account takes it
        await session.flush()
        tgt.linkedin_sub, tgt.linkedin_email, tgt.linkedin_linked_at = sub, li_email, li_at
        linkedin_moved = True
    await session.delete(src)
    await session.commit()

    # search index: the moved contacts are now the kept account's
    if delete_user_index_impl:
        try:
            delete_user_index_impl(str(src_id))
        except Exception as e:
            print(f"[ADMIN] search index cleanup for {src_id} failed: {e}")
    for c in (await session.execute(select(ContactDB).where(ContactDB.user_id == tgt_id))).scalars().all():
        await sync_search_index(c)
    await broadcast_live_event("user_deleted", {"userId": str(src_id), "name": src_name})
    return {"status": "success", "movedContacts": moved, "linkedinMoved": linkedin_moved,
            "message": f"{src_name}'s {moved} contact{'' if moved == 1 else 's'} and data moved to {tgt_name}; {src_name}'s account was removed."
                       + (" Their LinkedIn sign-in now opens this account." if linkedin_moved else "")}


# ==================== BROADCASTS & FIELD NOTIFICATIONS ====================

class BroadcastPayload(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    priority: Optional[str] = Field("normal", pattern="^(normal|urgent)$")
    recipientIds: Optional[List[str]] = Field(None, max_length=500)
    eventId: Optional[str] = None  # the planned event it's about (shown on the announcement and the event)


@api_router.get("/broadcasts/active")
async def get_active_broadcasts(session: AsyncSession = Depends(get_db), me: Optional[uuid.UUID] = Depends(request_user_id)):
    """Active team announcements. Reps get the ones sent to everyone or to them; the admin
    console (me = None) sees all."""
    res = await session.execute(
        select(AdminBroadcastDB).where(AdminBroadcastDB.is_active == True).order_by(AdminBroadcastDB.created_at.desc())
    )
    broadcasts = [b for b in res.scalars().all()
                  if me is None or not b.recipient_ids or str(me) in {str(x) for x in (b.recipient_ids or [])}]
    # the admin console also gets who sent it and who it went to (a rep only needs the message)
    names = {str(u.id): u.name for u in (await session.execute(select(UserDB))).scalars().all()} if me is None else {}
    return [
        {
            "id": str(b.id),
            "message": b.message,
            "priority": b.priority,
            "isActive": b.is_active,
            "createdAt": _aware(b.created_at).isoformat() if b.created_at else None,  # unknown, not "just now"
            **({
                "sentBy": names.get(str(b.admin_id)) if b.admin_id else None,
                "recipients": [{"id": str(x), "name": names.get(str(x)) or "Removed account"} for x in (b.recipient_ids or [])],
            } if me is None else {}),
        }
        for b in broadcasts
    ]


@api_router.get("/broadcasts/mine")
async def my_broadcasts(limit: int = Query(50, ge=1, le=200), session: AsyncSession = Depends(get_db),
                        me: Optional[uuid.UUID] = Depends(request_user_id)):
    """Every announcement sent to you (to everyone, or to you), live and ended, newest first —
    the app's announcements dropdown."""
    rows = (await session.execute(select(AdminBroadcastDB).order_by(AdminBroadcastDB.created_at.desc()).limit(limit * 4))).scalars().all()
    mine = [b for b in rows if me is None or not b.recipient_ids or str(me) in {str(x) for x in (b.recipient_ids or [])}][:limit]
    return [
        {
            "id": str(b.id),
            "message": b.message,
            "priority": b.priority,
            "isActive": b.is_active,
            "createdAt": _aware(b.created_at).isoformat() if b.created_at else None,  # unknown, not "just now"
            "endedAt": _aware(b.ended_at).isoformat() if b.ended_at else None,
        }
        for b in mine
    ]


@api_router.get("/broadcasts/stream")
async def my_broadcast_stream(request: Request, me: Optional[uuid.UUID] = Depends(request_user_id)):
    """Live feed for the app: an `announcements` event whenever an announcement is sent or ended,
    so the phone shows it right away. It carries no message (it may be for someone else) — the app
    then fetches /broadcasts/mine, which filters by recipient."""
    async def event_generator():
        queue: asyncio.Queue = new_sse_queue()
        sse_subscribers.append(queue)
        try:
            yield "event: ready\ndata: {}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=12.0)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                if event.get("type") == "broadcast_alert":
                    yield f"event: announcements\ndata: {json.dumps({'id': (event.get('data') or {}).get('id')})}\n\n"
        finally:
            if queue in sse_subscribers:
                sse_subscribers.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@api_router.post("/admin/broadcast")
async def send_broadcast(payload: BroadcastPayload, session: AsyncSession = Depends(get_db), admin_id: uuid.UUID = Depends(require_admin)):
    """Dispatch broadcast announcement alert to field reps on mobile."""
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Write a message first.")
    recipients = None
    if payload.recipientIds:
        # only real accounts (a typo or a removed rep would otherwise silently get nothing)
        try:
            wanted = {uuid.UUID(x) for x in payload.recipientIds}
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid recipient ID")
        found = set((await session.execute(select(UserDB.id).where(UserDB.id.in_(wanted)))).scalars().all())
        if wanted - found:
            raise HTTPException(status_code=400, detail=f"{len(wanted - found)} recipient(s) no longer have an account — refresh and pick again.")
        recipients = [str(x) for x in wanted]
    event_id = None
    if payload.eventId:
        try:
            event_id = uuid.UUID(payload.eventId)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid event ID")
        if not (await session.execute(select(ManagedEventDB.id).where(ManagedEventDB.id == event_id))).scalar_one_or_none():
            raise HTTPException(status_code=400, detail="That event was removed — pick it again")

    bid = uuid.uuid4()
    bc = AdminBroadcastDB(
        id=bid,
        admin_id=admin_id,
        message=message,
        priority=payload.priority or "normal",
        recipient_ids=recipients,
        event_id=event_id,
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )
    session.add(bc)
    await session.commit()

    # Live SSE feed (admin consoles only — reps fetch /broadcasts/active, which filters by recipient)
    await broadcast_live_event("broadcast_alert", {
        "id": str(bid),
        "message": message,
        "priority": payload.priority or "normal",
        "recipients": len(recipients) if recipients else "everyone",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return {
        "id": str(bid),
        "status": "dispatched",
        "message": message,
        "priority": payload.priority or "normal",
    }


@api_router.post("/broadcasts/{broadcast_id}/dismiss", dependencies=[Depends(require_admin)])
async def dismiss_broadcast(broadcast_id: str, session: AsyncSession = Depends(get_db)):
    """Dismiss or resolve an active broadcast."""
    try:
        bid = uuid.UUID(broadcast_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid broadcast ID")

    await session.execute(
        update(AdminBroadcastDB).where(AdminBroadcastDB.id == bid, AdminBroadcastDB.is_active == True)  # noqa: E712
        .values(is_active=False, ended_at=datetime.now(timezone.utc))
    )
    await session.commit()
    await broadcast_live_event("broadcast_alert", {"id": broadcast_id, "ended": True})
    return {"status": "dismissed", "id": broadcast_id}


@api_router.get("/admin/broadcasts/history", dependencies=[Depends(require_admin)])
async def broadcast_history(
    limit: int = Query(300, ge=1, le=2000),
    start: Optional[str] = Query(None, description="ISO 8601; only announcements sent at or after this instant"),
    end: Optional[str] = Query(None, description="ISO 8601; only announcements sent before this instant"),
    session: AsyncSession = Depends(get_db),
):
    """Every announcement ever sent, newest first: live and ended, who sent it, who it went to, the
    event it's about, and when it ended. start / end (e.g. one local day) narrow it to a window."""
    s, e = _parse_when(start, "start"), _parse_when(end, "end")
    if s and e and e <= s:
        raise HTTPException(status_code=400, detail="end must be after start")
    query = select(AdminBroadcastDB)
    if s is not None:
        query = query.where(AdminBroadcastDB.created_at >= s)
    if e is not None:
        query = query.where(AdminBroadcastDB.created_at < e)
    rows = (await session.execute(query.order_by(AdminBroadcastDB.created_at.desc()).limit(limit))).scalars().all()
    names = {str(u.id): u.name for u in (await session.execute(select(UserDB))).scalars().all()}
    ev_ids = {b.event_id for b in rows if b.event_id}
    events = {e.id: e.name for e in (await session.execute(select(ManagedEventDB).where(ManagedEventDB.id.in_(ev_ids)))).scalars().all()} if ev_ids else {}
    return [{
        "id": str(b.id),
        "message": b.message,
        "priority": b.priority,
        "isActive": bool(b.is_active),
        "createdAt": _iso(b.created_at),
        "endedAt": _iso(b.ended_at),
        "sentBy": names.get(str(b.admin_id)) if b.admin_id else None,
        "recipients": [{"id": str(x), "name": names.get(str(x)) or "Removed account"} for x in (b.recipient_ids or [])],
        "eventId": str(b.event_id) if b.event_id and b.event_id in events else None,
        "eventName": events.get(b.event_id) if b.event_id else None,
    } for b in rows]


# ==================== LEAD DEDUPLICATION & MERGE ====================

DUPLICATE_CONFIDENCE = {"email": 98, "phone": 92, "name": 80}
DUPLICATE_REASON_TEXT = {"email": "Same email", "phone": "Same phone number", "name": "Same name and company"}


async def _all_dismissed_pairs(session: AsyncSession) -> set:
    """Every (lo, hi) pair someone marked "not the same person" — whoever it was saved under, so a
    decision about two different reps' contacts holds from either side."""
    return {(d.contact_a, d.contact_b) for d in (await session.execute(select(DuplicateDismissalDB))).scalars().all()}


def _created(c: "ContactDB") -> datetime:
    return _aware(c.created_at) or datetime.min.replace(tzinfo=timezone.utc)


@api_router.get("/admin/contacts/duplicates", dependencies=[Depends(require_admin)])
async def find_duplicate_contacts(session: AsyncSession = Depends(get_db)):
    """Pairs of contacts anywhere in the workspace that look like the same person: a shared email,
    phone number (last 10 digits; any of the numbers in the field) or name + company (titles and
    legal suffixes ignored). Covers one rep's own contacts AND two reps who captured the same person
    (`crossRep`). Pairs someone marked "not a duplicate" are left out."""
    rows = (await session.execute(select(ContactDB))).scalars().all()
    owners = {u.id: u.name for u in (await session.execute(select(UserDB))).scalars().all()}
    dismissed = await _all_dismissed_pairs(session)
    keyed = [(c, _keys_for(c.name, c.company_name, c.email, c.phone)) for c in rows]
    out = []
    for a, b, reasons, strong in _duplicate_pairs(keyed, dismissed):
                pair = _pair(a.id, b.id)
                # the older one is the primary (keeps its id); the newer one gets merged into it
                primary, dup = (a, b) if _created(a) <= _created(b) else (b, a)
                out.append({
                    "id": f"{pair[0]}:{pair[1]}",
                    "confidence": _match_confidence(reasons, strong),
                    "reasons": reasons,
                    "reason": " · ".join(DUPLICATE_REASON_TEXT[r] for r in reasons),
                    # the kept contact's rep; for a cross-rep pair the merged-in one's rep is duplicateOwner
                    "owner": owners.get(primary.user_id, ""),
                    "ownerId": str(primary.user_id),
                    "duplicateOwner": owners.get(dup.user_id, ""),
                    "duplicateOwnerId": str(dup.user_id),
                    "crossRep": primary.user_id != dup.user_id,
                    "primary": format_contact(primary),
                    "duplicate": format_contact(dup),
                })
    views = await console_scores(session, None, [d["primary"] for d in out] + [d["duplicate"] for d in out])
    for i, d in enumerate(out):
        d["primary"], d["duplicate"] = views[i], views[len(out) + i]
    out.sort(key=lambda d: (-d["confidence"], d["crossRep"]))
    return out


@api_router.get("/admin/contacts/{contact_id}/duplicates", dependencies=[Depends(require_admin)])
async def admin_contact_duplicates(contact_id: str, session: AsyncSession = Depends(get_db)):
    """Every other contact in the workspace that looks like this one — the owner's own and other
    reps' (`sameRep` false) — minus pairs marked "not a duplicate". The admin console's contact panel."""
    try:
        cid = uuid.UUID(contact_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid contact ID format")
    contact = (await session.execute(select(ContactDB).where(ContactDB.id == cid))).scalar_one_or_none()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")
    keys = _keys_for(contact.name, contact.company_name, contact.email, contact.phone)
    if not (keys["email"] or keys["phone"] or keys["name"] or keys["person"]):
        return {"matches": []}
    owners = {u.id: u.name for u in (await session.execute(select(UserDB))).scalars().all()}
    dismissed = await _all_dismissed_pairs(session)
    others = [(c, _keys_for(c.name, c.company_name, c.email, c.phone))
              for c in (await session.execute(select(ContactDB).where(ContactDB.id != cid))).scalars().all()]
    # shared lines are judged across the whole workspace — the same population the Duplicates page uses
    shared = _shared_lines([keys] + [k for _, k in others])
    matches = []
    for c, k in others:
        reasons, strong = _match(keys, k, shared)
        if reasons and _pair(cid, c.id) not in dismissed:
            matches.append({"contact": format_contact(c), "reasons": reasons, "confidence": _match_confidence(reasons, strong),
                            "ownerName": owners.get(c.user_id, ""), "sameRep": c.user_id == contact.user_id})
    for m, v in zip(matches, await console_scores(session, None, [m["contact"] for m in matches])):
        m["contact"] = v
    # same rep first, then the strongest evidence
    matches.sort(key=lambda m: (not m["sameRep"], -m["confidence"]))
    return {"matches": matches}


class DismissDuplicatePayload(BaseModel):
    primaryId: str
    duplicateId: str


@api_router.post("/admin/contacts/duplicates/dismiss", dependencies=[Depends(require_admin)])
async def dismiss_duplicate_pair(payload: DismissDuplicatePayload, session: AsyncSession = Depends(get_db)):
    """Mark two contacts as NOT the same person (for their owner too), so they stop showing up."""
    try:
        a, b = uuid.UUID(payload.primaryId), uuid.UUID(payload.duplicateId)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid contact ID format")
    ca = (await session.execute(select(ContactDB).where(ContactDB.id == a))).scalar_one_or_none()
    cb = (await session.execute(select(ContactDB).where(ContactDB.id == b))).scalar_one_or_none()
    if not ca or not cb:
        raise HTTPException(status_code=404, detail="Contact not found")
    lo, hi = _pair(a, b)
    for owner in {ca.user_id, cb.user_id}:  # a pair across two reps is dismissed for both of them
        exists = (await session.execute(select(DuplicateDismissalDB.id).where(
            DuplicateDismissalDB.user_id == owner, DuplicateDismissalDB.contact_a == lo, DuplicateDismissalDB.contact_b == hi))).scalar_one_or_none()
        if not exists:
            session.add(DuplicateDismissalDB(id=uuid.uuid4(), user_id=owner, contact_a=lo, contact_b=hi))
    await session.commit()
    return {"status": "dismissed"}


class MergeContactsPayload(BaseModel):
    primaryId: str
    duplicateId: str
    mergedFields: Optional[Dict[str, Any]] = None


@api_router.post("/admin/contacts/merge", dependencies=[Depends(require_admin)])
async def merge_contacts(payload: MergeContactsPayload, session: AsyncSession = Depends(get_db)):
    """Merge a duplicate into the primary record. The duplicate goes to its owner's Recently
    deleted ("Merged into …") for 30 days, so a wrong merge can be undone by restoring it."""
    try:
        pid = uuid.UUID(payload.primaryId)
        did = uuid.UUID(payload.duplicateId)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid contact ID format")

    primary = (await session.execute(select(ContactDB).where(ContactDB.id == pid))).scalar_one_or_none()
    duplicate = (await session.execute(select(ContactDB).where(ContactDB.id == did))).scalar_one_or_none()
    if not primary:
        raise HTTPException(status_code=404, detail="Primary contact not found")
    if not duplicate:
        raise HTTPException(status_code=404, detail="Duplicate contact not found")

    archived = await contact_store.merge_contacts(session, primary, duplicate)
    return {
        "status": "merged",
        "primaryId": payload.primaryId,
        "duplicateId": payload.duplicateId,
        "message": f"Merged into {primary.name}. The duplicate is in Recently deleted for {RECENTLY_DELETED_DAYS} days.",
        "restorableUntil": _format_deleted(archived)["purgeAfter"],
    }


class BatchDeletePayload(BaseModel):
    contactIds: List[str]


@api_router.post("/admin/contacts/batch_delete", dependencies=[Depends(require_admin)])
async def batch_delete_contacts(payload: BatchDeletePayload, session: AsyncSession = Depends(get_db)):
    """Delete several contacts at once. Each goes to its owner's Recently deleted for 30 days."""
    valid_uuids = []
    for cid in payload.contactIds:
        try:
            valid_uuids.append(uuid.UUID(cid))
        except ValueError:
            pass

    contacts = []
    if valid_uuids:
        contacts = (await session.execute(select(ContactDB).where(ContactDB.id.in_(valid_uuids)))).scalars().all()
        await contact_store.delete_many(session, list(contacts))

    return {
        "status": "deleted",
        "count": len(contacts),
        "deletedIds": [str(c.id) for c in contacts],
        "message": f"{len(contacts)} contact(s) moved to Recently deleted for {RECENTLY_DELETED_DAYS} days",
    }


# ==================== CONTACT LISTS & SEGMENTS ====================

class CreateContactListPayload(BaseModel):
    name: str = Field(..., max_length=255)


def _list_out(l: "ContactListDB", count: int, names: Dict[Any, str]) -> dict:
    return {
        "id": str(l.id),
        "name": l.name,
        "memberCount": count,
        "createdAt": _iso(l.created_at),  # None when it wasn't recorded (never a made-up "now")
        "createdBy": names.get(l.admin_id) if l.admin_id else None,
    }


async def _assert_list_name_free(session: AsyncSession, name: str, except_id: Optional[uuid.UUID] = None) -> None:
    for lid, other in (await session.execute(select(ContactListDB.id, ContactListDB.name))).all():
        if lid != except_id and _key(other) == _key(name):
            raise HTTPException(status_code=409, detail=f"There's already a list called “{other}”")


@api_router.get("/admin/contact_lists", dependencies=[Depends(require_admin)])
async def get_contact_lists(session: AsyncSession = Depends(get_db)):
    """Every contact list, newest first, with how many contacts it has and who made it."""
    lists = (await session.execute(select(ContactListDB).order_by(ContactListDB.created_at.desc()))).scalars().all()
    counts = dict((await session.execute(
        select(ContactListMemberDB.list_id, func.count(ContactListMemberDB.id))
        .where(ContactListMemberDB.contact_id.in_(select(ContactDB.id)))  # only contacts that still exist
        .group_by(ContactListMemberDB.list_id))).all())
    names = {u.id: u.name for u in (await session.execute(select(UserDB))).scalars().all()}
    return [_list_out(l, counts.get(l.id, 0), names) for l in lists]


@api_router.post("/admin/contact_lists")
async def create_contact_list(payload: CreateContactListPayload, session: AsyncSession = Depends(get_db), admin_id: uuid.UUID = Depends(require_admin)):
    """Create a new contact list (names are unique, ignoring case and spacing)."""
    name = " ".join(payload.name.split())
    if not name:
        raise HTTPException(status_code=400, detail="The list needs a name.")
    await _assert_list_name_free(session, name)
    cl = ContactListDB(id=uuid.uuid4(), name=name, admin_id=admin_id, created_at=datetime.now(timezone.utc))
    session.add(cl)
    await session.commit()
    names = {u.id: u.name for u in (await session.execute(select(UserDB).where(UserDB.id == admin_id))).scalars().all()}
    await broadcast_live_event("list_changed", {"id": str(cl.id), "name": cl.name})
    return _list_out(cl, 0, names)


class AddContactsToListPayload(BaseModel):
    contactIds: List[str]


async def _load_list(session: AsyncSession, list_id: str) -> "ContactListDB":
    try:
        lid = uuid.UUID(list_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid list ID")
    cl = (await session.execute(select(ContactListDB).where(ContactListDB.id == lid))).scalar_one_or_none()
    if not cl:
        raise HTTPException(status_code=404, detail="List not found")
    return cl


@api_router.get("/admin/contact_lists/{list_id}/contacts", dependencies=[Depends(require_admin)])
async def get_list_contacts(list_id: str, session: AsyncSession = Depends(get_db)):
    """The contacts in a list (empty list → [])."""
    cl = await _load_list(session, list_id)
    added = dict((await session.execute(
        select(ContactListMemberDB.contact_id, ContactListMemberDB.added_at).where(ContactListMemberDB.list_id == cl.id))).all())
    rows = (await session.execute(select(ContactDB).where(ContactDB.id.in_(list(added))).order_by(ContactDB.name))).scalars().all() if added else []
    # each contact plus when it was put in this list
    views = await console_scores(session, None, [format_contact(c) for c in rows])
    return [{**v, "addedToListAt": _iso(added.get(c.id))} for v, c in zip(views, rows)]


@api_router.post("/admin/contact_lists/{list_id}/contacts", dependencies=[Depends(require_admin)])
async def add_contacts_to_list(list_id: str, payload: AddContactsToListPayload, session: AsyncSession = Depends(get_db)):
    """Add contacts to a list. Unknown ids and contacts already in the list are skipped."""
    cl = await _load_list(session, list_id)
    wanted = set()
    for cid in payload.contactIds:
        try:
            wanted.add(uuid.UUID(cid))
        except ValueError:
            pass
    existing = set((await session.execute(select(ContactDB.id).where(ContactDB.id.in_(wanted)))).scalars().all()) if wanted else set()
    already = set((await session.execute(select(ContactListMemberDB.contact_id).where(ContactListMemberDB.list_id == cl.id))).scalars().all())
    new = existing - already
    now = datetime.now(timezone.utc)
    for cid in new:
        session.add(ContactListMemberDB(id=uuid.uuid4(), list_id=cl.id, contact_id=cid, added_at=now))
    await session.commit()
    if new:
        await broadcast_live_event("list_changed", {"id": str(cl.id), "name": cl.name, "added": len(new)})
    return {"status": "added", "count": len(new), "alreadyInList": len(existing & already), "notFound": len(wanted - existing)}


@api_router.delete("/admin/contact_lists/{list_id}/contacts/{contact_id}", dependencies=[Depends(require_admin)])
async def remove_contact_from_list(list_id: str, contact_id: str, session: AsyncSession = Depends(get_db)):
    cl = await _load_list(session, list_id)
    try:
        cid = uuid.UUID(contact_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid contact ID")
    res = await session.execute(delete(ContactListMemberDB).where(ContactListMemberDB.list_id == cl.id, ContactListMemberDB.contact_id == cid))
    await session.commit()
    if not res.rowcount:
        raise HTTPException(status_code=404, detail="That contact isn't in this list")
    await broadcast_live_event("list_changed", {"id": str(cl.id), "removed": 1})
    return {"status": "removed"}


@api_router.patch("/admin/contact_lists/{list_id}", dependencies=[Depends(require_admin)])
async def rename_contact_list(list_id: str, payload: CreateContactListPayload, session: AsyncSession = Depends(get_db)):
    cl = await _load_list(session, list_id)
    name = " ".join(payload.name.split())
    if not name:
        raise HTTPException(status_code=400, detail="The list needs a name.")
    await _assert_list_name_free(session, name, except_id=cl.id)
    cl.name = name
    await session.commit()
    await broadcast_live_event("list_changed", {"id": str(cl.id), "name": cl.name})
    return {"id": str(cl.id), "name": cl.name}


@api_router.delete("/admin/contact_lists/{list_id}", dependencies=[Depends(require_admin)])
async def delete_contact_list(list_id: str, session: AsyncSession = Depends(get_db)):
    """Delete a list (the contacts themselves are kept)."""
    cl = await _load_list(session, list_id)
    await session.execute(delete(ContactListMemberDB).where(ContactListMemberDB.list_id == cl.id))
    await session.delete(cl)
    await session.commit()
    await broadcast_live_event("list_changed", {"id": list_id, "deleted": True})
    return {"status": "deleted", "id": list_id}


# ==================== AI PIPELINES (RESEARCH → PITCH → DECK) ====================

PIPELINE_RUNNING = ("pending", "researching", "scoring", "pitching", "generating_deck", "attaching")
# A run whose record hasn't moved for this long is dead (its task is gone); it no longer blocks a new run.
PIPELINE_STALE_MINUTES = 15
pipeline_settings_update_impl = None  # (enabled: bool) -> None   (the auto-pipeline switch)


def _research_out(data: Any) -> Optional[Dict[str, Any]]:
    """The structured research the AI returned (person, company, pain points, …), or None when it
    couldn't be read as JSON (the summary still has the text)."""
    if not isinstance(data, dict) or data.get("parse_error"):
        return None
    lst = lambda v: [str(x) for x in v if str(x).strip()] if isinstance(v, list) else ([str(v)] if v else [])
    person = data.get("person") if isinstance(data.get("person"), dict) else {}
    company = data.get("company") if isinstance(data.get("company"), dict) else {}
    return {
        "person": {k: str(person.get(k) or "") for k in ("name", "likely_title", "seniority", "authority")},
        "company": {k: str(company.get(k) or "") for k in ("name", "industry", "size_estimate", "products", "growth_signals")},
        "industry": str(data.get("industry") or company.get("industry") or ""),
        "painPoints": lst(data.get("pain_points")),
        "opportunities": lst(data.get("opportunities")),
        "talkingPoints": lst(data.get("talking_points")),
        "competitiveNotes": str(data.get("competitive_notes") or ""),
        "confidence": str(data.get("confidence_level") or ""),
    }


def _pipeline_out(p: "ContactPipelineDB", contact: Optional["ContactDB"] = None, owner: Optional[str] = None,
                  admin_score: Optional["AdminContactScoreDB"] = None) -> Dict[str, Any]:
    return {
        "id": str(p.id),
        "contactId": str(p.contact_id),
        "contactExists": contact is not None,  # False once the contact was deleted (nothing to open / re-run)
        "contactName": contact.name if contact else None,
        "company": _present(contact.company_name) if contact else None,
        # `score` = the console's (admin) score like everywhere in the console; the rep's own stays in repScore
        "score": admin_score.score if admin_score else None,
        "repScore": contact.lead_score if contact else None,
        "adminScore": ({"score": admin_score.score, "temperature": admin_score.temperature, "reasoning": admin_score.reasoning or "",
                        "scoredAt": _iso(admin_score.scored_at)} if admin_score else None),
        "scoreCompleted": bool(p.score_completed),
        "research": _research_out(p.research_data),
        "userId": str(p.user_id),
        "owner": owner,
        "status": p.status,
        "running": p.status in PIPELINE_RUNNING,
        "currentStep": p.current_step or "",
        "error": p.error_message,
        "researchSummary": p.research_summary or "",
        "pitchAngle": p.pitch_angle or "",
        "pitchSubject": p.pitch_email_subject or "",
        "pitchEmail": p.pitch_email_body or "",
        "slides": p.pitch_slides_content or [],
        "deckFileId": str(p.deck_file_id) if p.deck_file_id else None,
        "gammaDeckUrl": getattr(p, "gamma_deck_url", None),
        "startedAt": _iso(p.started_at),
        "completedAt": _iso(p.completed_at),
    }


def _pipeline_settings() -> Dict[str, Any]:
    return pipeline_settings_impl() if pipeline_settings_impl else {"aiConfigured": False, "gammaConfigured": False, "autoPipelineEnabled": False}


async def _close_stale_runs(session: AsyncSession, contact_id: Optional[uuid.UUID] = None) -> int:
    """Runs that still say "running" but haven't moved for PIPELINE_STALE_MINUTES lost their task
    (server restart / crash): mark them failed so nothing shows them as live forever."""
    now = datetime.now(timezone.utc)
    q = select(ContactPipelineDB).where(ContactPipelineDB.status.in_(PIPELINE_RUNNING))
    if contact_id is not None:
        q = q.where(ContactPipelineDB.contact_id == contact_id)
    closed = 0
    for r in (await session.execute(q)).scalars().all():
        last = _aware(r.updated_at) or _aware(r.started_at) or _aware(r.created_at)
        if last and now - last > timedelta(minutes=PIPELINE_STALE_MINUTES):
            r.status, r.current_step = "failed", "Stopped responding"
            r.error_message = f"This run stopped responding for over {PIPELINE_STALE_MINUTES} minutes (the server restarted or the AI hung) and was closed. Retry it."
            r.completed_at = now
            closed += 1
    if closed:
        await session.commit()
    return closed


@api_router.get("/admin/pipelines", dependencies=[Depends(require_admin)])
async def list_pipelines(status: Optional[str] = Query(None), limit: int = Query(100, ge=1, le=500),
                         offset: int = Query(0, ge=0), session: AsyncSession = Depends(get_db)):
    """Real pipeline runs (newest first, `offset`/`limit` pages) with what each step produced, status
    counts and setup. `total` = runs matching the status filter (all runs without one)."""
    await _close_stale_runs(session)
    # the console filters by group: running = any step in progress, complete = with or without a deck
    groups = {"running": list(PIPELINE_RUNNING), "complete": ["complete", "complete_no_deck"]}
    wanted = groups.get(status, [status]) if status else None
    q = select(ContactPipelineDB).order_by(ContactPipelineDB.started_at.desc(), ContactPipelineDB.id).offset(offset).limit(limit)
    if wanted:
        q = q.where(ContactPipelineDB.status.in_(wanted))
    runs = (await session.execute(q)).scalars().all()
    ids = {p.contact_id for p in runs}
    contacts = {c.id: c for c in (await session.execute(select(ContactDB).where(ContactDB.id.in_(ids)))).scalars().all()} if runs else {}
    admin_scores = {s.contact_id: s for s in (await session.execute(select(AdminContactScoreDB).where(AdminContactScoreDB.contact_id.in_(ids)))).scalars().all()} if runs else {}
    owners = {u.id: u.name for u in (await session.execute(select(UserDB))).scalars().all()}
    counts = dict((await session.execute(select(ContactPipelineDB.status, func.count(ContactPipelineDB.id)).group_by(ContactPipelineDB.status))).all())
    return {"pipelines": [_pipeline_out(p, contacts.get(p.contact_id), owners.get(p.user_id), admin_scores.get(p.contact_id)) for p in runs],
            "statusCounts": counts, "total": sum(n for s, n in counts.items() if s in wanted) if wanted else sum(counts.values()),
            "offset": offset, "settings": _pipeline_settings()}


class PipelineSettingsPayload(BaseModel):
    autoPipelineEnabled: bool


@api_router.post("/admin/pipelines/settings", dependencies=[Depends(require_admin)])
async def update_pipeline_settings(payload: PipelineSettingsPayload):
    """Turn the auto-pipeline on or off: after every card / QR scan the pipeline runs for the rep
    (their own run — it writes to their contact). Kept until the server restarts; set
    AUTO_PIPELINE_ENABLED=true in the server's .env to make it the default."""
    setter = _need(pipeline_settings_update_impl, "The pipeline settings")
    setter(payload.autoPipelineEnabled)
    return _pipeline_settings()


async def _start_pipeline(session: AsyncSession, contact_id: str, retry: bool) -> Dict[str, Any]:
    run = _need(pipeline_impl, "The AI pipeline")
    settings = _pipeline_settings()
    if not settings.get("aiConfigured"):
        raise HTTPException(status_code=503, detail="The AI isn't set up on the server (OPENROUTER_API_KEY), so the pipeline can't run.")
    try:
        cid = uuid.UUID(contact_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid contact ID")
    contact = (await session.execute(select(ContactDB).where(ContactDB.id == cid))).scalar_one_or_none()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")
    now = datetime.now(timezone.utc)
    alive = False
    for r in (await session.execute(select(ContactPipelineDB).where(
            ContactPipelineDB.contact_id == cid, ContactPipelineDB.status.in_(PIPELINE_RUNNING)))).scalars().all():
        last = _aware(r.updated_at) or _aware(r.started_at) or _aware(r.created_at)
        if last and now - last > timedelta(minutes=PIPELINE_STALE_MINUTES):
            # its task is gone (crash / killed worker): close it so it stops blocking new runs
            r.status, r.current_step = "failed", "Stopped responding"
            r.error_message = f"This run stopped responding for over {PIPELINE_STALE_MINUTES} minutes and was closed."
            r.completed_at = now
        else:
            alive = True
    await session.commit()
    if alive:
        return {"status": "already_running", "contactId": contact_id, "message": f"A pipeline is already running for {contact.name}."}
    if retry:
        await session.execute(delete(ContactPipelineDB).where(ContactPipelineDB.contact_id == cid, ContactPipelineDB.status == "failed"))
        await session.commit()
    # runs in the background against the contact's owner profile (it drives research + pitch);
    # admin=True: the run never writes to the rep's contact (score, notes, links stay the rep's)
    task = asyncio.create_task(run(contact_id, str(contact.user_id), admin=True))
    _pipeline_tasks.add(task)
    task.add_done_callback(_pipeline_tasks.discard)
    # The run writes its own record first thing; wait for it (≤ 3 s) so a status check right
    # after this answer already sees the run instead of "never run".
    for _ in range(60):
        await asyncio.sleep(0.05)
        if task.done() or (await session.execute(select(ContactPipelineDB.id).where(
                ContactPipelineDB.contact_id == cid, ContactPipelineDB.status.in_(PIPELINE_RUNNING)))).first():
            break
    await broadcast_live_event("pipeline_started", {"contactId": contact_id, "name": contact.name})
    return {"status": "started", "contactId": contact_id, "message": f"Pipeline started for {contact.name}."}


_pipeline_tasks: set = set()


@api_router.post("/contact/{contact_id}/pipeline/run", dependencies=[Depends(require_admin)])
async def run_contact_pipeline(contact_id: str, session: AsyncSession = Depends(get_db)):
    """Start the real pipeline: research the company → score → write the pitch → Gamma deck (if
    GAMMA_API_KEY) → attach the deck to the contact. Watch it in GET …/pipeline."""
    return await _start_pipeline(session, contact_id, retry=False)


@api_router.post("/contact/{contact_id}/pipeline/retry", dependencies=[Depends(require_admin)])
async def retry_contact_pipeline(contact_id: str, session: AsyncSession = Depends(get_db)):
    """Clear the contact's failed runs and start again."""
    return await _start_pipeline(session, contact_id, retry=True)


@api_router.get("/contact/{contact_id}/pipeline", dependencies=[Depends(require_admin)])
async def get_contact_pipeline(contact_id: str, session: AsyncSession = Depends(get_db)):
    """The contact's latest pipeline run and what it produced ({status: "none"} if never run)."""
    try:
        cid = uuid.UUID(contact_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid contact ID")
    await _close_stale_runs(session, cid)
    p = (await session.execute(select(ContactPipelineDB).where(ContactPipelineDB.contact_id == cid)
                               .order_by(ContactPipelineDB.started_at.desc()).limit(1))).scalars().first()
    if not p:
        return {"status": "none", "contactId": contact_id, "running": False, "settings": _pipeline_settings()}
    contact = (await session.execute(select(ContactDB).where(ContactDB.id == cid))).scalar_one_or_none()
    admin_score = (await session.execute(select(AdminContactScoreDB).where(AdminContactScoreDB.contact_id == cid))).scalar_one_or_none()
    return {**_pipeline_out(p, contact, None, admin_score), "settings": _pipeline_settings()}


# ==================== EXHIBITORS DIRECTORY ====================

@api_router.get("/exhibitors", dependencies=[Depends(request_user_id)])
async def list_exhibitors(
    search: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    hall: Optional[str] = Query(None),
    event: Optional[str] = Query(None, description="Only exhibitors of this event"),
    limit: int = Query(500, ge=1, le=2000),
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """Browse conference exhibitors directory with instant search and filters.
    For the signed-in user each booth also says whether they marked it visited."""
    query = select(ExhibitorDB).order_by(ExhibitorDB.name)
    conditions = []
    if search:
        s = f"%{search.strip()}%"
        conditions.append(or_(
            ExhibitorDB.name.ilike(s),
            ExhibitorDB.description.ilike(s),
            ExhibitorDB.booth.ilike(s)
        ))
    if category:
        conditions.append(ExhibitorDB.category == category)
    if hall:
        conditions.append(ExhibitorDB.hall == hall)
    if event:
        conditions.append(await event_filter(session, ExhibitorDB.event_name, event))

    if conditions:
        query = query.where(and_(*conditions))

    res = await session.execute(query.limit(limit))
    rows = res.scalars().all()
    visits = await _visits_for(session, me, {_key(r.event_name) for r in rows})
    return [_exhibitor_out(r, me, visits.get((_key(r.event_name), _key(r.name)))) for r in rows]


async def _visits_for(session: AsyncSession, me: Optional[uuid.UUID], event_keys) -> Dict[tuple, "BoothVisitDB"]:
    if me is None or not event_keys:
        return {}
    res = await session.execute(select(BoothVisitDB).where(
        BoothVisitDB.user_id == me, BoothVisitDB.event_key.in_(list(event_keys))))
    return {(v.event_key, v.exhibitor_key): v for v in res.scalars().all()}


def _exhibitor_out(r: "ExhibitorDB", me: Optional[uuid.UUID], visit: Optional["BoothVisitDB"]) -> Dict[str, Any]:
    return {
        "id": str(r.id),
        "event": r.event_name,
        "name": r.name,
        "booth": r.booth,
        "hall": r.hall,
        "category": r.category,
        "country": r.country,
        "website": r.website,
        "description": r.description,
        "subcategory": r.subcategory or None,
        "tags": _as_list(r.tags),
        "products": _as_list(r.products),
        # a booth a rep added from the app (vs the imported directory); they can edit / remove it
        "addedByMe": bool(me is not None and r.added_by == me),
        "userAdded": r.added_by is not None,
        # the signed-in user marked this booth as visited
        "visited": visit is not None,
        "visitedAt": visit.visited_at.isoformat() if visit and visit.visited_at else None,
        "visitNote": (visit.note or "") if visit else "",
    }


def _as_list(v) -> List[str]:
    """tags / products are JSON lists, but older rows may hold a JSON string or "a, b" text."""
    if not v:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    if s.startswith("["):
        try:
            return [str(x).strip() for x in json.loads(s) if str(x).strip()]
        except ValueError:
            pass
    return [x.strip() for x in s.split(",") if x.strip()]


@api_router.get("/exhibitors/categories", dependencies=[Depends(request_user_id)])
async def list_exhibitor_categories(event: Optional[str] = Query(None), session: AsyncSession = Depends(get_db)):
    """Exhibitor categories with counts, most common first — for one event with `event`, else all.
    Only real categories (blank ones are skipped); comes from imported and rep-added booths."""
    q = select(ExhibitorDB.category, func.count(ExhibitorDB.id)).where(
        ExhibitorDB.category.is_not(None), func.trim(ExhibitorDB.category) != "")
    if event:
        q = q.where(await event_filter(session, ExhibitorDB.event_name, event))
    rows = (await session.execute(q.group_by(ExhibitorDB.category).order_by(func.count(ExhibitorDB.id).desc()))).all()
    return [{"category": r[0].strip(), "count": r[1]} for r in rows]


class BoothPayload(BaseModel):
    event: Optional[str] = Field(None, max_length=255)
    name: Optional[str] = Field(None, max_length=500)
    booth: Optional[str] = Field(None, max_length=100)
    hall: Optional[str] = Field(None, max_length=100)
    category: Optional[str] = Field(None, max_length=500)
    website: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = Field(None, max_length=5000)
    # POST only: also mark it visited for me (the usual case — "I'm at this booth")
    visited: bool = False
    note: Optional[str] = Field(None, max_length=2000)


class VisitPayload(BaseModel):
    note: Optional[str] = Field(None, max_length=2000)


def _one_line(v, n: int) -> str:
    return " ".join(str(v or "").split())[:n]


async def _mobile_user(me: Optional[uuid.UUID]) -> uuid.UUID:
    if me is None:
        raise HTTPException(status_code=400, detail="Use the app for this (it's per user).")
    return me


async def _load_exhibitor(session: AsyncSession, exhibitor_id: str) -> "ExhibitorDB":
    try:
        xid = uuid.UUID(exhibitor_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid exhibitor ID")
    row = (await session.execute(select(ExhibitorDB).where(ExhibitorDB.id == xid))).scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Exhibitor not found")
    return row


async def _can_edit_booth(session: AsyncSession, row: "ExhibitorDB", me: Optional[uuid.UUID]) -> bool:
    if me is None:  # admin console
        return True
    if row.added_by == me:
        return True
    user = (await session.execute(select(UserDB).where(UserDB.id == me))).scalar_one_or_none()
    return bool(user and getattr(user, "is_admin", False))


async def _set_visit(session: AsyncSession, me: uuid.UUID, row: "ExhibitorDB", note: Optional[str]) -> "BoothVisitDB":
    ek, xk = _key(row.event_name), _key(row.name)
    visit = (await session.execute(select(BoothVisitDB).where(
        BoothVisitDB.user_id == me, BoothVisitDB.event_key == ek, BoothVisitDB.exhibitor_key == xk))).scalar_one_or_none()
    if visit is None:
        visit = BoothVisitDB(id=uuid.uuid4(), user_id=me, event_key=ek, exhibitor_key=xk,
                             note=(note or "").strip(), visited_at=datetime.now(timezone.utc))
        session.add(visit)
    elif note is not None:
        visit.note = note.strip()
    return visit


@api_router.post("/exhibitors")
async def add_booth(payload: BoothPayload, session: AsyncSession = Depends(get_db),
                    me: Optional[uuid.UUID] = Depends(request_user_id),
                    authorization: Optional[str] = Header(None)):
    """Add a booth to an event's exhibitor list from the app (e.g. one missing from the directory).
    If the event already lists a booth with that name, that one is returned (blank booth / hall
    filled in) instead of a duplicate. `visited: true` also marks it visited for you."""
    # Booths added by hand (app or console) keep an owner, so a "replace" directory import —
    # which only clears imported rows (added_by NULL) — never wipes them.
    added_by = me if me is not None else await require_admin(authorization)
    event, name = _one_line(payload.event, 255), _one_line(payload.name, 500)
    if not event or not name:
        raise HTTPException(status_code=400, detail="The booth needs a name and an event.")
    row = (await session.execute(select(ExhibitorDB).where(
        await event_filter(session, ExhibitorDB.event_name, event),
        func.lower(func.trim(ExhibitorDB.name)) == _key(name)).limit(1))).scalars().first()
    existed = row is not None
    if row is None:
        row = ExhibitorDB(
            id=uuid.uuid4(), event_name=event, name=name,
            booth=_one_line(payload.booth, 100), hall=_one_line(payload.hall, 100),
            category=_one_line(payload.category, 500), website=_one_line(payload.website, 500),
            description=(payload.description or "").strip(), country="", subcategory="",
            products=[], tags=[], added_by=added_by, created_at=datetime.now(timezone.utc),
        )
        session.add(row)
    else:
        for field, n in (("booth", 100), ("hall", 100), ("category", 500), ("website", 500)):
            v = _one_line(getattr(payload, field), n)
            if v and not getattr(row, field):
                setattr(row, field, v)
    visit = None
    if payload.visited and me is not None:
        visit = await _set_visit(session, me, row, payload.note)
    await session.commit()
    return {**_exhibitor_out(row, me, visit), "existed": existed}


@api_router.patch("/exhibitors/{exhibitor_id}")
async def update_booth(exhibitor_id: str, payload: BoothPayload, session: AsyncSession = Depends(get_db),
                       me: Optional[uuid.UUID] = Depends(request_user_id)):
    """Edit a booth you added (admins: any booth)."""
    row = await _load_exhibitor(session, exhibitor_id)
    if not await _can_edit_booth(session, row, me):
        raise HTTPException(status_code=403, detail="Only the person who added this booth can edit it.")
    old_key = _key(row.name)
    for field, n in (("name", 500), ("booth", 100), ("hall", 100), ("category", 500), ("website", 500)):
        v = getattr(payload, field)
        if v is not None:
            v = _one_line(v, n)
            if field == "name" and not v:
                raise HTTPException(status_code=400, detail="The booth needs a name.")
            setattr(row, field, v)
    if payload.description is not None:
        row.description = payload.description.strip()
    if _key(row.name) != old_key:  # renamed: keep everyone's "visited" on it
        await session.execute(update(BoothVisitDB).where(
            BoothVisitDB.event_key == _key(row.event_name), BoothVisitDB.exhibitor_key == old_key,
        ).values(exhibitor_key=_key(row.name)))
    await session.commit()
    visits = await _visits_for(session, me, {_key(row.event_name)})
    return _exhibitor_out(row, me, visits.get((_key(row.event_name), _key(row.name))))


@api_router.delete("/exhibitors/{exhibitor_id}")
async def delete_booth(exhibitor_id: str, session: AsyncSession = Depends(get_db),
                       me: Optional[uuid.UUID] = Depends(request_user_id)):
    """Remove a booth you added (admins: any booth). Visits to it are removed too."""
    row = await _load_exhibitor(session, exhibitor_id)
    if not await _can_edit_booth(session, row, me):
        raise HTTPException(status_code=403, detail="Only the person who added this booth can remove it.")
    await session.execute(delete(BoothVisitDB).where(
        BoothVisitDB.event_key == _key(row.event_name), BoothVisitDB.exhibitor_key == _key(row.name)))
    await session.delete(row)
    await session.commit()
    return {"status": "deleted", "id": exhibitor_id}


@api_router.put("/exhibitors/{exhibitor_id}/visit")
async def mark_booth_visited(exhibitor_id: str, payload: VisitPayload = Body(default_factory=VisitPayload),
                             session: AsyncSession = Depends(get_db),
                             me: Optional[uuid.UUID] = Depends(request_user_id)):
    """Mark a booth as visited by you (optionally with a note). Counts in the event's "Booths met"."""
    user = await _mobile_user(me)
    row = await _load_exhibitor(session, exhibitor_id)
    visit = await _set_visit(session, user, row, payload.note)
    await session.commit()
    return _exhibitor_out(row, user, visit)


@api_router.delete("/exhibitors/{exhibitor_id}/visit")
async def unmark_booth_visited(exhibitor_id: str, session: AsyncSession = Depends(get_db),
                               me: Optional[uuid.UUID] = Depends(request_user_id)):
    user = await _mobile_user(me)
    row = await _load_exhibitor(session, exhibitor_id)
    await session.execute(delete(BoothVisitDB).where(
        BoothVisitDB.user_id == user, BoothVisitDB.event_key == _key(row.event_name),
        BoothVisitDB.exhibitor_key == _key(row.name)))
    await session.commit()
    return _exhibitor_out(row, user, None)


class ImportExhibitorsPayload(BaseModel):
    exhibitors: List[Dict[str, Any]]
    # The event these exhibitors belong to (must match the event name users pick in the app).
    # An item's own "event_name" / "event" wins. Used to be hard-coded to "WHX 2026".
    event: Optional[str] = None
    # Replace that event's existing directory instead of adding duplicates.
    replace: bool = False


@api_router.post("/admin/import_exhibitors", dependencies=[Depends(require_admin)])
async def import_exhibitors(payload: ImportExhibitorsPayload, session: AsyncSession = Depends(get_db)):
    """Bulk import an event's exhibitor directory (JSON). Powers the event page's exhibitor list,
    "Booths met" (exhibitors whose company you have a contact from) and the event list."""
    clean = lambda v, n: " ".join(str(v or "").split())[:n]
    default_event = clean(payload.event, 255)
    rows, per_event, skipped = [], {}, 0
    for item in payload.exhibitors:
        name = clean(item.get("name"), 500)
        event = clean(item.get("event_name") or item.get("event") or default_event, 255)
        if not name or not event:
            skipped += 1
            continue
        rows.append(ExhibitorDB(
            id=uuid.uuid4(),
            event_name=event,
            name=name,
            booth=clean(item.get("booth"), 100),
            hall=clean(item.get("hall"), 100),
            category=clean(item.get("category"), 500),
            subcategory=clean(item.get("subcategory"), 500),
            country=clean(item.get("country"), 100),
            website=clean(item.get("website"), 500),
            description=str(item.get("description") or "").strip(),
            products=_as_list(item.get("products")),
            tags=_as_list(item.get("tags")),
            created_at=datetime.now(timezone.utc),
        ))
        per_event[event] = per_event.get(event, 0) + 1
    if not rows:
        raise HTTPException(status_code=400, detail="Nothing to import — each exhibitor needs a name, and set the event name.")
    replaced = 0
    if payload.replace:
        # Replaces the imported directory. Booths reps added in the app stay, unless the new
        # directory lists the same name (then it's the directory's row). Visits are kept by name.
        for event in per_event:
            ev = await event_filter(session, ExhibitorDB.event_name, event)
            res = await session.execute(delete(ExhibitorDB).where(ev, ExhibitorDB.added_by.is_(None)))
            replaced += res.rowcount or 0
            names = [_key(r.name) for r in rows if r.event_name == event]
            if names:
                await session.execute(delete(ExhibitorDB).where(
                    ev, ExhibitorDB.added_by.is_not(None), func.lower(func.trim(ExhibitorDB.name)).in_(names)))
    session.add_all(rows)
    await session.commit()
    return {"status": "imported", "count": len(rows), "skipped": skipped, "replaced": replaced, "events": per_event}


# ==================== FILES & EVENT COLLATERAL ====================

MAX_FILE_BYTES = 20 * 1024 * 1024
FILE_TYPES = {  # extension -> MIME type (sales collateral reps open on their phone)
    "pdf": "application/pdf",
    "ppt": "application/vnd.ms-powerpoint",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xls": "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "csv": "text/csv",
    "txt": "text/plain",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
}


def _stored_len(n) -> int:
    return int(n or 0)


@api_router.get("/files/list", dependencies=[Depends(request_user_id)])
async def list_event_files(event: Optional[str] = Query(None), session: AsyncSession = Depends(get_db)):
    """Sales material & event documents the admins uploaded (Admin console → Files).

    `placeholder` = a demo row whose stored content isn't the real file (the old upload saved only a
    name + a fake size); the app hides those and they can't be downloaded."""
    q = (select(EventFileDB.id, EventFileDB.original_filename, EventFileDB.filename, EventFileDB.file_type,
                EventFileDB.mime_type, EventFileDB.category, EventFileDB.file_size, EventFileDB.description,
                EventFileDB.download_count, EventFileDB.event_name, EventFileDB.created_at,
                func.length(EventFileDB.file_data).label("stored"))
         .where(EventFileDB.is_active == True)  # noqa: E712
         .order_by(EventFileDB.created_at.desc()))
    if event:
        # files for that event, plus ones meant for every event (no event set)
        q = q.where(or_(await event_filter(session, EventFileDB.event_name, event),
                        EventFileDB.event_name.is_(None), EventFileDB.event_name == ""))
    rows = (await session.execute(q)).all()
    return [
        {
            "id": str(r.id),
            "filename": r.original_filename or r.filename,
            "fileType": r.file_type,
            "mimeType": r.mime_type,
            "category": r.category,
            "fileSize": _stored_len(r.stored),
            "description": r.description or "",
            "downloadCount": r.download_count or 0,
            "eventName": r.event_name or None,
            "placeholder": _stored_len(r.stored) == 0 or _stored_len(r.stored) != int(r.file_size or 0),
            "createdAt": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


class UploadFilePayload(BaseModel):
    filename: str = Field(..., max_length=255)
    # the file itself, base64 (Admin console reads the picked file). Required — no more name-only rows.
    contentBase64: str
    mimeType: Optional[str] = None
    category: Optional[str] = "general"
    description: Optional[str] = ""
    # which event it's for; empty = every event
    event: Optional[str] = None


@api_router.post("/admin/files/upload")
async def upload_event_file(payload: UploadFilePayload, session: AsyncSession = Depends(get_db), uid: uuid.UUID = Depends(require_admin)):
    """Upload an event document (PDF, deck, sheet, image…, max 20 MB). Reps see it in the app
    (Me → Team files) and can open / save it."""
    name = _one_line(payload.filename, 255)
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in FILE_TYPES:
        raise HTTPException(status_code=400, detail=f"File type not allowed. Use: {', '.join(sorted(FILE_TYPES))}.")
    try:
        data = base64.b64decode(payload.contentBase64.split(",", 1)[-1], validate=True)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="The file content isn't valid base64.")
    if not data:
        raise HTTPException(status_code=400, detail="The file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=400, detail=f"File too large — max 20 MB (this one is {len(data) / 1024 / 1024:.1f} MB).")
    fid = uuid.uuid4()
    ef = EventFileDB(
        id=fid,
        filename=re.sub(r"[^\w\s\-.]", "_", name),
        original_filename=name,
        file_type=ext,
        mime_type=FILE_TYPES[ext],
        file_size=len(data),
        file_data=data,
        description=(payload.description or "").strip(),
        event_name=_one_line(payload.event, 255),
        category=_one_line(payload.category or "general", 100) or "general",
        uploaded_by=uid,
        download_count=0,
        created_at=datetime.now(timezone.utc),
    )
    session.add(ef)
    await session.commit()
    await broadcast_live_event("files_changed", {"id": str(fid), "filename": name, "event": ef.event_name})
    return {"id": str(fid), "filename": name, "fileSize": len(data), "status": "uploaded"}


def _file_payload(name: str, mime: Optional[str], data: bytes) -> Dict[str, Any]:
    return {"filename": name, "mimeType": mime or "application/octet-stream", "fileSize": len(data),
            "contentBase64": base64.b64encode(data).decode()}


@api_router.get("/files/{file_id}/download")
async def download_event_file(file_id: str, session: AsyncSession = Depends(get_db), me: Optional[uuid.UUID] = Depends(request_user_id)):
    """The file's content (base64 JSON, so the app can save / open it with its sign-in).
    `downloadCount` counts reps opening it in the app; the admin console (me = None) doesn't add to it."""
    try:
        fid = uuid.UUID(file_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file ID")
    f = (await session.execute(select(EventFileDB).where(EventFileDB.id == fid, EventFileDB.is_active == True))).scalar_one_or_none()  # noqa: E712
    if not f:
        raise HTTPException(status_code=404, detail="File not found")
    data = bytes(f.file_data or b"")
    if not data or len(data) != int(f.file_size or 0):
        raise HTTPException(status_code=410, detail="This is a demo placeholder without the real file. Ask an admin to upload it again.")
    if me is not None:  # a rep in the app — the admin console's own downloads don't count
        f.download_count = (f.download_count or 0) + 1
        await session.commit()
    return _file_payload(f.original_filename or f.filename, f.mime_type, data)


@api_router.delete("/admin/files/{file_id}", dependencies=[Depends(require_admin)])
async def delete_event_file(file_id: str, session: AsyncSession = Depends(get_db)):
    """Delete event collateral document."""
    try:
        fid = uuid.UUID(file_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file ID")

    await session.execute(delete(EventFileDB).where(EventFileDB.id == fid))
    await session.commit()
    await broadcast_live_event("files_changed", {"id": file_id, "deleted": True})
    return {"status": "deleted", "id": file_id}


@api_router.get("/contact/{contact_id}/files")
async def get_contact_files(
    contact_id: str,
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """Documents actually attached to this contact (contact_files), newest first.
    The app (me set) never sees admin-only files; the admin console (me None) sees everything."""
    contact = await load_contact(session, contact_id, me)
    query = select(ContactFileDB).where(ContactFileDB.contact_id == contact.id, ContactFileDB.is_active == True)  # noqa: E712
    if me is not None:
        query = query.where(or_(ContactFileDB.category.is_(None), ContactFileDB.category != ADMIN_FILE_CATEGORY))
    res = await session.execute(query.order_by(ContactFileDB.created_at.desc()))
    return [
        {
            "id": str(f.id),
            "filename": f.original_filename,
            "fileType": f.file_type,
            "category": f.category,
            "fileSize": f.file_size,
            "description": f.description or "",
            "createdAt": f.created_at.isoformat() if f.created_at else None,
        }
        for f in res.scalars().all()
    ]


class ContactFilePayload(BaseModel):
    filename: str = Field(..., max_length=255)
    contentBase64: str
    description: Optional[str] = Field(None, max_length=2000)


@api_router.post("/contact/{contact_id}/files")
async def attach_contact_file(
    contact_id: str,
    payload: ContactFilePayload,
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """Attach a document to one of your contacts (proposal, brochure they gave you, photo…), max 20 MB."""
    contact = await load_contact(session, contact_id, me)
    name = _one_line(payload.filename, 255)
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in FILE_TYPES:
        raise HTTPException(status_code=400, detail=f"File type not allowed. Use: {', '.join(sorted(FILE_TYPES))}.")
    try:
        data = base64.b64decode(payload.contentBase64.split(",", 1)[-1], validate=True)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="The file content isn't valid base64.")
    if not data:
        raise HTTPException(status_code=400, detail="The file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=400, detail=f"File too large — max 20 MB (this one is {len(data) / 1024 / 1024:.1f} MB).")
    now = datetime.now(timezone.utc)
    f = ContactFileDB(
        id=uuid.uuid4(), contact_id=contact.id,
        filename=re.sub(r"[^\w\s\-.]", "_", name), original_filename=name,
        file_type=ext, mime_type=FILE_TYPES[ext], file_size=len(data), file_data=data,
        description=(payload.description or "").strip(), category="general",
        uploaded_by=me or contact.user_id, is_active=True, download_count=0, created_at=now, updated_at=now,
    )
    session.add(f)
    await session.commit()
    await broadcast_live_event("contact_updated", {"contactId": str(contact.id), "name": contact.name, "file": name})
    return {"id": str(f.id), "filename": name, "fileType": ext, "category": "general", "fileSize": len(data),
            "description": f.description, "createdAt": now.isoformat()}


@api_router.delete("/contact/files/{file_id}")
async def remove_contact_file(
    file_id: str,
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """Remove a file from one of your contacts."""
    try:
        fid = uuid.UUID(file_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file ID")
    f = (await session.execute(select(ContactFileDB).where(ContactFileDB.id == fid, ContactFileDB.is_active == True))).scalar_one_or_none()  # noqa: E712
    if not f or (me is not None and f.category == ADMIN_FILE_CATEGORY):  # the app can't touch admin-only files
        raise HTTPException(status_code=404, detail="File not found")
    await load_contact(session, str(f.contact_id), me)  # 404 unless it's your contact
    await session.delete(f)
    await session.commit()
    return {"status": "deleted", "id": file_id}


@api_router.get("/contact/files/{file_id}/download")
async def download_contact_file(
    file_id: str,
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """A document attached to one of your contacts (base64 JSON)."""
    try:
        fid = uuid.UUID(file_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file ID")
    f = (await session.execute(select(ContactFileDB).where(ContactFileDB.id == fid, ContactFileDB.is_active == True))).scalar_one_or_none()  # noqa: E712
    if not f or (me is not None and f.category == ADMIN_FILE_CATEGORY):  # admin-only files aren't served to the app
        raise HTTPException(status_code=404, detail="File not found")
    await load_contact(session, str(f.contact_id), me)  # 404 unless it's your contact
    data = bytes(f.file_data or b"")
    if not data:
        raise HTTPException(status_code=410, detail="This file has no content.")
    f.download_count = (f.download_count or 0) + 1
    await session.commit()
    return _file_payload(f.original_filename or f.filename, f.mime_type, data)


# ==================== AI SCORING, ENRICHMENT & ADMIN NOTES ====================

# Scoring (rep and admin, profiles, fingerprints, storage) is scoring.score; the admin console's score is
# saved only to admin_contact_scores — the rep's score and data are never written from the admin console.


@api_router.get("/contact/{contact_id}/admin_score", dependencies=[Depends(require_admin)])
async def get_admin_score(contact_id: str, session: AsyncSession = Depends(get_db)):
    """The admin console's own score for this contact ({adminScore: null} if never scored here)."""
    try:
        cid = uuid.UUID(contact_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid contact ID")
    row = (await session.execute(select(AdminContactScoreDB).where(AdminContactScoreDB.contact_id == cid))).scalar_one_or_none()
    return {"contactId": contact_id, "adminScore": admin_score_out(row)}


def _require_scorer():
    if not scoring.configured():
        raise HTTPException(status_code=503, detail=scoring.NOT_AVAILABLE_DETAIL)


@api_router.post("/contact/{contact_id}/score")
async def score_single_contact(
    contact_id: str,
    # force=true asks the AI again even if nothing changed since the last score
    force: bool = Query(False),
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """Score a contact with the AI (OpenRouter) against its owner's profile (targets, products, goals).

    From the app (me = the rep): the rep's own score on the contact.
    From the admin console (me None): the admin's separate score — the rep's score and data are
    never changed by admin scoring.
    """
    contact = await load_contact(session, contact_id, me)
    owner_id, name = contact.user_id, contact.name
    _require_scorer()

    if me is None:
        result = await scoring.score(contact_id, "admin", force=force)
        await broadcast_live_event("admin_score_updated", {"contactId": contact_id, "name": name, "score": result["score"]})
        return {
            "contactId": contact_id, "admin": True,
            "score": result["score"], "scoreBand": band_for(result["score"]), "temperature": result["temperature"],
            "reasoning": result["reasoning"], "recommendedActions": result.get("recommended_actions", []),
            "breakdown": result.get("breakdown", {}), "unchanged": bool(result.get("unchanged")),
            "previousScore": result.get("previous_score"), "runs": result.get("runs"), "spread": result.get("spread"),
            "model": result.get("model"), "scoredAt": result.get("scored_at"),
        }

    result = await scoring.score(contact_id, "rep", force=force, owner=owner_id)
    band = band_for(result["score"])

    await broadcast_live_event("score_updated", {
        "contactId": contact_id,
        "name": name,
        "score": result["score"],
        "scoreBand": band,
    })

    return {
        "contactId": contact_id,
        "score": result["score"],
        "scoreBand": band,
        "temperature": result["temperature"],
        "reasoning": result["reasoning"],
        "recommendedActions": result.get("recommended_actions", []),
        "breakdown": result.get("breakdown", {}),
        # unchanged = nothing this score depends on changed, so the saved score was kept
        "unchanged": bool(result.get("unchanged")),
        "previousScore": result.get("previous_score"),
        "runs": result.get("runs"),
        "spread": result.get("spread"),
        "model": result.get("model"),
        "scoredAt": result.get("scored_at"),
    }


SCORE_ALL_CONCURRENCY = 4  # AI calls in flight at once (each score is itself up to SCORING_RUNS calls)


@api_router.post("/contacts/score_all")
async def score_all_contacts(
    limit: int = Query(25, ge=1, le=100),
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """Score every unscored contact. From the app (me = the rep): your own contacts, against your profile.

    From the admin console (me None): (re)ranks the whole workspace against the admin sales profile —
    contacts with no admin score first, then ones whose admin score is out of date (the profile or the
    contact changed). An up-to-date admin score is kept without an AI call and isn't counted; `limit`
    caps the AI calls per request and `pending` is how many contacts this run didn't reach. Whether
    the rep scored a contact doesn't matter, and the rep's score is never written.
    Contacts are scored a few at a time in parallel, so a batch takes about a quarter of the time.
    """
    state = {"scored": 0, "errors": 0, "first_error": None, "stop": False}

    async def attempt(call, label) -> Optional[dict]:
        try:
            return await call
        except Exception as e:
            detail = getattr(e, 'detail', None) or str(e)
            print(f"[SCORING] Could not score {label}: {detail}")
            state["errors"] += 1
            state["first_error"] = state["first_error"] or detail
            if isinstance(e, HTTPException) and e.status_code == 503:
                state["stop"] = True  # AI unavailable (bad key / rate limit): the rest would fail the same way
            return None

    _require_scorer()
    if me is None:
        rows = (await session.execute(
            select(ContactDB.id).outerjoin(AdminContactScoreDB, AdminContactScoreDB.contact_id == ContactDB.id)
            .order_by(AdminContactScoreDB.contact_id.is_not(None), AdminContactScoreDB.scored_at.asc(), ContactDB.created_at.desc())
        )).scalars().all()
        checked = 0
        while checked < len(rows) and state["scored"] < limit and not state["stop"]:
            wave = rows[checked:checked + min(SCORE_ALL_CONCURRENCY, limit - state["scored"])]
            checked += len(wave)
            for result in await asyncio.gather(*(attempt(scoring.score(cid, "admin"), cid) for cid in wave)):
                if result is not None and not result.get("unchanged"):
                    state["scored"] += 1
        if rows and state["scored"] == 0 and state["first_error"]:
            raise HTTPException(status_code=503, detail=str(state["first_error"]))
        await broadcast_live_event("batch_score_completed", {"scoredCount": state["scored"], "admin": True})
        return {"status": "completed", "scoredCount": state["scored"], "errors": state["errors"], "pending": len(rows) - checked}

    query = select(ContactDB.id, ContactDB.user_id).where(ContactDB.lead_score.is_(None), ContactDB.user_id == me)
    all_pending = (await session.execute(query.order_by(ContactDB.created_at.desc()))).all()
    batch = all_pending[:limit]  # each score is an AI call; keep one request reasonably short

    for i in range(0, len(batch), SCORE_ALL_CONCURRENCY):
        if state["stop"]:
            break
        wave = batch[i:i + SCORE_ALL_CONCURRENCY]
        for result in await asyncio.gather(*(attempt(scoring.score(cid, "rep", owner=owner), cid) for cid, owner in wave)):
            if result is not None:
                state["scored"] += 1

    if batch and state["scored"] == 0 and state["first_error"]:
        raise HTTPException(status_code=503, detail=str(state["first_error"]))

    await broadcast_live_event("batch_score_completed", {"scoredCount": state["scored"]})
    return {
        "status": "completed", "scoredCount": state["scored"], "errors": state["errors"],
        "pending": len(all_pending) - state["scored"],  # still unscored after this batch
    }


@api_router.post("/contact/{contact_id}/enrich")
async def enrich_contact(
    contact_id: str,
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """Company enrichment. No data provider is connected, so this says so instead of
    returning invented firmographics (it used to make up headcount, funding and tech stack)."""
    await load_contact(session, contact_id, me)
    raise HTTPException(
        status_code=501,
        detail="Company enrichment isn't connected to a data provider yet. Use lead scoring or the AI assistant instead.",
    )


class AdminNotePayload(BaseModel):
    notes: str


@api_router.put("/admin/contact/{contact_id}/note", dependencies=[Depends(require_admin)])
async def update_contact_admin_note(contact_id: str, payload: AdminNotePayload, session: AsyncSession = Depends(get_db)):
    """Save private admin intelligence notes on a contact."""
    try:
        cid = uuid.UUID(contact_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid contact ID")

    c_res = await session.execute(select(ContactDB).where(ContactDB.id == cid))
    contact = c_res.scalar_one_or_none()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")

    contact.admin_notes = payload.notes
    await contact_store.save(session, contact, "contact_updated")
    return {"status": "updated", "contactId": contact_id, "adminNotes": contact.admin_notes}


def _vcard_escape(value: Optional[str]) -> str:
    return (value or "").replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


@api_router.get("/contact/{contact_id}/qr")
async def get_contact_qr(
    contact_id: str,
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """A real, scannable vCard QR code built from the contact's current details."""
    c = await load_contact(session, contact_id, me)
    parts = (c.name or "").split()
    first, last = (parts[0] if parts else ""), " ".join(parts[1:])
    lines = ["BEGIN:VCARD", "VERSION:3.0",
             f"N:{_vcard_escape(last)};{_vcard_escape(first)};;;",
             f"FN:{_vcard_escape(c.name)}"]
    if _present(c.company_name):
        lines.append(f"ORG:{_vcard_escape(c.company_name)}")
    if c.job_title:
        lines.append(f"TITLE:{_vcard_escape(c.job_title)}")
    if _present(c.phone):
        lines.append(f"TEL;TYPE=WORK,VOICE:{_vcard_escape(c.phone)}")
    if _present(c.email):
        lines.append(f"EMAIL;TYPE=INTERNET:{_vcard_escape(c.email)}")
    if _present(c.linkedin):
        lines.append(f"URL:{_vcard_escape(c.linkedin)}")
    lines.append("END:VCARD")

    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=8, border=2)
    qr.add_data("\n".join(lines))
    qr.make(fit=True)
    buf = BytesIO()
    qr.make_image(image_factory=PilImage, fill_color="black", back_color="white").save(buf, format="PNG")
    return {
        "contactId": contact_id,
        "name": c.name,
        "vcard": "\n".join(lines),
        "qrDataUrl": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}",
    }


# ==================== AI COPILOT & ASSISTANT ====================

@api_router.get("/models", dependencies=[Depends(require_admin)])
async def get_available_models():
    """The AI models this server is configured to use (OpenRouter), and what each is for."""
    return _need(models_impl, "The model list")()


class ConversePayload(BaseModel):
    query: str = Field(..., max_length=4000)
    conversationHistory: Optional[List[Dict[str, str]]] = Field(None, max_length=40)
    # "openui": the answer may hold ```openui-lang blocks (contact cards, stats, charts, drafts…)
    uiFormat: Optional[str] = Field(None, pattern="^(openui|text)$")


# Injected by app.py: the real assistant over all users' contacts (the admin console's copilot).
converse_impl = None         # async (query, history, ui_format) -> run_converse() result
converse_stream_impl = None  # async (query, history, ui_format) -> StreamingResponse (SSE: meta · model · delta · done · error)


def _converse_contacts(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The contacts the answer drew on, so its contact cards can open the real contact."""
    out = []
    for c in result.get("retrieved_contacts") or []:
        cid = str(c.get("id") or "")
        if not cid:
            continue
        out.append({"id": cid, "name": _present(c.get("name")) or "", "company": _present(c.get("company_name")),
                    "score": c.get("lead_score"), "temperature": c.get("lead_temperature")})
    return out


def _converse_history(h: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    return [{"role": m.get("role", "user"), "content": str(m.get("content") or "")[:6000]}
            for m in (h or []) if m.get("role") in ("user", "assistant") and m.get("content")][-20:]


@api_router.post("/converse", dependencies=[Depends(require_admin)])
async def converse_ai(payload: ConversePayload):
    """Admin console copilot: the same AI assistant as the app, over the whole workspace."""
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Ask a question first")
    impl = _need(converse_impl, "The AI assistant")
    result = await impl(payload.query.strip(), _converse_history(payload.conversationHistory), payload.uiFormat)
    return {
        "reply": result["response"],
        "format": result.get("format", "text"),
        "modelUsed": result["model"],
        "contacts": _converse_contacts(result),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@api_router.post("/converse/stream", dependencies=[Depends(require_admin)])
async def converse_ai_stream(payload: ConversePayload):
    """The copilot, streamed as Server-Sent Events so the answer shows as it's written:
    meta {format, retrieved_contacts} · model {model} · delta {text} · done {…full answer} · error {detail}."""
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Ask a question first")
    impl = _need(converse_stream_impl, "The AI assistant")
    return await impl(payload.query.strip(), _converse_history(payload.conversationHistory), payload.uiFormat)


@api_router.post("/converse/stream/files", dependencies=[Depends(require_admin)])
async def converse_ai_stream_files(
    query: str = Form("", max_length=4000),
    conversationHistory: Optional[str] = Form(None),
    uiFormat: Optional[str] = Form(None, pattern="^(openui|text)$"),
    files: List[UploadFile] = File(...),
):
    """The copilot with attached files (images, PDFs, text/CSV; 3 files, 10 MB each), streamed like /converse/stream.
    conversationHistory is a JSON list of {role, content} (multipart can't carry a list)."""
    history: List[Dict[str, str]] = []
    if conversationHistory:
        try:
            parsed = json.loads(conversationHistory)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="conversationHistory must be a JSON list")
        if not isinstance(parsed, list):
            raise HTTPException(status_code=400, detail="conversationHistory must be a JSON list")
        history = [{"role": str(m.get("role", "")), "content": str(m.get("content") or "")} for m in parsed[-40:] if isinstance(m, dict)]
    impl = _need(converse_stream_impl, "The AI assistant")
    return await impl(query.strip(), _converse_history(history), uiFormat, files)


# Injected by app.py: voice recording → text (the same audio model and limits as the app's voice questions).
transcribe_impl = None  # async (UploadFile) -> str


@api_router.post("/transcribe", dependencies=[Depends(require_admin)])
async def transcribe_question(file: UploadFile = File(...)):
    """A spoken question (browser recording, usually audio/webm) → its text, for the assistant's voice input."""
    impl = _need(transcribe_impl, "Voice input")
    return {"transcript": await impl(file)}


# ==================== SYSTEM HEALTH, BACKUPS & WEBHOOK ====================

BACKUP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backups")
BACKUP_KEEP = int(os.environ.get("BACKUP_KEEP", "10"))
_last_backup: Dict[str, Any] = {}


def _sqlite_path() -> Optional[str]:
    """Path of the SQLite database file, or None when the primary database isn't SQLite."""
    from database import DATABASE_URL as _url
    if not _url or "sqlite" not in _url:
        return None
    path = _url.split("///", 1)[-1].split("?", 1)[0]
    return path if os.path.isabs(path) else os.path.join(os.getcwd(), path)


def _snapshots() -> List[Dict[str, Any]]:
    if not os.path.isdir(BACKUP_DIR):
        return []
    out = []
    for f in sorted(os.listdir(BACKUP_DIR), reverse=True):
        if f.endswith(".db"):
            full = os.path.join(BACKUP_DIR, f)
            out.append({"file": f, "sizeBytes": os.path.getsize(full),
                        "createdAt": datetime.fromtimestamp(os.path.getmtime(full), timezone.utc).isoformat()})
    return out


@api_router.post("/admin/backup", dependencies=[Depends(require_admin)])
async def trigger_admin_backup():
    """Back up the database now — to the backup database (BACKUP_DATABASE_URL) when configured,
    otherwise a consistent copy of the SQLite file into backend/backups/ (newest BACKUP_KEEP kept)."""
    global _last_backup
    started = datetime.now(timezone.utc)
    if backup_configured_impl and backup_configured_impl():
        result = await _need(backup_impl, "Database backup")()
        _last_backup = _db_backup_out(result, "console") or {"status": "error", "kind": "backup-database", "at": started.isoformat(), "message": "The backup failed."}
        if _last_backup["status"] == "error":
            raise HTTPException(status_code=502, detail=_last_backup["message"] or "The backup failed.")
        return _last_backup  # "partial": some tables failed — the message names them
    src = _sqlite_path()
    if not src or not os.path.exists(src):
        raise HTTPException(status_code=400, detail="No backup target: set BACKUP_DATABASE_URL (a second Postgres database) to enable backups.")
    import sqlite3
    os.makedirs(BACKUP_DIR, exist_ok=True)
    name = f"eventscout-{started.strftime('%Y%m%d-%H%M%S')}.db"
    dest = os.path.join(BACKUP_DIR, name)

    def _copy():
        s, d = sqlite3.connect(src), sqlite3.connect(dest)
        try:
            s.backup(d)  # consistent copy even while the app is writing
        finally:
            d.close()
            s.close()
    try:
        await asyncio.to_thread(_copy)
    except Exception as e:
        _last_backup = {"status": "error", "kind": "sqlite-file", "at": started.isoformat(), "message": str(e)}
        raise HTTPException(status_code=500, detail=f"The backup failed: {e}")
    for old in _snapshots()[BACKUP_KEEP:]:
        try:
            os.remove(os.path.join(BACKUP_DIR, old["file"]))
        except OSError:
            pass
    _last_backup = {"status": "success", "kind": "sqlite-file", "at": started.isoformat(), "file": name,
                    "sizeBytes": os.path.getsize(dest), "message": f"Saved {name}"}
    return _last_backup


def _db_backup_out(result: Optional[Dict[str, Any]], trigger: str) -> Optional[Dict[str, Any]]:
    """run_backup()'s result → the console's lastBackup shape (None when there's no result)."""
    if not result or not result.get("timestamp"):
        return None
    status = result.get("status")
    tables = result.get("tables") or {}
    return {
        "status": status if status in ("success", "partial") else "error",
        "kind": "backup-database",
        "trigger": trigger,
        "at": result["timestamp"],
        "message": result.get("summary") or result.get("message") or "",
        "tables": {k: (v if isinstance(v, int) else None) for k, v in tables.items()},
        "failedTables": result.get("failed_tables") or [k for k, v in tables.items() if not isinstance(v, int)],
        "totalRows": result.get("total_rows"),
        "durationSeconds": result.get("duration_seconds"),
    }


@api_router.get("/admin/backup/status", dependencies=[Depends(require_admin)])
async def get_backup_status():
    """How backups are set up, the last one (run from the console, at startup or on the schedule),
    and the snapshots on disk."""
    db_backup = bool(backup_configured_impl and backup_configured_impl())
    snaps = _snapshots()
    last = _last_backup or None
    if db_backup and last_backup_impl:
        # run_backup's own record covers the startup and periodic runs too; show whichever is newer
        auto = _db_backup_out(last_backup_impl(), "automatic")
        if auto and (not last or auto["at"] > last.get("at", "")):
            last = auto
    info = system_info_impl() if system_info_impl else {}
    return {
        "target": "backup-database" if db_backup else ("sqlite-file" if _sqlite_path() else None),
        "configured": db_backup or bool(_sqlite_path()),
        "lastBackup": last,
        "snapshots": snaps,
        "lastSnapshotAt": snaps[0]["createdAt"] if snaps else None,
        "keep": BACKUP_KEEP,
        "intervalHours": info.get("backupIntervalHours") if db_backup else None,
    }


_SNAPSHOT_NAME = re.compile(r"^eventscout-\d{8}-\d{6}\.db$")


@api_router.get("/admin/backup/snapshots/{file}", dependencies=[Depends(require_admin)])
async def download_backup_snapshot(file: str):
    """Download one SQLite snapshot from backend/backups/ (only files this server wrote)."""
    if not _SNAPSHOT_NAME.match(file):
        raise HTTPException(status_code=400, detail="Not a backup snapshot name")
    full = os.path.join(BACKUP_DIR, file)
    if not os.path.isfile(full):
        raise HTTPException(status_code=404, detail="That snapshot no longer exists")
    from fastapi.responses import FileResponse
    return FileResponse(full, media_type="application/vnd.sqlite3", filename=file)


@api_router.post("/admin/webhook/test", dependencies=[Depends(require_admin)])
async def test_webhook():
    """Send a real test event to the configured webhook (WEBHOOK_URL, e.g. n8n) and report what came back."""
    url = (webhook_url_impl() if webhook_url_impl else os.environ.get("WEBHOOK_URL", "")).strip()
    if not url:
        raise HTTPException(status_code=400, detail="No webhook is configured. Set WEBHOOK_URL on the server (e.g. your n8n webhook).")
    import httpx, time as _t
    payload = {"event": "eventscout.ping", "source": "EventScout admin console", "timestamp": datetime.now(timezone.utc).isoformat()}
    start = _t.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url, json=payload)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"The webhook didn't answer: {e.__class__.__name__}: {e}")
    ms = round((_t.perf_counter() - start) * 1000)
    host = urlparse(url).netloc
    return {"status": "delivered" if r.is_success else "rejected", "statusCode": r.status_code, "latencyMs": ms,
            "target": host, "response": r.text[:500]}


async def admin_export_contacts(session: AsyncSession, event: Optional[str] = None, userId: Optional[str] = None,
                                minScore: Optional[int] = None, listId: Optional[str] = None):
    """The contacts an admin export covers (every contact, or narrowed by event, rep, minimum score or
    list) \u2192 (contacts, filename parts, {user_id: owner name}). Shared by the CSV export and the CRM
    exports / push so they always pick the same rows. Scores are the admin console's own: the minimum
    score filters on it, it orders the rows, and each contact carries it as `console_score` (not a column —
    never saved); the rep's score stays `lead_score`."""
    query = _with_admin_score(select(ContactDB, ADMIN_SCORE))
    parts = []
    if userId:
        try:
            uid = uuid.UUID(userId)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid userId")
        query = query.where(ContactDB.user_id == uid)
        u = (await session.execute(select(UserDB.name).where(UserDB.id == uid))).scalar_one_or_none()
        if u is None:
            raise HTTPException(status_code=404, detail="That person no longer has an account")
        parts.append(u)
    if minScore is not None:
        query = query.where(ADMIN_SCORE >= minScore)
        parts.append("hot-warm" if minScore == WARM_MIN else f"score-{minScore}-plus")
    if listId:
        try:
            lid = uuid.UUID(listId)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid listId")
        lst = (await session.execute(select(ContactListDB).where(ContactListDB.id == lid))).scalar_one_or_none()
        if lst is None:
            raise HTTPException(status_code=404, detail="That list no longer exists")
        query = query.where(ContactDB.id.in_(select(ContactListMemberDB.contact_id).where(ContactListMemberDB.list_id == lid)))
        parts.append(f"list-{lst.name}")
    if event:
        query = query.where(await event_filter(session, ContactDB.event_name, event))
        parts.append(event)
    order = (ADMIN_SCORE.desc(), ContactDB.created_at.desc()) if minScore is not None else (ContactDB.created_at.desc(),)
    contacts = []
    for c, score in (await session.execute(query.order_by(*order))).all():
        c.console_score = score  # plain attribute, not mapped: read-only view for the export / CRM
        contacts.append(c)
    if event:  # spacing-insensitive, like everywhere else event names are compared
        contacts = [c for c in contacts if _key(c.event_name) == _key(event)]
    owners = {u.id: u.name for u in (await session.execute(select(UserDB))).scalars().all()}
    return contacts, parts, owners


def export_filename(parts: List[str], prefix: str = "eventscout", ext: str = "csv") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug = re.sub(r"[^\w-]+", "-", "-".join(parts).lower()).strip("-")[:80] or "all-contacts"
    return f"{prefix}-{slug}-{stamp}.{ext}"


@api_router.get("/admin/export", dependencies=[Depends(require_admin)])
async def export_all_contacts_csv(
    event: Optional[str] = Query(None, description="only contacts met at this event (name, case / spacing-insensitive)"),
    userId: Optional[str] = Query(None, description="only this rep's contacts"),
    minScore: Optional[int] = Query(None, ge=0, le=100, description="only contacts scoring at least this (e.g. 70 = hot & warm)"),
    listId: Optional[str] = Query(None, description="only the members of this contact list"),
    columns: Optional[str] = Query(None, description="comma-separated headers to include, in order (default: every column). See GET /api/admin/export-presets for the list"),
    session: AsyncSession = Depends(get_db),
):
    """Contacts as a CSV file (opens in Excel / Google Sheets): every contact, or narrowed by event,
    rep, minimum score or list. No row cap — the file always has every matching contact."""
    import csv
    try:
        headers = contact_view.parse_export_columns(columns)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    contacts, parts, owners = await admin_export_contacts(session, event, userId, minScore, listId)
    buf = io.StringIO()
    w = safe_csv_writer(buf)  # formula-injection safe (sanitize.py)
    w.writerow(headers)
    for c in contacts:
        owner = owners.get(c.user_id, "")
        w.writerow([contact_view.ADMIN_EXPORT_COLUMNS[h](c, owner) for h in headers])
    return Response(content="\ufeff" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{export_filename(parts)}"',
                             "X-Row-Count": str(len(contacts)), "Access-Control-Expose-Headers": "Content-Disposition, X-Row-Count"})


# Export presets: per admin, saved on their user row. The ChampHarbinger preset is built in and always first.
@api_router.get("/admin/export-presets")
async def list_export_presets(admin_id: uuid.UUID = Depends(require_admin), session: AsyncSession = Depends(get_db)):
    """Every column the export can carry, the default order, and this admin's presets (built-in first)."""
    saved = (await session.execute(select(UserDB.export_presets).where(UserDB.id == admin_id))).scalar_one_or_none() or []
    return {"columns": list(contact_view.ADMIN_EXPORT_COLUMNS), "default": list(contact_view.ADMIN_EXPORT_HEADER),
            "presets": [contact_view.HARBINGER_PRESET] + [dict(p, builtin=False) for p in saved]}


@api_router.put("/admin/export-presets")
async def save_export_presets(body: dict = Body(...), admin_id: uuid.UUID = Depends(require_admin),
                              session: AsyncSession = Depends(get_db)):
    """Replace this admin's saved presets: {"presets": [{name, columns, filters}]}. Built-in ones are not stored."""
    try:
        presets = contact_view.validate_presets([p for p in (body.get("presets") or []) if not (p or {}).get("builtin")])
    except (ValueError, AttributeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    await session.execute(update(UserDB).where(UserDB.id == admin_id).values(export_presets=presets))
    await session.commit()
    return await list_export_presets(admin_id, session)


# ==================== ADMIN · EVENT PLANNER ====================
# Events the admins plan and staff (Admin console → Announcements → Events): add an event with its
# full details, attach floor plans (event files, category "floorplan"), pick the team that goes, and
# announce it. The event name is the shared key with contacts, the booth directory, event files and
# reps' event lists, so assigning a rep adds the event to the event list in their app.

EVENT_STATUSES = ("suggested", "interested", "planned", "attending", "done", "dismissed")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DETAIL_KEYS = ("hours", "registration", "tickets", "booth", "dressCode", "contactName", "contactEmail", "contactPhone", "notes")


class PlannerEventPayload(BaseModel):
    name: Optional[str] = Field(None, max_length=255)
    status: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    venue: Optional[str] = Field(None, max_length=255)
    address: Optional[str] = Field(None, max_length=500)
    city: Optional[str] = Field(None, max_length=120)
    country: Optional[str] = Field(None, max_length=120)
    website: Optional[str] = Field(None, max_length=500)
    organizer: Optional[str] = Field(None, max_length=255)
    industry: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = Field(None, max_length=5000)
    details: Optional[Dict[str, Any]] = None


class PlannerTeamPayload(BaseModel):
    # the whole team after the change: [{"userId": "...", "role": "lead" | "member"}]
    members: List[Dict[str, str]] = Field(default_factory=list, max_length=500)


class PlannerAnnouncePayload(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    priority: Optional[str] = Field("normal", pattern="^(normal|urgent)$")
    audience: str = Field("team", pattern="^(team|everyone)$")


def _clean_date(v: Optional[str], label: str) -> Optional[str]:
    v = (v or "").strip()
    if not v:
        return None
    if not _DATE_RE.match(v):
        raise HTTPException(status_code=400, detail=f"{label} must be a date (YYYY-MM-DD)")
    try:
        datetime.strptime(v, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail=f"{label} isn't a real date")
    return v


def _clean_website(v: Optional[str]) -> str:
    v = (v or "").strip()
    if v and not re.match(r"^https?://", v, re.IGNORECASE):
        v = f"https://{v}"
    return v[:500]


async def _planner_rows(session: AsyncSession, only: Optional[uuid.UUID] = None) -> List[Dict[str, Any]]:
    q = select(ManagedEventDB)
    if only is not None:
        q = q.where(ManagedEventDB.id == only)
    events = (await session.execute(q)).scalars().all()
    if not events:
        return []
    ids = [e.id for e in events]
    users = {u.id: u for u in (await session.execute(select(UserDB))).scalars().all()}
    photos = {}
    for p in (await session.execute(select(UserProfileDB.user_id, UserProfileDB.profile_data))).all():
        photos[p.user_id] = _user_photo(p.profile_data)
    team: Dict[uuid.UUID, list] = {}
    for a in (await session.execute(select(EventAssignmentDB).where(EventAssignmentDB.event_id.in_(ids)))).scalars().all():
        u = users.get(a.user_id)
        if not u:
            continue  # the account was removed
        team.setdefault(a.event_id, []).append({
            "userId": str(u.id), "name": u.name, "email": u.email, "role": a.role,
            "isAdmin": bool(u.is_admin), "photoUrl": photos.get(u.id), "assignedAt": _iso(a.assigned_at),
        })
    keys = {_key(e.name) for e in events}
    files: Dict[str, list] = {}
    for f in (await session.execute(
            select(EventFileDB.id, EventFileDB.original_filename, EventFileDB.filename, EventFileDB.file_type, EventFileDB.category,
                   EventFileDB.description, EventFileDB.event_name, EventFileDB.created_at, func.length(EventFileDB.file_data).label("stored"))
            .where(EventFileDB.is_active == True))).all():  # noqa: E712
        k = _key(f.event_name)
        if k in keys and f.stored:
            files.setdefault(k, []).append({
                "id": str(f.id), "filename": f.original_filename or f.filename, "fileType": f.file_type, "fileSize": int(f.stored or 0),
                "category": f.category, "description": f.description or "", "createdAt": _iso(f.created_at),
                "floorPlan": _is_floor_plan(f.category, f.original_filename or f.filename, f.description),
            })
    met: Dict[str, int] = {}
    for name, n in (await session.execute(select(ContactDB.event_name, func.count(ContactDB.id)).group_by(ContactDB.event_name))).all():
        if _key(name) in keys:
            met[_key(name)] = met.get(_key(name), 0) + n
    out = []
    for e in events:
        members = sorted(team.get(e.id, []), key=lambda m: (m["role"] != "lead", m["name"].lower()))
        out.append({
            "id": str(e.id), "name": e.name, "status": e.status,
            "startDate": e.start_date, "endDate": e.end_date,
            "venue": e.venue or "", "address": e.address or "", "city": e.city or "", "country": e.country or "",
            "website": e.website or "", "organizer": e.organizer or "", "industry": e.industry or "",
            "description": e.description or "", "details": e.details or {},
            "source": e.source, "sourceUrl": e.source_url or "",
            "createdBy": users[e.created_by].name if e.created_by in users else None,
            "createdAt": _iso(e.created_at), "updatedAt": _iso(e.updated_at),
            "team": members, "files": files.get(_key(e.name), []), "contactsMet": met.get(_key(e.name), 0),
        })
    # upcoming first (soonest), then undated, then past (most recent first)
    today = datetime.now(timezone.utc).date().isoformat()
    out.sort(key=lambda r: (0, r["startDate"]) if (r["endDate"] or r["startDate"] or "") >= today and r["startDate"]
             else (1, r["name"].lower()) if not r["startDate"] else (2, "".join(chr(255 - ord(c)) for c in r["startDate"])))
    return out


async def _load_planner_event(session: AsyncSession, event_id: str) -> "ManagedEventDB":
    try:
        eid = uuid.UUID(event_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid event ID")
    e = (await session.execute(select(ManagedEventDB).where(ManagedEventDB.id == eid))).scalar_one_or_none()
    if not e:
        raise HTTPException(status_code=404, detail="Event not found")
    return e


def _apply_event_fields(e: "ManagedEventDB", p: PlannerEventPayload) -> None:
    if p.status is not None:
        if p.status not in EVENT_STATUSES:
            raise HTTPException(status_code=400, detail=f"Status must be one of: {', '.join(EVENT_STATUSES)}")
        e.status = p.status
    if p.startDate is not None:
        e.start_date = _clean_date(p.startDate, "Start date")
    if p.endDate is not None:
        e.end_date = _clean_date(p.endDate, "End date")
    if e.start_date and e.end_date and e.end_date < e.start_date:
        raise HTTPException(status_code=400, detail="The end date is before the start date")
    for attr, val, n in (("venue", p.venue, 255), ("address", p.address, 500), ("city", p.city, 120), ("country", p.country, 120),
                         ("organizer", p.organizer, 255), ("industry", p.industry, 255)):
        if val is not None:
            setattr(e, attr, _one_line(val, n))
    if p.website is not None:
        e.website = _clean_website(p.website)
    if p.description is not None:
        e.description = p.description.strip()
    if p.details is not None:
        e.details = {k: str(v).strip()[:2000] for k, v in p.details.items() if k in _DETAIL_KEYS and v is not None and str(v).strip()}


@api_router.get("/admin/planner/events", dependencies=[Depends(require_admin)])
async def planner_list_events(session: AsyncSession = Depends(get_db)):
    """Every planned / tracked event with its details, team, floor plans & files, and contacts met there."""
    return await _planner_rows(session)


@api_router.post("/admin/planner/events")
async def planner_create_event(payload: PlannerEventPayload, session: AsyncSession = Depends(get_db), admin_id: uuid.UUID = Depends(require_admin)):
    name = _one_line(payload.name, 255)
    if not name:
        raise HTTPException(status_code=400, detail="Give the event a name")
    for other in (await session.execute(select(ManagedEventDB.name))).scalars().all():
        if _key(other) == _key(name):
            raise HTTPException(status_code=409, detail=f"“{other}” is already in your events")
    e = ManagedEventDB(id=uuid.uuid4(), name=name, status="planned", source="manual", created_by=admin_id,
                       created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc), details={})
    _apply_event_fields(e, payload)
    session.add(e)
    await session.commit()
    await broadcast_live_event("event_planned", {"id": str(e.id), "name": e.name})
    return (await _planner_rows(session, e.id))[0]


@api_router.patch("/admin/planner/events/{event_id}", dependencies=[Depends(require_admin)])
async def planner_update_event(event_id: str, payload: PlannerEventPayload, session: AsyncSession = Depends(get_db)):
    """Update any detail. A rename also renames the event's files and the team's app event lists
    (contacts already met there keep the name they were captured with)."""
    e = await _load_planner_event(session, event_id)
    old = e.name
    if payload.name is not None:
        name = _one_line(payload.name, 255)
        if not name:
            raise HTTPException(status_code=400, detail="The event needs a name")
        if _key(name) != _key(old):
            for other in (await session.execute(select(ManagedEventDB.name).where(ManagedEventDB.id != e.id))).scalars().all():
                if _key(other) == _key(name):
                    raise HTTPException(status_code=409, detail=f"“{other}” is already in your events")
        if name != old:
            e.name = name
            # by name only — loading the file rows would pull every uploaded file's bytes into memory
            await session.execute(update(EventFileDB).where(
                await event_filter(session, EventFileDB.event_name, old)).values(event_name=name))
            # every rep's event list / active event (team or not), so no app is left on the old name
            for uid in (await session.execute(select(UserProfileDB.user_id))).scalars().all():
                await event_identity.set_profile_event(session, uid, old, add=False, rename_to=name)
            team_ids = (await session.execute(select(EventAssignmentDB.user_id).where(EventAssignmentDB.event_id == e.id))).scalars().all()
            for uid in team_ids:
                await event_identity.set_profile_event(session, uid, name, add=True)
            for n in (await session.execute(select(EventNoteDB))).scalars().all():  # reps' notes / reminders follow the event
                if _key(n.event_name) == _key(old):
                    n.event_name = name
            # booths, booth visits and contacts met there move with it (the events list groups by name)
            for x in (await session.execute(select(ExhibitorDB).where(await event_filter(session, ExhibitorDB.event_name, old)))).scalars().all():
                x.event_name = name
            if _key(old) != _key(name):
                taken = {(v.user_id, v.exhibitor_key) for v in (await session.execute(
                    select(BoothVisitDB).where(BoothVisitDB.event_key == _key(name)))).scalars().all()}
                for v in (await session.execute(select(BoothVisitDB).where(BoothVisitDB.event_key == _key(old)))).scalars().all():
                    if (v.user_id, v.exhibitor_key) in taken:
                        await session.delete(v)  # already visited under the new name
                    else:
                        v.event_key = _key(name)
            await session.execute(update(ContactDB).where(await event_filter(session, ContactDB.event_name, old))
                                  .values(event_name=name))
    _apply_event_fields(e, payload)
    e.updated_at = datetime.now(timezone.utc)
    await session.commit()
    await broadcast_live_event("event_planned", {"id": str(e.id), "name": e.name})
    return (await _planner_rows(session, e.id))[0]


@api_router.delete("/admin/planner/events/{event_id}", dependencies=[Depends(require_admin)])
async def planner_delete_event(event_id: str, session: AsyncSession = Depends(get_db)):
    """Remove the plan and its team (the event leaves the team's app lists). Contacts met there and
    the event's files stay — delete files on the Files page."""
    e = await _load_planner_event(session, event_id)
    for uid in (await session.execute(select(EventAssignmentDB.user_id).where(EventAssignmentDB.event_id == e.id))).scalars().all():
        await event_identity.set_profile_event(session, uid, e.name, add=False)
    await session.execute(delete(EventAssignmentDB).where(EventAssignmentDB.event_id == e.id))
    await session.delete(e)
    await session.commit()
    await broadcast_live_event("event_planned", {"id": event_id, "deleted": True})
    return {"status": "deleted", "id": event_id}


@api_router.put("/admin/planner/events/{event_id}/team")
async def planner_set_team(event_id: str, payload: PlannerTeamPayload, session: AsyncSession = Depends(get_db), admin_id: uuid.UUID = Depends(require_admin)):
    """Set who goes (the whole team, with at most one lead). New members get the event in their app's
    event list; people taken off lose it there."""
    e = await _load_planner_event(session, event_id)
    wanted: Dict[uuid.UUID, str] = {}
    for m in payload.members:
        try:
            uid = uuid.UUID(str(m.get("userId", "")))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid team member ID")
        role = m.get("role") or "member"
        if role not in ("lead", "member"):
            raise HTTPException(status_code=400, detail="Role must be lead or member")
        wanted[uid] = role
    if sum(1 for r in wanted.values() if r == "lead") > 1:
        raise HTTPException(status_code=400, detail="Pick one team lead")
    found = set((await session.execute(select(UserDB.id).where(UserDB.id.in_(list(wanted))))).scalars().all()) if wanted else set()
    if set(wanted) - found:
        raise HTTPException(status_code=400, detail="Someone on the team no longer has an account — refresh and try again")
    current = {a.user_id: a for a in (await session.execute(select(EventAssignmentDB).where(EventAssignmentDB.event_id == e.id))).scalars().all()}
    for uid, a in current.items():
        if uid not in wanted:
            await session.delete(a)
            await event_identity.set_profile_event(session, uid, e.name, add=False)
    now = datetime.now(timezone.utc)
    for uid, role in wanted.items():
        if uid in current:
            current[uid].role = role
        else:
            session.add(EventAssignmentDB(id=uuid.uuid4(), event_id=e.id, user_id=uid, role=role, assigned_by=admin_id, assigned_at=now))
            await event_identity.set_profile_event(session, uid, e.name, add=True)
    e.updated_at = now
    await session.commit()
    await broadcast_live_event("event_planned", {"id": str(e.id), "name": e.name, "team": len(wanted)})
    return (await _planner_rows(session, e.id))[0]


@api_router.post("/admin/planner/events/{event_id}/announce")
async def planner_announce(event_id: str, payload: PlannerAnnouncePayload, session: AsyncSession = Depends(get_db), admin_id: uuid.UUID = Depends(require_admin)):
    """Send the event as a team announcement — to the event's team, or to everyone."""
    e = await _load_planner_event(session, event_id)
    message = payload.message.strip()
    recipients = None
    if payload.audience == "team":
        recipients = [str(uid) for uid in (await session.execute(select(EventAssignmentDB.user_id).where(EventAssignmentDB.event_id == e.id))).scalars().all()]
        if not recipients:
            raise HTTPException(status_code=400, detail="This event has no team yet — add people first, or send it to everyone")
    bid = uuid.uuid4()
    session.add(AdminBroadcastDB(id=bid, admin_id=admin_id, message=message, priority=payload.priority or "normal",
                                 recipient_ids=recipients, event_id=e.id, is_active=True, created_at=datetime.now(timezone.utc)))
    await session.commit()
    await broadcast_live_event("broadcast_alert", {"id": str(bid), "message": message, "priority": payload.priority or "normal",
                                                   "recipients": len(recipients) if recipients else "everyone", "event": e.name,
                                                   "timestamp": datetime.now(timezone.utc).isoformat()})
    return {"id": str(bid), "status": "dispatched", "recipients": len(recipients) if recipients else None}


class PlannerExtractPayload(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


@api_router.post("/admin/planner/extract", dependencies=[Depends(require_admin)])
async def planner_extract(payload: PlannerExtractPayload):
    """Read an event's details out of free text (an announcement being written) with the AI, so it
    can be saved as an event without typing it twice. Nothing is saved here and nothing is invented:
    fields the text doesn't state come back empty for the admin to fill in."""
    import ai_client
    if not ai_client.ai_configured:
        raise HTTPException(status_code=503, detail="The AI isn't set up on the server (OPENROUTER_API_KEY), so details can't be read automatically — fill the form in by hand.")
    today = datetime.now(timezone.utc).date().isoformat()
    prompt = (
        "Extract the event described in this team announcement. Use ONLY facts stated in the text; "
        "leave a field as an empty string when the text doesn't say it. Dates as YYYY-MM-DD "
        f"(today is {today}; resolve dates without a year to the next occurrence). "
        "Reply with JSON only:\n"
        '{"name": "", "startDate": "", "endDate": "", "venue": "", "address": "", "city": "", "country": "", '
        '"website": "", "organizer": "", "industry": "", "description": "", '
        '"details": {"booth": "", "hours": "", "registration": "", "tickets": "", "dressCode": "", '
        '"contactName": "", "contactEmail": "", "contactPhone": "", "notes": ""}}\n\n'
        f"Announcement:\n{payload.text.strip()}"
    )
    try:
        text, _ = await ai_client.chat([{"role": "user", "content": prompt}], max_tokens=700, temperature=0.0, json_mode=True,
                                       timeout=40.0, title="Event Scout Event Extract")
        data = ai_client.parse_json(text)
    except ai_client.AIError as e:
        raise HTTPException(status_code=503, detail=f"The AI couldn't read it right now ({e.reason}) — fill the form in by hand.")
    except (ValueError, TypeError):
        raise HTTPException(status_code=502, detail="The AI's answer wasn't usable — fill the form in by hand.")
    s = lambda v, n=500: _one_line(v, n) if isinstance(v, (str, int, float)) else ""
    date = lambda v: s(v, 10) if _DATE_RE.match(s(v, 10)) else ""
    det = data.get("details") if isinstance(data.get("details"), dict) else {}
    out = {
        "name": s(data.get("name"), 255), "startDate": date(data.get("startDate")), "endDate": date(data.get("endDate")),
        "venue": s(data.get("venue"), 255), "address": s(data.get("address")), "city": s(data.get("city"), 120),
        "country": s(data.get("country"), 120), "website": _clean_website(s(data.get("website"))) if s(data.get("website")) else "",
        "organizer": s(data.get("organizer"), 255), "industry": s(data.get("industry"), 255),
        "description": str(data.get("description") or "").strip()[:2000],
        "details": {k: str(det.get(k)).strip()[:500] for k in _DETAIL_KEYS if det.get(k) and str(det.get(k)).strip()},
    }
    # the website only if it's literally in the text (never a guessed URL)
    if out["website"] and out["website"].replace("https://", "").replace("http://", "").split("/")[0].lower() not in payload.text.lower():
        out["website"] = ""
    return out


@api_router.get("/admin/planner/discovery", dependencies=[Depends(require_admin)])
async def planner_discovery_status():
    """Whether automatic event discovery (a web scraper / search service) is connected. Until one is,
    nothing is suggested — events are added by hand."""
    return {"connected": False, "provider": None,
            "message": "Event discovery isn't connected yet. Once a web scraper or search service is set up, suggested events show up here."}


# ==================== MY EVENTS (app: event details · calendar · notes & reminders) ====================
# The planner's event details (dates, venue, address, booth, hours…) for the events a rep is on the
# team for, or has in their own event list. Read-only for reps; admins edit them in the console.

@api_router.get("/me/events")
async def my_planned_events(scope: Optional[str] = Query(None), session: AsyncSession = Depends(get_db),
                            me: Optional[uuid.UUID] = Depends(request_user_id)):
    """Default: events you're on the team for or have in your list. `scope=all`: every event the
    admins have scheduled (not just suggested / dismissed ones), with `myRole` null where you're not on it."""
    user = await _mobile_user(me)
    prof = (await session.execute(select(UserProfileDB.profile_data).where(UserProfileDB.user_id == user))).scalar_one_or_none()
    listed = {_key(e) for e in (_profile_events(prof)[0] or [])}
    mine = {a.event_id: a.role for a in (await session.execute(
        select(EventAssignmentDB).where(EventAssignmentDB.user_id == user))).scalars().all()}
    everyone = scope == "all"
    events = [e for e in (await session.execute(select(ManagedEventDB))).scalars().all()
              if e.id in mine or ((everyone or _key(e.name) in listed) and e.status not in ("suggested", "dismissed"))]
    if not events:
        return []
    rows = {r["id"]: r for r in await _planner_rows(session)}
    out = []
    for e in events:
        r = rows.get(str(e.id))
        if not r:
            continue
        lead = next((m["name"] for m in r["team"] if m["role"] == "lead"), None)
        out.append({
            **{k: r[k] for k in ("id", "name", "status", "startDate", "endDate", "venue", "address", "city", "country",
                                 "website", "organizer", "industry", "description", "details", "updatedAt")},
            "myRole": mine.get(e.id),  # "lead" | "member" | None (in your list, not on the team)
            "lead": lead,
            "team": [{"name": m["name"], "role": m["role"]} for m in r["team"]],
            "floorPlans": sum(1 for f in r["files"] if f["floorPlan"]),
        })
    order = {r["id"]: i for i, r in enumerate(rows.values())}  # the planner's order: upcoming first
    out.sort(key=lambda r: order.get(r["id"], 0))
    return out


_NOTE_TEXT_MAX = 2000


class EventNotePayload(BaseModel):
    event: Optional[str] = Field(None, max_length=255)
    date: Optional[str] = None       # YYYY-MM-DD, "" clears
    text: Optional[str] = Field(None, max_length=_NOTE_TEXT_MAX)
    remindAt: Optional[str] = None   # ISO date-time with offset, "" clears
    done: Optional[bool] = None


def _note_out(n: "EventNoteDB") -> Dict[str, Any]:
    return {"id": str(n.id), "event": n.event_name, "date": n.date, "text": n.text, "remindAt": _iso(n.remind_at),
            "done": bool(n.done), "createdAt": _iso(n.created_at), "updatedAt": _iso(n.updated_at)}


def _parse_remind_at(v: str) -> Optional[datetime]:
    v = (v or "").strip()
    if not v:
        return None
    try:
        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail="The reminder time isn't a valid date and time")
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _apply_note_fields(n: "EventNoteDB", p: EventNotePayload) -> None:
    if p.event is not None:
        name = _one_line(p.event, 255)
        if not name:
            raise HTTPException(status_code=400, detail="Pick the event this note is for")
        n.event_name = name
    if p.date is not None:
        n.date = _clean_date(p.date, "Date")
    if p.text is not None:
        n.text = p.text.strip()
    if p.remindAt is not None:
        n.remind_at = _parse_remind_at(p.remindAt)
    if p.done is not None:
        n.done = p.done


async def _load_note(session: AsyncSession, me: uuid.UUID, note_id: str) -> "EventNoteDB":
    try:
        nid = uuid.UUID(note_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid note ID")
    n = (await session.execute(select(EventNoteDB).where(EventNoteDB.id == nid, EventNoteDB.user_id == me))).scalar_one_or_none()
    if not n:
        raise HTTPException(status_code=404, detail="Note not found")
    return n


@api_router.get("/me/event-notes")
async def list_event_notes(event: Optional[str] = None, session: AsyncSession = Depends(get_db),
                           me: Optional[uuid.UUID] = Depends(request_user_id)):
    """Your notes & reminders — all of them, or one event's with `event`."""
    user = await _mobile_user(me)
    notes = (await session.execute(select(EventNoteDB).where(EventNoteDB.user_id == user))).scalars().all()
    if event:
        notes = [n for n in notes if _key(n.event_name) == _key(event)]
    notes.sort(key=lambda n: (n.date or "9999-99-99", _iso(n.remind_at) or "", _iso(n.created_at) or ""))
    return [_note_out(n) for n in notes]


@api_router.post("/me/event-notes")
async def create_event_note(payload: EventNotePayload, session: AsyncSession = Depends(get_db),
                            me: Optional[uuid.UUID] = Depends(request_user_id)):
    user = await _mobile_user(me)
    if not (payload.text or "").strip():
        raise HTTPException(status_code=400, detail="Write the note first")
    if not (payload.event or "").strip():
        raise HTTPException(status_code=400, detail="Pick the event this note is for")
    now = datetime.now(timezone.utc)
    n = EventNoteDB(id=uuid.uuid4(), user_id=user, done=False, created_at=now, updated_at=now)
    _apply_note_fields(n, payload)
    session.add(n)
    await session.commit()
    return _note_out(n)


@api_router.patch("/me/event-notes/{note_id}")
async def update_event_note(note_id: str, payload: EventNotePayload, session: AsyncSession = Depends(get_db),
                            me: Optional[uuid.UUID] = Depends(request_user_id)):
    user = await _mobile_user(me)
    n = await _load_note(session, user, note_id)
    _apply_note_fields(n, payload)
    if not n.text:
        raise HTTPException(status_code=400, detail="A note can't be empty — delete it instead")
    n.updated_at = datetime.now(timezone.utc)
    await session.commit()
    return _note_out(n)


@api_router.delete("/me/event-notes/{note_id}")
async def delete_event_note(note_id: str, session: AsyncSession = Depends(get_db),
                            me: Optional[uuid.UUID] = Depends(request_user_id)):
    user = await _mobile_user(me)
    n = await _load_note(session, user, note_id)
    await session.delete(n)
    await session.commit()
    return {"status": "deleted", "id": note_id}


# ==================== EMAIL VERIFICATION (check-if-email-exists) ====================
# Is this address real? Syntax · domain takes mail (MX) · disposable / role / free provider · the
# mail server's answer for the mailbox (RCPT TO, nothing is sent) · catch-all. See email_verify.py.
# Results are cached per address; rechecking asks the mail server again.

_EMAIL_CACHE_DAYS = 7
_EMAIL_RATE: Dict[str, List[float]] = {}   # who → recent mail-server checks (cached answers are free)
_EMAIL_RATE_MAX = 30                       # per 10 minutes, so this server doesn't get flagged for probing


class EmailVerifyPayload(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)
    force: bool = False   # ignore the cached answer
    smtp: bool = True     # False: syntax + domain checks only (instant)


class EmailStatusPayload(BaseModel):
    emails: List[str] = Field(default_factory=list, max_length=1000)


def _email_key(e: str) -> str:
    return (e or "").strip().lower()


def _email_out(row: "EmailVerificationDB", cached: bool) -> Dict[str, Any]:
    return {**(row.result or {}), "is_reachable": row.is_reachable, "checked_at": _iso(row.checked_at), "cached": cached}


@api_router.post("/email/verify")
async def verify_email_address(payload: EmailVerifyPayload, session: AsyncSession = Depends(get_db),
                               me: Optional[uuid.UUID] = Depends(request_user_id)):
    """Check one address (cached for a week unless `force`)."""
    import time
    import email_verify
    key = _email_key(payload.email)
    if not key or key in ("n/a", "na", "none"):
        raise HTTPException(status_code=400, detail="Enter an email address to check")
    row = await session.get(EmailVerificationDB, key)
    fresh = row and row.checked_at and (datetime.now(timezone.utc) - (row.checked_at if row.checked_at.tzinfo else row.checked_at.replace(tzinfo=timezone.utc))).days < _EMAIL_CACHE_DAYS
    # a cached full check answers both kinds of request; a cached quick check only a quick one
    if row and fresh and not payload.force and ((row.result or {}).get("smtp_checked") or not payload.smtp):
        return _email_out(row, cached=True)
    if payload.smtp:
        who = str(me) if me else "admin"
        now = time.time()
        recent = [t for t in _EMAIL_RATE.get(who, []) if now - t < 600]
        if len(recent) >= _EMAIL_RATE_MAX:
            raise HTTPException(status_code=429, detail="Too many email checks in a short time — try again in a few minutes")
        _EMAIL_RATE[who] = recent + [now]
    result = await email_verify.verify_email(payload.email, smtp=payload.smtp)
    if not row:
        row = EmailVerificationDB(email=key)
        session.add(row)
    # don't let a quick check overwrite a full one that's still fresh
    if payload.smtp or not (fresh and (row.result or {}).get("smtp_checked")):
        row.is_reachable = result["is_reachable"]
        row.result = result
        row.checked_at = datetime.now(timezone.utc)
        await session.commit()
    return {**result, "cached": False}


@api_router.post("/email/status", dependencies=[Depends(request_user_id)])
async def email_status(payload: EmailStatusPayload, session: AsyncSession = Depends(get_db)):
    """Cached results for many addresses at once (no new checks) — {email: {is_reachable, checked_at}}."""
    keys = list({_email_key(e) for e in payload.emails if _email_key(e)})
    if not keys:
        return {}
    rows = (await session.execute(select(EmailVerificationDB).where(EmailVerificationDB.email.in_(keys)))).scalars().all()
    return {r.email: {"is_reachable": r.is_reachable, "checked_at": _iso(r.checked_at)} for r in rows}
