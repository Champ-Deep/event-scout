"""Admin dashboard API (docs/admin-dashboard-spec.md section 5).

Router factory so app.py only needs an import and one include_router line.
`core` is the app module: it supplies verify_admin, faiss_index and the
WEBHOOK_URL / AUTO_PIPELINE_ENABLED globals this module reads and writes.
"""

import csv
import io
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, StrictBool
from sqlalchemy import Date, case, cast, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import aliased

from database import AppSettingDB, ContactDB, ContactPipelineDB, UserDB, get_session_factory

EXPORT_MAX_ROWS = 20000
EXPORT_COLUMNS = ["Name", "Company", "Email", "Phone", "Rep", "Team lead", "Event", "Score", "Temperature", "Source", "Created"]

Owner = aliased(UserDB)  # rep who owns the contact
Lead = aliased(UserDB)   # that rep's team lead


class TeamLeadBody(BaseModel):
    team_lead_id: Optional[uuid.UUID]  # required key, null clears


class ReassignBody(BaseModel):
    to_user_id: uuid.UUID


class SettingsBody(BaseModel):
    webhook_url: Optional[str] = None
    auto_pipeline_enabled: Optional[StrictBool] = None


def _uuid(value: str, what: str = "id") -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid {what}")


def _clean(v):
    """Legacy rows store 'N/A' for blanks; the dashboard shows blanks as empty."""
    return "" if v is None or v == "N/A" else v


def _csv_cell(v) -> str:
    # spreadsheet formula injection guard
    s = "" if v is None else str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s


def _lead_conditions(q, team_lead, rep, temperature, event, date_filter):
    """Single WHERE builder shared by the list and the export so they cannot diverge."""
    c = []
    if q:
        esc = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pat = f"%{esc}%"
        c.append(or_(*[col.ilike(pat, escape="\\") for col in
                       (ContactDB.name, ContactDB.email, ContactDB.company_name, ContactDB.notes)]))
    if team_lead == "none":
        c.append(Owner.team_lead_id.is_(None))
    elif team_lead:
        tl = _uuid(team_lead, "team_lead")
        c.append(or_(Owner.team_lead_id == tl, Owner.id == tl))
    if rep:
        c.append(Owner.id == _uuid(rep, "rep"))
    if temperature == "unscored":
        c.append(ContactDB.lead_temperature.is_(None))
    elif temperature:
        c.append(ContactDB.lead_temperature == temperature)
    if event:
        c.append(ContactDB.event_name == event)
    now = datetime.now(timezone.utc)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if date_filter == "today":
        c.append(ContactDB.created_at >= today)
    elif date_filter == "yesterday":
        c += [ContactDB.created_at >= today - timedelta(days=1), ContactDB.created_at < today]
    elif date_filter == "last_week":
        c.append(ContactDB.created_at >= now - timedelta(days=7))
    elif date_filter == "last_month":
        c.append(ContactDB.created_at >= now - timedelta(days=30))
    return c


_SORTS = {
    "created_desc": lambda: [ContactDB.created_at.desc(), ContactDB.id],
    "score_desc": lambda: [ContactDB.lead_score.desc().nulls_last(), ContactDB.created_at.desc(), ContactDB.id],
    "name_asc": lambda: [func.lower(ContactDB.name), ContactDB.id],
}


def _leads_select(conditions, sort):
    return (
        select(ContactDB, Owner.name.label("rep_name"), Owner.team_lead_id.label("tl_id"), Lead.name.label("tl_name"))
        .join(Owner, Owner.id == ContactDB.user_id)
        .outerjoin(Lead, Lead.id == Owner.team_lead_id)
        .where(*conditions)
        .order_by(*_SORTS[sort]())
    )


def _lead_row(row):
    c = row.ContactDB
    return {
        "id": str(c.id), "user_id": str(c.user_id), "user_name": row.rep_name,
        "team_lead_id": str(row.tl_id) if row.tl_id else None, "team_lead_name": row.tl_name,
        "name": c.name, "company_name": _clean(c.company_name), "email": _clean(c.email),
        "phone": _clean(c.phone), "event_name": c.event_name or "",
        "lead_score": c.lead_score, "lead_temperature": c.lead_temperature,
        "source": c.source or "manual", "created_at": c.created_at.isoformat() if c.created_at else "",
    }


async def _rep_rows(session, user_id=None):
    """One grouped query for rep stats (5.2); user_id narrows it to one rep."""
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    temp = lambda t: func.coalesce(func.sum(case((ContactDB.lead_temperature == t, 1), else_=0)), 0)
    stmt = (
        select(
            Owner.id, Owner.name, Owner.email, Owner.is_admin, Owner.team_lead_id, Lead.name.label("tl_name"),
            func.count(ContactDB.id), temp("hot"), temp("warm"), temp("cold"),
            func.coalesce(func.sum(case((ContactDB.created_at >= start, 1), else_=0)), 0),
            func.max(ContactDB.created_at),
        )
        .select_from(Owner)
        .outerjoin(ContactDB, ContactDB.user_id == Owner.id)
        .outerjoin(Lead, Lead.id == Owner.team_lead_id)
        .group_by(Owner.id, Lead.name)
        .order_by(Owner.name, Owner.id)
    )
    if user_id:
        stmt = stmt.where(Owner.id == user_id)
    return [
        {
            "user_id": str(r[0]), "name": r[1], "email": r[2], "is_admin": r[3],
            "team_lead_id": str(r[4]) if r[4] else None, "team_lead_name": r[5],
            "contact_count": r[6], "hot_count": int(r[7]), "warm_count": int(r[8]), "cold_count": int(r[9]),
            "today_count": int(r[10]), "last_scan_at": r[11].isoformat() if r[11] else None,
        }
        for r in (await session.execute(stmt)).all()
    ]


async def _faiss_rebuild(core, session, user_id: uuid.UUID):
    # mirrors the per-user builder in app.py startup_event (inline there, so copied here)
    rows = (await session.execute(select(ContactDB).where(ContactDB.user_id == user_id))).scalars().all()
    core.faiss_index.build_for_user(str(user_id), [
        {
            "id": str(c.id), "name": c.name, "email": c.email or "N/A", "phone": c.phone or "N/A",
            "linkedin": c.linkedin or "N/A", "company_name": c.company_name or "N/A", "notes": c.notes or "",
            "links": c.links or [], "source": c.source or "manual", "linkedin_source": c.linkedin_source or "card",
            "lead_score": c.lead_score, "lead_temperature": c.lead_temperature,
            "lead_score_reasoning": c.lead_score_reasoning or "",
            "lead_score_breakdown": c.lead_score_breakdown or {},
            "lead_recommended_actions": c.lead_recommended_actions or [],
            "created_at": c.created_at.isoformat() if c.created_at else "",
        }
        for c in rows
    ])


def _settings_view(core, can_write):
    return {
        "status": "success", "can_write": can_write,
        "settings": {"webhook_url": core.WEBHOOK_URL, "auto_pipeline_enabled": core.AUTO_PIPELINE_ENABLED},
        "info": {
            "gamma_configured": bool(core.GAMMA_API_KEY),
            "openrouter_configured": bool(core.OPENROUTER_API_KEY),
            "backup_configured": bool(core.ASYNC_BACKUP_URL),
        },
    }


async def load_settings(core):
    """Startup: persisted settings override the env defaults. Never blocks boot."""
    try:
        async with get_session_factory()() as session:
            for row in (await session.execute(select(AppSettingDB))).scalars().all():
                if row.key == "webhook_url" and isinstance(row.value, str):
                    core.WEBHOOK_URL = row.value
                elif row.key == "auto_pipeline_enabled" and isinstance(row.value, bool):
                    core.AUTO_PIPELINE_ENABLED = row.value
        print(f"[SETTINGS] Loaded: webhook {'set' if core.WEBHOOK_URL else 'not set'}, auto_pipeline={core.AUTO_PIPELINE_ENABLED}")
    except Exception as e:
        print(f"[SETTINGS] Load failed, keeping env defaults: {e}")


async def _session():
    factory = get_session_factory()
    if factory is None:
        raise HTTPException(status_code=503, detail="Database not available")
    async with factory() as session:
        yield session


def build_router(core) -> APIRouter:
    r = APIRouter()
    admin = Depends(core.verify_admin)
    Q_TEMP = Query(None, pattern="^(hot|warm|cold|unscored)$")
    Q_DATE = Query(None, pattern="^(today|yesterday|last_week|last_month)$")
    Q_SORT = Query("created_desc", pattern="^(created_desc|score_desc|name_asc)$")

    @r.get("/admin/leads")
    async def list_leads(
        admin_id: str = admin, session=Depends(_session),
        q: Optional[str] = Query(None, min_length=1), team_lead: Optional[str] = None, rep: Optional[str] = None,
        temperature: Optional[str] = Q_TEMP, event: Optional[str] = None, date_filter: Optional[str] = Q_DATE,
        sort: str = Q_SORT, page: int = Query(1, ge=1), limit: int = Query(50, ge=1, le=200),
    ):
        cond = _lead_conditions(q, team_lead, rep, temperature, event, date_filter)
        total = (await session.execute(
            select(func.count()).select_from(ContactDB).join(Owner, Owner.id == ContactDB.user_id).where(*cond)
        )).scalar() or 0
        offset = (page - 1) * limit
        rows = (await session.execute(_leads_select(cond, sort).limit(limit).offset(offset))).all()
        return {"status": "success", "total": total, "page": page, "limit": limit,
                "has_more": offset + len(rows) < total, "contacts": [_lead_row(x) for x in rows]}

    # declared before /admin/leads/{contact_id} so "export" is not read as an id
    @r.get("/admin/leads/export")
    async def export_leads(
        admin_id: str = admin, session=Depends(_session),
        q: Optional[str] = Query(None, min_length=1), team_lead: Optional[str] = None, rep: Optional[str] = None,
        temperature: Optional[str] = Q_TEMP, event: Optional[str] = None, date_filter: Optional[str] = Q_DATE,
        sort: str = Q_SORT,
    ):
        cond = _lead_conditions(q, team_lead, rep, temperature, event, date_filter)
        total = (await session.execute(
            select(func.count()).select_from(ContactDB).join(Owner, Owner.id == ContactDB.user_id).where(*cond)
        )).scalar() or 0
        if total > EXPORT_MAX_ROWS:
            raise HTTPException(status_code=422, detail="Too many rows, narrow the filter")
        rows = (await session.execute(_leads_select(cond, sort))).all()
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(EXPORT_COLUMNS)
        for x in rows:
            d = _lead_row(x)
            w.writerow([_csv_cell(v) for v in (
                d["name"], d["company_name"], d["email"], d["phone"], d["user_name"], d["team_lead_name"],
                d["event_name"], d["lead_score"], d["lead_temperature"], d["source"], d["created_at"])])
        return Response(
            content=("\ufeff" + buf.getvalue()).encode("utf-8"), media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=event_scout_leads_{datetime.now(timezone.utc):%Y%m%d}.csv",
                     "X-Total-Count": str(len(rows))},
        )

    @r.get("/admin/leads/{contact_id}")
    async def get_lead(contact_id: str, admin_id: str = admin, session=Depends(_session)):
        cid = _uuid(contact_id, "contact_id")
        row = (await session.execute(_leads_select([ContactDB.id == cid], "created_desc"))).first()
        if not row:
            raise HTTPException(status_code=404, detail="Contact not found")
        c = row.ContactDB
        d = _lead_row(row)
        d.update({
            "linkedin": _clean(c.linkedin), "linkedin_source": c.linkedin_source or "card",
            "lead_score_reasoning": c.lead_score_reasoning or "",
            "lead_score_breakdown": c.lead_score_breakdown or {},
            "lead_recommended_actions": c.lead_recommended_actions or [],
            "notes": c.notes or "", "admin_notes": c.admin_notes or "", "links": c.links or [],
            "audio_notes_count": len(c.audio_notes) if c.audio_notes else 0,
            "photo_base64": c.photo_base64,
            "updated_at": c.updated_at.isoformat() if c.updated_at else "",
        })
        return {"status": "success", "contact": d}

    @r.post("/admin/leads/{contact_id}/reassign")
    async def reassign_lead(contact_id: str, body: ReassignBody, admin_id: str = admin, session=Depends(_session)):
        cid = _uuid(contact_id, "contact_id")
        contact = await session.get(ContactDB, cid)
        if not contact:
            raise HTTPException(status_code=404, detail="Contact not found")
        target = await session.get(UserDB, body.to_user_id)
        if not target:
            raise HTTPException(status_code=404, detail="Target rep not found")
        if contact.user_id == target.id:
            raise HTTPException(status_code=400, detail="Already owned by this rep")
        source = await session.get(UserDB, contact.user_id)
        from_id = contact.user_id
        try:
            await session.execute(update(ContactDB).where(ContactDB.id == cid).values(user_id=target.id))
            await session.execute(update(ContactPipelineDB).where(ContactPipelineDB.contact_id == cid).values(user_id=target.id))
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        print(f"[REASSIGN] admin={admin_id} contact={cid} from={from_id} to={target.id}")
        try:
            for uid in (from_id, target.id):
                await _faiss_rebuild(core, session, uid)
            synced = True
        except Exception as e:
            print(f"[REASSIGN] FAISS rebuild failed (DB is source of truth, startup rebuilds): {e}")
            synced = False
        return {"status": "success", "contact_id": str(cid), "from_user_id": str(from_id),
                "from_name": source.name if source else "", "to_user_id": str(target.id),
                "to_name": target.name, "faiss_synced": synced}

    @r.get("/admin/reps")
    async def list_reps(admin_id: str = admin, session=Depends(_session)):
        reps = await _rep_rows(session)
        names = {x["user_id"]: x["name"] for x in reps}
        counts = {}
        for x in reps:
            if x["team_lead_id"]:
                counts[x["team_lead_id"]] = counts.get(x["team_lead_id"], 0) + 1
        leads = sorted(({"user_id": k, "name": names.get(k, ""), "rep_count": v} for k, v in counts.items()),
                       key=lambda t: (t["name"], t["user_id"]))
        return {"status": "success", "reps": reps, "team_leads": leads}

    @r.get("/admin/reps/{rep_id}/performance")
    async def rep_performance(rep_id: str, days: int = Query(14, ge=1, le=90), admin_id: str = admin,
                              session=Depends(_session)):
        uid = _uuid(rep_id, "user_id")
        found = await _rep_rows(session, uid)
        if not found:
            raise HTTPException(status_code=404, detail="User not found")
        today = datetime.now(timezone.utc).date()
        first = today - timedelta(days=days - 1)
        day = cast(func.timezone("UTC", ContactDB.created_at), Date)
        by_day = dict((await session.execute(
            select(day, func.count()).where(ContactDB.user_id == uid, day >= first).group_by(day))).all())
        avg = (await session.execute(select(func.avg(ContactDB.lead_score)).where(ContactDB.user_id == uid))).scalar()
        events = (await session.execute(
            select(ContactDB.event_name, func.count().label("n"))
            .where(ContactDB.user_id == uid, ContactDB.event_name.is_not(None), ContactDB.event_name != "")
            .group_by(ContactDB.event_name).order_by(func.count().desc(), ContactDB.event_name).limit(5))).all()
        return {
            "status": "success", "rep": found[0],
            "avg_score": round(float(avg), 1) if avg is not None else None,
            "daily": [{"date": (first + timedelta(days=i)).isoformat(), "count": by_day.get(first + timedelta(days=i), 0)}
                      for i in range(days)],
            "by_event": [{"event_name": e, "count": n} for e, n in events],
        }

    @r.put("/admin/users/{target_id}/team_lead")  # not {user_id}: verify_admin owns the user_id query param
    async def set_team_lead(target_id: str, body: TeamLeadBody, admin_id: str = admin, session=Depends(_session)):
        uid = _uuid(target_id, "user_id")
        if body.team_lead_id == uid:
            raise HTTPException(status_code=400, detail="A user cannot be their own team lead")
        user = await session.get(UserDB, uid)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        lead = None
        if body.team_lead_id:
            lead = await session.get(UserDB, body.team_lead_id)
            if not lead:
                raise HTTPException(status_code=404, detail="Team lead not found")
        user.team_lead_id = body.team_lead_id
        await session.commit()
        return {"status": "success", "user_id": str(uid),
                "team_lead_id": str(body.team_lead_id) if body.team_lead_id else None,
                "team_lead_name": lead.name if lead else None}

    async def _can_write(session, admin_id):
        user = await session.get(UserDB, uuid.UUID(admin_id))
        return bool(user) and user.email.strip().lower() in core.ADMIN_EMAILS

    @r.get("/admin/settings")
    async def get_settings(admin_id: str = admin, session=Depends(_session)):
        return _settings_view(core, await _can_write(session, admin_id))

    @r.put("/admin/settings")
    async def put_settings(body: SettingsBody, admin_id: str = admin, session=Depends(_session)):
        if not await _can_write(session, admin_id):
            raise HTTPException(status_code=403, detail="Owner access required")
        changes = {}
        if body.webhook_url is not None:
            url = body.webhook_url.strip()
            if url:
                parsed = urlparse(url)
                if parsed.scheme != "https" or not parsed.hostname or len(url) > 500:
                    raise HTTPException(status_code=422, detail="webhook_url must be empty or an https:// URL with a host, max 500 chars")
            changes["webhook_url"] = url
        if body.auto_pipeline_enabled is not None:
            changes["auto_pipeline_enabled"] = body.auto_pipeline_enabled
        now = datetime.now(timezone.utc)
        for key, value in changes.items():
            stmt = pg_insert(AppSettingDB).values(key=key, value=value, updated_at=now, updated_by=uuid.UUID(admin_id))
            await session.execute(stmt.on_conflict_do_update(
                index_elements=["key"], set_={"value": stmt.excluded.value, "updated_at": now, "updated_by": stmt.excluded.updated_by}))
        await session.commit()
        # globals only after the commit succeeds, so memory never runs ahead of the table
        if "webhook_url" in changes:
            core.WEBHOOK_URL = changes["webhook_url"]
        if "auto_pipeline_enabled" in changes:
            core.AUTO_PIPELINE_ENABLED = changes["auto_pipeline_enabled"]
        return _settings_view(core, True)

    return r
