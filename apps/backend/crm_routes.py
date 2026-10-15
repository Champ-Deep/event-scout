"""
Admin console → Exports → Send to CRM (HubSpot / Salesforce / Zoho CRM / Pipedrive). See crm.py.

  GET  /api/admin/crm                        which CRMs are set up (+ the env vars still missing) and each one's last push
  POST /api/admin/crm/{crm}/test             a real authenticated call → the connected account
  GET  /api/admin/crm/{crm}/csv?filters      CRM-ready CSV (that CRM's import column names), same filters as /admin/export
  POST /api/admin/crm/{crm}/push  {filters}  starts a push of the matching contacts → job (poll it)
  GET  /api/admin/crm/jobs/{id}              progress: processed / created / updated / skipped / failed (+ reasons)
  POST /api/admin/crm/jobs/{id}/cancel       stop after the contacts already in flight

Pushes run in the background (one per CRM at a time) with a few contacts in flight at once; each
contact's CRM id is saved in crm_links so sending again updates instead of duplicating. Jobs live in
memory: progress survives page reloads, not a server restart (the links in the DB do).
"""

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import crm
from database import CrmLinkDB, get_session_factory
import contact_view
from api_routes import get_db, require_admin, admin_export_contacts, export_filename

crm_router = APIRouter(prefix="/api/admin/crm", tags=["CRM"], dependencies=[Depends(require_admin)])

PUSH_CONCURRENCY = 4
MAX_FAILURES_KEPT = 200

_jobs: Dict[str, Dict[str, Any]] = {}
_last_by_crm: Dict[str, str] = {}  # crm → newest job id


class PushFilters(BaseModel):
    event: Optional[str] = Field(None, max_length=255)
    userId: Optional[str] = None
    minScore: Optional[int] = Field(None, ge=0, le=100)
    listId: Optional[str] = None


def _connector_class(key: str):
    c = crm.CONNECTORS.get(key)
    if c is None:
        raise HTTPException(status_code=404, detail=f"Unknown CRM '{key}'. Use one of: {', '.join(crm.CRM_KEYS)}")
    return c


def _require_configured(cls):
    missing = cls.missing()
    if missing:
        raise HTTPException(status_code=400, detail=f"{cls.label} isn't connected yet. Set {', '.join(missing)} on the server and restart it.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _public(job: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in job.items() if not k.startswith("_")}


def _scope_label(parts: List[str]) -> str:
    return " · ".join(str(p) for p in parts) if parts else "All contacts"


@crm_router.get("")
async def list_crms():
    out = []
    for key in crm.CRM_KEYS:
        info = crm.connector_info(key)
        last = _jobs.get(_last_by_crm.get(key, ""))
        info["lastPush"] = _public(last) if last else None
        out.append(info)
    return {"crms": out}


@crm_router.post("/{key}/test")
async def test_crm(key: str):
    cls = _connector_class(key)
    _require_configured(cls)
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            account = await cls(client).ping()
        except crm.CrmError as e:
            raise HTTPException(status_code=502, detail=e.message)
    return {"ok": True, "crm": key, "label": cls.label, "account": account, "latencyMs": round((time.perf_counter() - start) * 1000)}


@crm_router.get("/{key}/csv")
async def crm_csv(
    key: str,
    event: Optional[str] = Query(None),
    userId: Optional[str] = Query(None),
    minScore: Optional[int] = Query(None, ge=0, le=100),
    listId: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_db),
):
    """Works without credentials: a file to upload through that CRM's own import wizard."""
    cls = _connector_class(key)
    contacts, parts, owners = await admin_export_contacts(session, event, userId, minScore, listId)
    leads = [crm.make_lead(contact_view.crm_view(c, owners.get(c.user_id, ""))) for c in contacts]
    filename = export_filename(parts, prefix=f"eventscout-{key}")
    return Response(content=crm.crm_csv(key, leads), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"', "X-Row-Count": str(len(leads)),
                             "Access-Control-Expose-Headers": "Content-Disposition, X-Row-Count"})


@crm_router.post("/{key}/push", status_code=202)
async def start_push(key: str, body: PushFilters, admin_id: uuid.UUID = Depends(require_admin),
                     session: AsyncSession = Depends(get_db)):
    cls = _connector_class(key)
    _require_configured(cls)
    running = _jobs.get(_last_by_crm.get(key, ""))
    if running and running["status"] == "running":
        raise HTTPException(status_code=409, detail=f"A push to {cls.label} is already running ({running['processed']}/{running['total']}).")
    contacts, parts, owners = await admin_export_contacts(session, body.event, body.userId, body.minScore, body.listId)
    if not contacts:
        raise HTTPException(status_code=400, detail="No contacts match — nothing to send.")
    leads = [crm.make_lead(contact_view.crm_view(c, owners.get(c.user_id, ""))) for c in contacts]
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id, "crm": key, "label": cls.label, "record": cls.record, "scope": _scope_label(parts),
        "status": "running", "total": len(leads), "processed": 0, "created": 0, "updated": 0, "skipped": 0, "failed": 0,
        "failures": [], "error": None, "startedAt": _now(), "finishedAt": None, "startedBy": str(admin_id),
        "_cancel": False,
    }
    _jobs[job_id] = job
    _last_by_crm[key] = job_id
    job["_task"] = asyncio.create_task(_run_push(job, cls, leads))
    return _public(job)


@crm_router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="That push is no longer known to the server (it restarted).")
    return _public(job)


@crm_router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="That push is no longer known to the server (it restarted).")
    if job["status"] == "running":
        job["_cancel"] = True
    return _public(job)


async def _run_push(job: Dict[str, Any], cls, leads: List[crm.Lead]):
    factory = get_session_factory()
    try:
        if factory is None:
            raise crm.CrmError("Database unavailable", fatal=True)
        async with factory() as session:
            ids = [uuid.UUID(l.contact_id) for l in leads]
            links: Dict[str, CrmLinkDB] = {}
            for i in range(0, len(ids), 500):
                rows = (await session.execute(select(CrmLinkDB).where(
                    CrmLinkDB.crm == job["crm"], CrmLinkDB.contact_id.in_(ids[i:i + 500])))).scalars().all()
                links.update({str(r.contact_id): r for r in rows})

            async with httpx.AsyncClient(timeout=30) as client:
                conn = cls(client)
                sem = asyncio.Semaphore(PUSH_CONCURRENCY)

                async def one(lead: crm.Lead):
                    async with sem:
                        if job["_cancel"] or job["status"] != "running":
                            return lead, None, None
                        link = links.get(lead.contact_id)
                        try:
                            action, ext = await conn.upsert(lead, link.external_id if link else None)
                            return lead, action, ext
                        except crm.CrmError as e:
                            if e.fatal:
                                job["_cancel"] = True  # the rest of the chunk stops too
                                raise
                            return lead, "failed", e.message
                        except Exception as e:  # a CRM answering in an unexpected shape fails one row, not the push
                            return lead, "failed", f"Unexpected answer from {cls.label}: {e.__class__.__name__}"

                # chunks keep DB writes on this one coroutine (an AsyncSession isn't safe to share)
                for i in range(0, len(leads), 25):
                    if job["_cancel"]:
                        break
                    results = await asyncio.gather(*(one(l) for l in leads[i:i + 25]))
                    for lead, action, ext in results:
                        if action is None:
                            continue
                        job["processed"] += 1
                        if action == "failed":
                            job["failed"] += 1
                            if len(job["failures"]) < MAX_FAILURES_KEPT:
                                job["failures"].append({"contactId": lead.contact_id, "name": lead.name or lead.email, "reason": ext})
                            continue
                        job[action] += 1
                        link = links.get(lead.contact_id)
                        if link:
                            link.external_id, link.synced_at = ext, datetime.now(timezone.utc)
                        else:
                            link = CrmLinkDB(contact_id=uuid.UUID(lead.contact_id), crm=job["crm"], external_id=ext)
                            session.add(link)
                            links[lead.contact_id] = link
                    await session.commit()
        job["status"] = "cancelled" if job["_cancel"] else "done"
    except crm.CrmError as e:
        job["status"], job["error"] = "failed", e.message
    except Exception as e:
        job["status"], job["error"] = "failed", f"The push stopped: {e.__class__.__name__}: {e}"
    finally:
        job["finishedAt"] = _now()
        job.pop("_task", None)
        print(f"[CRM] {job['label']} push {job['status']}: {job['created']} created, {job['updated']} updated, "
              f"{job['skipped']} skipped, {job['failed']} failed of {job['total']}" + (f" — {job['error']}" if job["error"] else ""))
