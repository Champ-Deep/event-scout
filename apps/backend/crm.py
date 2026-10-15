"""
CRM connectors for the admin console's Exports → "Send to CRM".

Four CRMs, each configured only through server env vars (never stored in the DB or sent to the browser):

  HubSpot     HUBSPOT_ACCESS_TOKEN                      private-app token; scopes crm.objects.contacts.read/write
                                                       (+ notes write for the EventScout note)
  Salesforce  SALESFORCE_DOMAIN, SALESFORCE_CLIENT_ID,  connected app with the OAuth client-credentials flow
              SALESFORCE_CLIENT_SECRET                  (DOMAIN = your My Domain host, e.g. acme.my.salesforce.com)
              [SALESFORCE_API_VERSION=v61.0, SALESFORCE_LEAD_SOURCE]
  Zoho CRM    ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET,       self-client refresh token (scope ZohoCRM.modules.leads.ALL)
              ZOHO_REFRESH_TOKEN                        [ZOHO_ACCOUNTS_URL=https://accounts.zoho.com (.eu/.in/… for
              [ZOHO_API_VERSION=v6, ZOHO_LEAD_SOURCE]    other data centres)]
  Pipedrive   PIPEDRIVE_API_TOKEN                       [PIPEDRIVE_COMPANY_DOMAIN=acme → acme.pipedrive.com]

What a contact becomes: HubSpot contact · Salesforce lead · Zoho lead · Pipedrive person (+ organization).
Every push is an upsert: a contact sent before (crm_links) updates that record; otherwise an existing
record with the same email is updated; otherwise a new one is created. Updates only send fields that
have a value, so nothing typed in the CRM is blanked out. Event, score, LinkedIn and notes go into the
lead Description (Salesforce / Zoho) or a note on the new record (HubSpot / Pipedrive).
"""

import asyncio
import csv

from sanitize import safe_csv_writer
import html
import io
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import httpx

CRM_KEYS = ("hubspot", "salesforce", "zoho", "pipedrive")


class CrmError(Exception):
    """A CRM call failed. `fatal` = every other contact would fail the same way (bad credentials,
    missing permission, CRM unreachable), so the push stops instead of failing row by row."""

    def __init__(self, message: str, fatal: bool = False, status: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.fatal = fatal
        self.status = status


@dataclass
class Lead:
    contact_id: str
    name: str
    first: str
    last: str
    email: str
    phone: str
    company: str
    title: str
    linkedin: str
    event: str
    score: Optional[int]
    level: str
    notes: str
    owner: str
    source: str

    def summary(self) -> str:
        """The plain-text block that carries what a CRM has no standard field for."""
        lines = []
        met = f"Met at {self.event}" if self.event else "Captured"
        by = f" by {self.owner}" if self.owner else ""
        lines.append(f"{met}{by} via EventScout" + (f" ({self.source})" if self.source else ""))
        if self.score is not None:
            lines.append(f"EventScout score: {self.score}/100" + (f" ({self.level})" if self.level else ""))
        if self.linkedin:
            lines.append(f"LinkedIn: {self.linkedin}")
        if self.notes.strip():
            lines += ["", "Notes:", self.notes.strip()]
        return "\n".join(lines)


_PREFIXES = {"dr", "mr", "mrs", "ms", "miss", "prof", "sir"}


def split_name(name: str) -> Tuple[str, str]:
    """"Dr. Maya van Patel" → ("Maya", "van Patel"); a one-word name is the last name (CRMs require one)."""
    words = [w for w in str(name or "").split() if w]
    while len(words) > 1 and words[0].lower().rstrip(".") in _PREFIXES:
        words = words[1:]
    if not words:
        return "", ""
    if len(words) == 1:
        return "", words[0]
    return words[0], " ".join(words[1:])


def _clip(value: str, n: int) -> str:
    return value if len(value) <= n else value[: n - 1] + "…"


def _first_email(value: str) -> str:
    """Card fields sometimes hold two addresses ("a@x.com / b@y.com") — CRMs want exactly one."""
    m = re.search(r"[\w.+'-]+@[\w-]+(\.[\w-]+)+", value or "")
    return m.group(0).lower() if m else ""


def _note_html(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def _error_text(r: httpx.Response) -> str:
    """The CRM's own error message, whichever shape it uses."""
    try:
        data = r.json()
    except ValueError:
        return (r.text or "").strip()[:240] or f"HTTP {r.status_code}"
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, dict):
        if isinstance(data.get("data"), list) and data["data"] and isinstance(data["data"][0], dict):
            data = data["data"][0]
        for k in ("message", "error_description", "error", "errorCode", "code"):
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()[:240]
    return f"HTTP {r.status_code}"


# ---------------------------------------------------------------- connectors

class Connector:
    key = ""
    label = ""
    env: Tuple[str, ...] = ()           # required env vars
    record = "record"                   # what a contact becomes there (for the UI)

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    @classmethod
    def missing(cls) -> List[str]:
        return [k for k in cls.env if not os.environ.get(k, "").strip()]

    @classmethod
    def configured(cls) -> bool:
        return not cls.missing()

    async def request(self, method: str, url: str, **kw) -> httpx.Response:
        """One call with retries on rate limits / 5xx / timeouts; 401 and 403 are fatal for the push."""
        for attempt in range(4):
            try:
                r = await self.client.request(method, url, headers=await self.headers(), **kw)
            except httpx.TimeoutException:
                if attempt < 3:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise CrmError(f"{self.label} didn't answer in time.")
            except httpx.HTTPError as e:
                raise CrmError(f"Couldn't reach {self.label}: {e.__class__.__name__}.", fatal=True)
            if (r.status_code == 429 or r.status_code >= 500) and attempt < 3:
                try:
                    wait = float(r.headers.get("retry-after", "") or 2 ** attempt)
                except ValueError:
                    wait = 2 ** attempt
                await asyncio.sleep(min(max(wait, 0.5), 30))
                continue
            if r.status_code == 401:
                raise CrmError(f"{self.label} rejected the credentials: {_error_text(r)}. Check the server's {self.label} settings.",
                               fatal=True, status=401)
            if r.status_code == 403:
                raise CrmError(f"{self.label} denied access: {_error_text(r)}. The token or app is missing a permission.",
                               fatal=True, status=403)
            return r
        return r  # unreachable; keeps type checkers calm

    async def headers(self) -> Dict[str, str]:
        return {}

    def fail(self, r: httpx.Response, what: str) -> CrmError:
        return CrmError(f"{what}: {_error_text(r)}", status=r.status_code)

    async def ping(self) -> str:
        """A cheap authenticated read; returns a short description of the connected account."""
        raise NotImplementedError

    async def upsert(self, lead: Lead, external_id: Optional[str]) -> Tuple[str, str]:
        """→ (action, external_id); action = created | updated | skipped."""
        raise NotImplementedError


class HubSpot(Connector):
    key, label, record = "hubspot", "HubSpot", "contact"
    env = ("HUBSPOT_ACCESS_TOKEN",)
    base = "https://api.hubapi.com"

    async def headers(self):
        return {"Authorization": f"Bearer {os.environ['HUBSPOT_ACCESS_TOKEN'].strip()}"}

    async def ping(self):
        r = await self.request("GET", f"{self.base}/crm/v3/objects/contacts", params={"limit": 1})
        if not r.is_success:
            raise self.fail(r, "HubSpot test failed")
        return "Contacts API reachable"

    @staticmethod
    def props(lead: Lead) -> Dict[str, str]:
        p = {"email": lead.email, "firstname": lead.first, "lastname": lead.last, "phone": lead.phone,
             "company": lead.company, "jobtitle": lead.title}
        return {k: v for k, v in p.items() if v}

    async def _patch(self, ident: str, props: Dict[str, str], by_email: bool) -> Optional[str]:
        url = f"{self.base}/crm/v3/objects/contacts/{quote(ident, safe='')}"  # "+" and "@" in emails
        r = await self.request("PATCH", url, params={"idProperty": "email"} if by_email else None, json={"properties": props})
        if r.status_code == 404:
            return None
        if not r.is_success:
            raise self.fail(r, "HubSpot update failed")
        return str(r.json().get("id"))

    async def upsert(self, lead, external_id):
        props = self.props(lead)
        if external_id:
            found = await self._patch(external_id, props, by_email=False)
            if found:
                return "updated", found
        if lead.email:
            found = await self._patch(lead.email, props, by_email=True)
            if found:
                return "updated", found
        r = await self.request("POST", f"{self.base}/crm/v3/objects/contacts", json={"properties": props})
        if r.status_code == 409:  # created by someone else since we looked
            m = re.search(r"Existing ID:\s*(\d+)", _error_text(r))
            if m:
                found = await self._patch(m.group(1), props, by_email=False)
                if found:
                    return "updated", found
        if not r.is_success:
            raise self.fail(r, "HubSpot create failed")
        new_id = str(r.json().get("id"))
        await self._note(new_id, lead)
        return "created", new_id

    async def _note(self, contact_id: str, lead: Lead):
        """The EventScout summary as a note on the new contact. Best effort: a token without the
        notes scope still gets the contact itself."""
        body = {
            "properties": {"hs_timestamp": datetime.now(timezone.utc).isoformat(), "hs_note_body": _note_html(lead.summary())},
            "associations": [{"to": {"id": contact_id},
                              "types": [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 202}]}],
        }
        try:
            await self.request("POST", f"{self.base}/crm/v3/objects/notes", json=body)
        except CrmError:
            pass


class Salesforce(Connector):
    key, label, record = "salesforce", "Salesforce", "lead"
    env = ("SALESFORCE_DOMAIN", "SALESFORCE_CLIENT_ID", "SALESFORCE_CLIENT_SECRET")

    def __init__(self, client):
        super().__init__(client)
        self._token: Optional[str] = None
        self._instance = ""

    @property
    def version(self):
        return os.environ.get("SALESFORCE_API_VERSION", "").strip() or "v61.0"

    async def _login(self):
        domain = re.sub(r"^https?://", "", os.environ["SALESFORCE_DOMAIN"].strip()).strip("/")
        try:
            r = await self.client.post(f"https://{domain}/services/oauth2/token", data={
                "grant_type": "client_credentials",
                "client_id": os.environ["SALESFORCE_CLIENT_ID"].strip(),
                "client_secret": os.environ["SALESFORCE_CLIENT_SECRET"].strip(),
            })
        except httpx.HTTPError as e:
            raise CrmError(f"Couldn't reach Salesforce at {domain}: {e.__class__.__name__}.", fatal=True)
        if not r.is_success:
            raise CrmError(f"Salesforce sign-in failed: {_error_text(r)}. Check SALESFORCE_DOMAIN / CLIENT_ID / CLIENT_SECRET "
                           "and that the connected app allows the client-credentials flow.", fatal=True, status=r.status_code)
        data = r.json()
        self._token = data.get("access_token")
        self._instance = (data.get("instance_url") or f"https://{domain}").rstrip("/")

    async def headers(self):
        if not self._token:
            await self._login()
        return {"Authorization": f"Bearer {self._token}"}

    async def api(self, path: str) -> str:
        if not self._token:
            await self._login()
        return f"{self._instance}/services/data/{self.version}{path}"

    async def ping(self):
        r = await self.request("GET", await self.api("/query"), params={"q": "SELECT Id FROM Lead LIMIT 1"})
        if not r.is_success:
            raise self.fail(r, "Salesforce test failed")
        return self._instance.replace("https://", "")

    @staticmethod
    def fields(lead: Lead) -> Dict[str, str]:
        f = {
            "FirstName": _clip(lead.first, 40),
            "LastName": _clip(lead.last or lead.name or lead.email or "Unknown", 80),
            "Company": _clip(lead.company or "[not provided]", 255),
            "Title": _clip(lead.title, 128),
            "Email": _clip(lead.email, 80),
            "Phone": _clip(lead.phone, 40),
            "Description": _clip(lead.summary(), 32000),
            "LeadSource": os.environ.get("SALESFORCE_LEAD_SOURCE", "").strip(),
        }
        return {k: v for k, v in f.items() if v}

    async def _patch(self, lead_id: str, fields: Dict[str, str]) -> Optional[str]:
        """→ "updated", "skipped" (already converted) or None (that lead no longer exists)."""
        r = await self.request("PATCH", await self.api(f"/sobjects/Lead/{lead_id}"), json=fields)
        if r.status_code in (200, 204):
            return "updated"
        text = r.text or ""
        if r.status_code == 404 or "ENTITY_IS_DELETED" in text:
            return None
        if "CANNOT_UPDATE_CONVERTED_LEAD" in text:
            return "skipped"
        raise self.fail(r, "Salesforce update failed")

    async def upsert(self, lead, external_id):
        fields = self.fields(lead)
        if external_id:
            done = await self._patch(external_id, fields)
            if done:
                return done, external_id
        if lead.email:
            email = lead.email.replace("\\", "\\\\").replace("'", "\\'")
            r = await self.request("GET", await self.api("/query"), params={
                "q": f"SELECT Id FROM Lead WHERE Email = '{email}' AND IsConverted = false ORDER BY CreatedDate DESC LIMIT 1"})
            if not r.is_success:
                raise self.fail(r, "Salesforce lookup failed")
            rows = r.json().get("records") or []
            if rows:
                done = await self._patch(rows[0]["Id"], fields)
                if done:
                    return done, rows[0]["Id"]
        r = await self.request("POST", await self.api("/sobjects/Lead"), json=fields)
        if not r.is_success:
            raise self.fail(r, "Salesforce create failed")
        return "created", str(r.json().get("id"))


class Zoho(Connector):
    key, label, record = "zoho", "Zoho CRM", "lead"
    env = ("ZOHO_CLIENT_ID", "ZOHO_CLIENT_SECRET", "ZOHO_REFRESH_TOKEN")

    def __init__(self, client):
        super().__init__(client)
        self._token: Optional[str] = None
        self._api_domain = ""

    @property
    def version(self):
        return os.environ.get("ZOHO_API_VERSION", "").strip() or "v6"

    async def _login(self):
        accounts = (os.environ.get("ZOHO_ACCOUNTS_URL", "").strip() or "https://accounts.zoho.com").rstrip("/")
        try:
            r = await self.client.post(f"{accounts}/oauth/v2/token", params={
                "grant_type": "refresh_token",
                "refresh_token": os.environ["ZOHO_REFRESH_TOKEN"].strip(),
                "client_id": os.environ["ZOHO_CLIENT_ID"].strip(),
                "client_secret": os.environ["ZOHO_CLIENT_SECRET"].strip(),
            })
        except httpx.HTTPError as e:
            raise CrmError(f"Couldn't reach Zoho at {accounts}: {e.__class__.__name__}.", fatal=True)
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if not r.is_success or not data.get("access_token"):
            # Zoho answers 200 with {"error": "invalid_code"} for a bad refresh token
            raise CrmError(f"Zoho sign-in failed: {data.get('error') or _error_text(r)}. Check ZOHO_REFRESH_TOKEN / CLIENT_ID / "
                           "CLIENT_SECRET and that ZOHO_ACCOUNTS_URL matches your data centre (.com, .eu, .in, .com.au…).", fatal=True)
        self._token = data["access_token"]
        self._api_domain = (data.get("api_domain") or "https://www.zohoapis.com").rstrip("/")

    async def headers(self):
        if not self._token:
            await self._login()
        return {"Authorization": f"Zoho-oauthtoken {self._token}"}

    async def api(self, path: str) -> str:
        if not self._token:
            await self._login()
        return f"{self._api_domain}/crm/{self.version}{path}"

    async def ping(self):
        r = await self.request("GET", await self.api("/Leads"), params={"fields": "Last_Name", "per_page": 1})
        if r.status_code not in (200, 204):
            raise self.fail(r, "Zoho test failed")
        return self._api_domain.replace("https://", "")

    @staticmethod
    def fields(lead: Lead) -> Dict[str, str]:
        f = {
            "First_Name": _clip(lead.first, 40),
            "Last_Name": _clip(lead.last or lead.name or lead.email or "Unknown", 80),
            "Company": _clip(lead.company or "[not provided]", 200),
            "Designation": _clip(lead.title, 100),
            "Email": _clip(lead.email, 100),
            "Phone": _clip(lead.phone, 30),
            "Description": _clip(lead.summary(), 32000),
            "Lead_Source": os.environ.get("ZOHO_LEAD_SOURCE", "").strip(),
        }
        return {k: v for k, v in f.items() if v}

    @staticmethod
    def _row(r: httpx.Response) -> Dict[str, Any]:
        try:
            rows = r.json().get("data") or []
        except ValueError:
            return {}
        return rows[0] if rows and isinstance(rows[0], dict) else {}

    async def upsert(self, lead, external_id):
        rec = self.fields(lead)
        if external_id:
            r = await self.request("PUT", await self.api(f"/Leads/{external_id}"), json={"data": [rec]})
            row = self._row(r)
            if row.get("code") == "SUCCESS":
                return "updated", external_id
            if row.get("code") != "INVALID_DATA" and r.status_code != 404:
                raise CrmError(f"Zoho update failed: {row.get('message') or _error_text(r)}", status=r.status_code)
            # that lead was deleted or converted in Zoho — fall through and match by email / create
        if lead.email:
            r = await self.request("POST", await self.api("/Leads/upsert"), json={"data": [rec], "duplicate_check_fields": ["Email"]})
        else:
            r = await self.request("POST", await self.api("/Leads"), json={"data": [rec]})
        row = self._row(r)
        if row.get("code") != "SUCCESS":
            raise CrmError(f"Zoho save failed: {row.get('message') or _error_text(r)}", status=r.status_code)
        new_id = str((row.get("details") or {}).get("id"))
        return ("updated" if row.get("action") == "update" else "created"), new_id


class Pipedrive(Connector):
    key, label, record = "pipedrive", "Pipedrive", "person"
    env = ("PIPEDRIVE_API_TOKEN",)

    def __init__(self, client):
        super().__init__(client)
        self._orgs: Dict[str, int] = {}
        self._org_lock = asyncio.Lock()

    @property
    def base(self):
        sub = re.sub(r"^https?://", "", os.environ.get("PIPEDRIVE_COMPANY_DOMAIN", "").strip()).split(".")[0]
        return f"https://{sub}.pipedrive.com/api/v1" if sub else "https://api.pipedrive.com/v1"

    async def headers(self):
        return {"x-api-token": os.environ["PIPEDRIVE_API_TOKEN"].strip()}

    async def ping(self):
        r = await self.request("GET", f"{self.base}/users/me")
        if not r.is_success:
            raise self.fail(r, "Pipedrive test failed")
        d = r.json().get("data") or {}
        return d.get("company_name") or d.get("company_domain") or "Connected"

    async def _org_id(self, company: str) -> Optional[int]:
        """The organization called exactly `company` (created if missing), cached for the push."""
        if not company:
            return None
        key = company.lower()
        async with self._org_lock:  # two contacts from the same company must not create it twice
            if key in self._orgs:
                return self._orgs[key]
            org_id = None
            if len(company) >= 2:
                r = await self.request("GET", f"{self.base}/organizations/search",
                                       params={"term": company, "fields": "name", "exact_match": "true", "limit": 1})
                if r.is_success:
                    items = ((r.json().get("data") or {}).get("items")) or []
                    if items:
                        org_id = items[0]["item"]["id"]
            if org_id is None:
                r = await self.request("POST", f"{self.base}/organizations", json={"name": _clip(company, 255)})
                if not r.is_success:
                    raise self.fail(r, "Pipedrive organization create failed")
                org_id = r.json()["data"]["id"]
            self._orgs[key] = org_id
            return org_id

    async def _update(self, person_id, body) -> bool:
        r = await self.request("PUT", f"{self.base}/persons/{person_id}", json=body)
        if r.status_code in (404, 410):
            return False
        if not r.is_success:
            raise self.fail(r, "Pipedrive update failed")
        return bool((r.json().get("data") or {}).get("active_flag", True))

    async def upsert(self, lead, external_id):
        org_id = await self._org_id(lead.company)
        # updates leave email / phone alone: Pipedrive replaces the whole list, which would drop any
        # address or number someone added in Pipedrive
        body: Dict[str, Any] = {"name": _clip(lead.name or lead.email or "Unknown", 255)}
        if org_id:
            body["org_id"] = org_id
        if lead.title:
            body["job_title"] = _clip(lead.title, 255)
        if external_id and await self._update(external_id, body):
            return "updated", str(external_id)
        if lead.email:
            r = await self.request("GET", f"{self.base}/persons/search",
                                   params={"term": lead.email, "fields": "email", "exact_match": "true", "limit": 1})
            if not r.is_success:
                raise self.fail(r, "Pipedrive lookup failed")
            items = ((r.json().get("data") or {}).get("items")) or []
            if items and await self._update(items[0]["item"]["id"], body):
                return "updated", str(items[0]["item"]["id"])
        if lead.email:
            body["email"] = [{"value": lead.email, "primary": True, "label": "work"}]
        if lead.phone:
            body["phone"] = [{"value": lead.phone, "primary": True, "label": "work"}]
        r = await self.request("POST", f"{self.base}/persons", json=body)
        if not r.is_success:
            raise self.fail(r, "Pipedrive create failed")
        new_id = r.json()["data"]["id"]
        try:  # best effort, like HubSpot's note
            await self.request("POST", f"{self.base}/notes", json={"content": _note_html(lead.summary()), "person_id": new_id})
        except CrmError:
            pass
        return "created", str(new_id)


CONNECTORS = {c.key: c for c in (HubSpot, Salesforce, Zoho, Pipedrive)}


def connector_info(key: str) -> Dict[str, Any]:
    c = CONNECTORS[key]
    return {"key": c.key, "label": c.label, "record": c.record, "configured": c.configured(), "missing": c.missing()}


# ---------------------------------------------------------------- CRM-ready CSV files

def _lead_source(env_name: str) -> str:
    return os.environ.get(env_name, "").strip() or "EventScout"


def crm_csv(key: str, leads: List[Lead]) -> str:
    """A CSV with the column names that CRM's import wizard maps on its own."""
    if key == "hubspot":
        head = ["First Name", "Last Name", "Email", "Phone Number", "Company Name", "Job Title", "LinkedIn URL",
                "Event", "Lead Score", "Lead Level", "EventScout Owner", "Notes"]
        rows = [[l.first, l.last, l.email, l.phone, l.company, l.title, l.linkedin, l.event,
                 "" if l.score is None else l.score, l.level, l.owner, l.notes] for l in leads]
    elif key == "salesforce":
        head = ["First Name", "Last Name", "Company", "Title", "Email", "Phone", "Lead Source", "Description"]
        src = _lead_source("SALESFORCE_LEAD_SOURCE")
        rows = [[l.first, l.last or l.name, l.company or "[not provided]", l.title, l.email, l.phone, src, l.summary()] for l in leads]
    elif key == "zoho":
        head = ["First Name", "Last Name", "Company", "Designation", "Email", "Phone", "Lead Source", "Description"]
        src = _lead_source("ZOHO_LEAD_SOURCE")
        rows = [[l.first, l.last or l.name, l.company or "[not provided]", l.title, l.email, l.phone, src, l.summary()] for l in leads]
    elif key == "pipedrive":
        head = ["Name", "First name", "Last name", "Email", "Phone", "Organization", "Job title", "Note"]
        rows = [[l.name, l.first, l.last, l.email, l.phone, l.company, l.title, l.summary()] for l in leads]
    else:
        raise ValueError(key)
    buf = io.StringIO()
    w = safe_csv_writer(buf)  # formula-injection safe (sanitize.py)
    w.writerow(head)
    w.writerows(rows)
    return "﻿" + buf.getvalue()


def make_lead(view: Dict[str, Any]) -> Lead:
    """A contact_view.crm_view() dict → Lead ("N/A" placeholders, score and level already resolved there,
    the same way the app and the console show them)."""
    first, last = split_name(view.get("name") or "")
    return Lead(
        contact_id=view["id"], name=view.get("name") or "", first=first, last=last,
        email=_first_email(view.get("email") or ""), phone=view.get("phone") or "",
        company=view.get("company") or "", title=view.get("title") or "", linkedin=view.get("linkedin") or "",
        event=view.get("event") or "", score=view.get("score"), level=view.get("level") or "",
        notes=view.get("notes") or "", owner=view.get("owner") or "", source=view.get("source") or "",
    )
