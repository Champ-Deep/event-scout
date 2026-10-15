"""Email verification — the checks https://github.com/reacherhq/check-if-email-exists does, in Python.

For one address:
  1. syntax      — is it a well-formed address (email-validator)
  2. mx          — does the domain exist and accept mail (MX records; an A record counts, RFC 5321)
  3. misc        — disposable / role account (info@, sales@…) / free provider (gmail.com…)
  4. smtp        — ask the domain's mail server whether the mailbox exists (RCPT TO), without
                   sending anything, and whether it accepts every address (catch-all)

Result `is_reachable` (same values as Reacher):
  safe     the mailbox exists and isn't a catch-all / disposable / role address
  risky    exists but catch-all, disposable, role account or full inbox
  invalid  bad syntax, the domain takes no mail, or the server says the mailbox doesn't exist
  unknown  the mail server couldn't be asked (port 25 blocked, greylisting, timeout)

Port 25 is blocked on many networks and hosts, so the SMTP step often ends as "unknown" — the
first three checks still run. To use a real Reacher server instead (it handles Yahoo / Hotmail
specially and can run behind proxies), set REACHER_BACKEND_URL (and REACHER_SECRET if it has one):
`docker run -p 8080:8080 reacherhq/backend:v0.11.7` → REACHER_BACKEND_URL=http://localhost:8080

Env: EMAIL_VERIFY_SMTP=0 skips the SMTP step · EMAIL_VERIFY_FROM / EMAIL_VERIFY_HELO set the
sender used in the SMTP conversation · EMAIL_VERIFY_TIMEOUT (seconds, default 10).
"""
import asyncio
import os
import random
import smtplib
import socket
import string
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import dns.asyncresolver
import dns.exception
import dns.resolver
from email_validator import EmailNotValidError, validate_email

REACHER_URL = os.environ.get("REACHER_BACKEND_URL", "").rstrip("/")
REACHER_SECRET = os.environ.get("REACHER_SECRET", "")
SMTP_ENABLED = os.environ.get("EMAIL_VERIFY_SMTP", "1").strip().lower() not in ("0", "false", "no", "off")
TIMEOUT = float(os.environ.get("EMAIL_VERIFY_TIMEOUT", "10") or 10)
HELO = os.environ.get("EMAIL_VERIFY_HELO", "").strip() or (socket.getfqdn() or "localhost")
FROM = (os.environ.get("EMAIL_VERIFY_FROM", "").strip() or os.environ.get("SMTP_FROM", "").strip()
        or f"verify@{HELO if '.' in HELO else 'localhost'}")

ROLE_ACCOUNTS = {
    "abuse", "account", "accounts", "admin", "administrator", "billing", "careers", "contact", "contactus",
    "customercare", "customerservice", "enquiries", "enquiry", "events", "feedback", "finance", "hello", "help",
    "helpdesk", "hostmaster", "hr", "info", "inquiries", "inquiry", "jobs", "legal", "mail", "marketing", "media",
    "news", "newsletter", "no-reply", "noreply", "do-not-reply", "donotreply", "office", "orders", "partners",
    "postmaster", "press", "privacy", "recruitment", "reservations", "root", "sales", "security", "service",
    "services", "shop", "spam", "support", "team", "webmaster", "welcome",
}
FREE_PROVIDERS = {
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.in", "yahoo.co.uk", "ymail.com", "rocketmail.com",
    "outlook.com", "hotmail.com", "hotmail.co.uk", "live.com", "msn.com", "icloud.com", "me.com", "mac.com",
    "aol.com", "proton.me", "protonmail.com", "pm.me", "zoho.com", "zohomail.com", "gmx.com", "gmx.de", "gmx.net",
    "mail.com", "yandex.com", "yandex.ru", "rediffmail.com", "qq.com", "163.com", "126.com", "sina.com",
    "web.de", "tutanota.com", "tuta.io", "fastmail.com", "hey.com", "inbox.com", "mail.ru", "libero.it",
}
# Common throwaway-mail domains (Reacher uses the much longer `mailchecker` list).
DISPOSABLE = {
    "10minutemail.com", "10minutemail.net", "20minutemail.com", "33mail.com", "anonaddy.me", "burnermail.io",
    "discard.email", "dispostable.com", "dropmail.me", "emailondeck.com", "fakeinbox.com", "getairmail.com",
    "getnada.com", "guerrillamail.biz", "guerrillamail.com", "guerrillamail.de", "guerrillamail.info",
    "guerrillamail.net", "guerrillamail.org", "guerrillamailblock.com", "harakirimail.com", "inboxkitten.com",
    "jetable.org", "mail.tm", "mail-temp.com", "maildrop.cc", "mailinator.com", "mailinator.net", "mailnesia.com",
    "mailpoof.com", "mintemail.com", "moakt.com", "mohmal.com", "mytemp.email", "nada.email", "sharklasers.com",
    "spam4.me", "spamgourmet.com", "temp-mail.io", "temp-mail.org", "tempail.com", "tempmail.com", "tempmail.dev",
    "tempmail.net", "tempmailo.com", "tempr.email", "throwawaymail.com", "trashmail.com", "trashmail.de",
    "trashmail.net", "yopmail.com", "yopmail.fr", "yopmail.net", "emailfake.com", "fakemail.net", "mailcatch.com",
    "mailsac.com", "tmpmail.org", "tmail.ws", "byom.de", "spambox.us", "grr.la", "pokemail.net", "mvrht.net",
}

_NOT_EXIST = ("does not exist", "doesn't exist", "no such user", "user unknown", "unknown user", "invalid recipient",
              "recipient rejected", "address rejected", "mailbox unavailable", "no mailbox", "not found",
              "unknown recipient", "invalid mailbox", "account does not exist", "no such recipient", "undeliverable")
_FULL = ("full", "quota", "over quota", "insufficient storage")
_DISABLED = ("disabled", "deactivated", "suspended", "inactive", "blocked account", "account has been")


def _syntax(email: str) -> Dict[str, Any]:
    raw = (email or "").strip()
    try:
        v = validate_email(raw, check_deliverability=False)
        return {"address": raw, "is_valid_syntax": True, "username": v.local_part, "domain": v.domain.lower(),
                "normalized_email": v.normalized}
    except EmailNotValidError as e:
        user, _, dom = raw.partition("@")
        return {"address": raw, "is_valid_syntax": False, "username": user, "domain": dom.lower(),
                "normalized_email": None, "error": str(e)}


async def _mx(domain: str) -> Tuple[List[str], Optional[str]]:
    """Mail servers for the domain, best first, or ([], why not)."""
    resolver = dns.asyncresolver.Resolver()
    resolver.lifetime = 8
    try:
        answers = await resolver.resolve(domain, "MX")
        recs = sorted(((r.preference, str(r.exchange).rstrip(".")) for r in answers), key=lambda x: x[0])
        hosts = [h for _, h in recs if h and h != "."]
        return (hosts, None) if hosts else ([], "The domain says it accepts no email (null MX)")
    except dns.resolver.NXDOMAIN:
        return [], "The domain doesn't exist"
    except dns.resolver.NoAnswer:
        try:  # no MX: mail goes to the domain's own address (RFC 5321 §5.1)
            await resolver.resolve(domain, "A")
            return [domain], None
        except dns.exception.DNSException:
            return [], "The domain has no mail server"
    except dns.exception.Timeout:
        return [], "DNS lookup timed out"
    except dns.exception.DNSException as e:
        return [], f"DNS lookup failed ({e.__class__.__name__})"


def _smtp_probe(hosts: List[str], email: str, domain: str) -> Dict[str, Any]:
    """RCPT TO the address (and a random one, for catch-all) — nothing is sent. Blocking: run in a thread."""
    out: Dict[str, Any] = {"can_connect_smtp": False, "has_full_inbox": False, "is_catch_all": False,
                           "is_deliverable": False, "is_disabled": False, "error": None, "server": None, "code": None}
    last_err = None
    for host in hosts[:2]:
        s = smtplib.SMTP(timeout=TIMEOUT)
        try:
            s.connect(host, 25)
            code, _ = s.ehlo(HELO)
            if code != 250:
                s.helo(HELO)
            code, msg = s.mail(FROM)
            if code >= 400:
                last_err = f"The mail server refused the sender ({code})"
                continue
            out["can_connect_smtp"], out["server"] = True, host
            code, msg = s.rcpt(email)
            text = (msg or b"").decode("utf-8", "ignore").lower()
            out["code"] = code
            if code in (250, 251):
                out["is_deliverable"] = True
            elif code == 552 or any(w in text for w in _FULL):
                out["has_full_inbox"] = True
            elif any(w in text for w in _DISABLED):
                out["is_disabled"] = True
            elif 500 <= code < 600:
                out["is_deliverable"] = False
                out["error"] = None if any(w in text for w in _NOT_EXIST) else f"Rejected: {code} {text[:120]}"
                out["rejected"] = True
            else:  # 4xx: greylisting / try later
                out["error"] = f"The mail server said try later ({code})"
                out["temporary"] = True
            if out["is_deliverable"]:
                fake = "".join(random.choices(string.ascii_lowercase + string.digits, k=18)) + "@" + domain
                c2, _ = s.rcpt(fake)
                out["is_catch_all"] = c2 in (250, 251)
            return out
        except (socket.timeout, TimeoutError):
            last_err = "The mail server didn't answer (port 25 may be blocked on this network)"
        except (ConnectionRefusedError, smtplib.SMTPConnectError):
            last_err = "The mail server refused the connection"
        except (smtplib.SMTPServerDisconnected, smtplib.SMTPResponseException, OSError) as e:
            last_err = f"Couldn't talk to the mail server ({e.__class__.__name__})"
        finally:
            try:
                s.close()
            except Exception:
                pass
    out["error"] = last_err or "Couldn't reach the mail server"
    return out


def _verdict(r: Dict[str, Any]) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    syn, mx, misc, smtp = r["syntax"], r["mx"], r["misc"], r.get("smtp") or {}
    if not syn["is_valid_syntax"]:
        return "invalid", [syn.get("error") or "Not a valid email address"]
    if not mx["accepts_mail"]:
        return "invalid", [mx.get("error") or "The domain doesn't accept email"]
    reasons.append("The domain accepts email")
    if misc["is_disposable"]:
        reasons.append("Disposable / throwaway address")
    if misc["is_role_account"]:
        reasons.append(f"Role address ({syn['username']}@) — reaches a team, not a person")
    if misc["is_free_provider"]:
        reasons.append("Personal mail provider (not a company domain)")
    if not r.get("smtp_checked"):
        reasons.append("Mailbox not checked (mail-server check is off)")
        return ("risky" if misc["is_disposable"] else "unknown"), reasons
    if not smtp.get("can_connect_smtp"):
        reasons.append(smtp.get("error") or "Couldn't ask the mail server")
        return ("risky" if misc["is_disposable"] else "unknown"), reasons
    if smtp.get("is_disabled"):
        return "invalid", reasons + ["The mailbox is disabled"]
    if smtp.get("has_full_inbox"):
        return "risky", reasons + ["The mailbox is full"]
    if smtp.get("rejected"):
        return "invalid", reasons + [smtp.get("error") or "The mail server says this mailbox doesn't exist"]
    if not smtp.get("is_deliverable"):
        return "unknown", reasons + [smtp.get("error") or "The mail server didn't confirm the mailbox"]
    if smtp.get("is_catch_all"):
        return "risky", reasons + ["The domain accepts every address (catch-all), so this one can't be confirmed"]
    reasons.append("The mailbox exists")
    return ("risky" if misc["is_disposable"] or misc["is_role_account"] else "safe"), reasons


async def _via_reacher(email: str) -> Dict[str, Any]:
    import httpx
    headers = {"x-reacher-secret": REACHER_SECRET} if REACHER_SECRET else {}
    async with httpx.AsyncClient(timeout=60) as c:
        resp = await c.post(f"{REACHER_URL}/v0/check_email", json={"to_email": email}, headers=headers)
        resp.raise_for_status()
        d = resp.json()
    syn, mx, misc, smtp = d.get("syntax") or {}, d.get("mx") or {}, d.get("misc") or {}, d.get("smtp") or {}
    dom = (syn.get("domain") or email.partition("@")[2]).lower()
    r = {
        "input": email,
        "syntax": {"address": syn.get("address") or email, "is_valid_syntax": bool(syn.get("is_valid_syntax")),
                   "username": syn.get("username") or "", "domain": dom, "normalized_email": syn.get("address")},
        "mx": {"accepts_mail": bool(mx.get("accepts_mail")), "records": mx.get("records") or []},
        "misc": {"is_disposable": bool(misc.get("is_disposable")), "is_role_account": bool(misc.get("is_role_account")),
                 "is_free_provider": dom in FREE_PROVIDERS},
        "smtp": {k: bool(smtp.get(k)) for k in ("can_connect_smtp", "has_full_inbox", "is_catch_all", "is_deliverable", "is_disabled")},
        "smtp_checked": True,
        "source": "reacher",
    }
    reachable = d.get("is_reachable") or "unknown"
    _, reasons = _verdict({**r, "smtp": {**r["smtp"], "rejected": reachable == "invalid" and r["smtp"]["can_connect_smtp"]}})
    return {**r, "is_reachable": reachable, "reasons": reasons}


async def verify_email(email: str, smtp: Optional[bool] = None) -> Dict[str, Any]:
    """Check one address. `smtp=False` skips the mail-server step (fast: syntax + DNS only)."""
    email = (email or "").strip()
    now = datetime.now(timezone.utc).isoformat()
    want_smtp = SMTP_ENABLED if smtp is None else (smtp and SMTP_ENABLED)
    if REACHER_URL and want_smtp:
        try:
            return {**await _via_reacher(email), "checked_at": now}
        except Exception as e:  # fall back to the built-in checks
            print(f"[EMAIL VERIFY] Reacher at {REACHER_URL} failed: {e}")

    syn = _syntax(email)
    dom = syn["domain"]
    local = (syn.get("username") or "").lower()
    r: Dict[str, Any] = {
        "input": email, "syntax": syn,
        "mx": {"accepts_mail": False, "records": []},
        "misc": {"is_disposable": dom in DISPOSABLE, "is_role_account": local in ROLE_ACCOUNTS or local.split("+")[0] in ROLE_ACCOUNTS,
                 "is_free_provider": dom in FREE_PROVIDERS},
        "smtp": None, "smtp_checked": False, "source": "builtin",
    }
    if syn["is_valid_syntax"]:
        hosts, err = await _mx(dom)
        r["mx"] = {"accepts_mail": bool(hosts), "records": hosts, **({"error": err} if err else {})}
        if hosts and want_smtp:
            r["smtp"] = await asyncio.to_thread(_smtp_probe, hosts, syn["normalized_email"] or email, dom)
            r["smtp_checked"] = True
    reachable, reasons = _verdict(r)
    return {**r, "is_reachable": reachable, "reasons": reasons, "checked_at": now}
