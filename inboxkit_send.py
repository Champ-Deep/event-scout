"""InboxKit send-out for Event Scout. Built from Celsus docs/inboxkit-api-contract.md.

InboxKit has no send endpoint (contract section 0). The path is:
  1. GET /v1/api/mailboxes/show-credentials?email=<from>  (Bearer INBOXKIT_API_KEY, X-Workspace-Id)
  2. SMTP submit as that mailbox with its app_password
  3. our own Message-ID is the idempotency handle (contract section 7); a local ledger stops resends

Env: INBOXKIT_API_KEY, INBOXKIT_WORKSPACE_ID. Optional: INBOXKIT_SMTP_HOST (default smtp.gmail.com),
INBOXKIT_SMTP_PORT (default 465, implicit TLS).

CLI:  python inboxkit_send.py --from a@x.com --to b@y.com --subject S --body B [--dry-run]
Live sends are never run by tests. Open contract items (U1 to U12) are not resolved here.

shortcut: no attachments (contract has no attachment field), Google host default only, one recipient.
Upgrade when Deep answers U2/U11 or a Microsoft mailbox is needed.
"""
import argparse
import email.utils
import hashlib
import json
import logging
import os
import random
import re
import smtplib
import sys
import time
from email.message import EmailMessage
from pathlib import Path

import httpx

log = logging.getLogger("inboxkit_send")

BASE_URL = "https://api.inboxkit.com/v1/api"  # not .../v1, that 404s (contract section 1)
_ADDR = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_ATTEMPTS = 4


class InboxKitError(Exception):
    """Base class. `retryable` marks transient failures only."""
    retryable = False


class AuthError(InboxKitError):
    """401/403 from InboxKit, or SMTP auth refused."""


class RateLimitError(InboxKitError):
    retryable = True


class TransientError(InboxKitError):
    """5xx, network failure, or a temporary (4xx) SMTP reply."""
    retryable = True


class ValidationError(InboxKitError):
    """Bad input, other 4xx from InboxKit (incl. 'Mailbox is not active yet'), permanent SMTP reject."""


def _check_addr(label: str, value: str) -> str:
    addr = email.utils.parseaddr(value or "")[1]
    if not _ADDR.match(addr):
        raise ValidationError(f"invalid {label} address: {value!r}")
    return addr


def build_message(sender: str, to: str, subject: str, body: str, html: str | None = None,
                  reply_to: str | None = None, idempotency_key: str | None = None) -> EmailMessage:
    """Validate and build the exact message that SMTP will carry."""
    sender, to = _check_addr("from", sender), _check_addr("to", to)
    if not (subject or "").strip() or not (body or "").strip():
        raise ValidationError("subject and body are required")
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = sender, to, subject
    if reply_to:
        msg["Reply-To"] = _check_addr("reply-to", reply_to)
    msg["Date"] = email.utils.formatdate(localtime=False)
    domain = sender.split("@", 1)[1]
    if idempotency_key:  # same key, same Message-ID, so a retry is recognisable downstream
        msg["Message-ID"] = f"<{hashlib.sha256(idempotency_key.encode()).hexdigest()[:32]}@{domain}>"
    else:
        msg["Message-ID"] = email.utils.make_msgid(domain=domain)
    msg.set_content(body)
    if html:
        msg.add_alternative(html, subtype="html")
    return msg


def _raise_for_status(resp: httpx.Response) -> None:
    code = resp.status_code
    if code < 400:
        return
    try:
        detail = resp.json().get("message", "")
    except Exception:
        detail = ""
    text = f"InboxKit HTTP {code}: {detail}"  # never include headers or body fields
    if code in (401, 403):
        raise AuthError(text)
    if code == 429:
        raise RateLimitError(text)
    if code >= 500:
        raise TransientError(text)
    raise ValidationError(text)


def _retry(fn, sleep=time.sleep):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn()
        except InboxKitError as e:
            if not e.retryable or attempt == MAX_ATTEMPTS:
                raise
            delay = min(2 ** attempt, 30) * random.uniform(0.5, 1.0)  # exponential with jitter
            log.warning("transient failure (%s), retry %d in %.1fs", e, attempt, delay)
            sleep(delay)


def fetch_app_password(client: httpx.Client, mailbox: str, sleep=time.sleep) -> str:
    def call() -> str:
        try:
            resp = client.get(f"{BASE_URL}/mailboxes/show-credentials", params={"email": mailbox})
        except httpx.TransportError as e:
            raise TransientError(f"network error: {type(e).__name__}") from e
        _raise_for_status(resp)
        pw = (resp.json().get("app_password") or "").replace(" ", "")
        if not pw:
            raise ValidationError("InboxKit returned no app_password for this mailbox")
        return pw
    return _retry(call, sleep)


def _smtp_submit(msg, mailbox, password, host, port, smtp_factory, sleep):
    def call():
        try:
            with smtp_factory(host, port, timeout=30) as s:
                s.login(mailbox, password)
                s.send_message(msg)
        except smtplib.SMTPAuthenticationError as e:
            raise AuthError(f"SMTP auth refused ({e.smtp_code})") from e
        except (smtplib.SMTPConnectError, smtplib.SMTPHeloError, ConnectionRefusedError) as e:
            raise TransientError(f"SMTP connect failed: {type(e).__name__}") from e  # nothing sent yet
        except smtplib.SMTPResponseException as e:
            if 400 <= e.smtp_code < 500:
                raise TransientError(f"SMTP temporary reply {e.smtp_code}") from e
            raise ValidationError(f"SMTP rejected {e.smtp_code}") from e
        except smtplib.SMTPRecipientsRefused as e:
            raise ValidationError("SMTP recipient refused") from e
        # shortcut: timeouts and mid-DATA disconnects are not retried, they may have delivered.
    _retry(call, sleep)


def send(msg: EmailMessage, *, api_key: str, workspace_id: str, host: str = "smtp.gmail.com",
         port: int = 465, ledger: Path | None = None, transport: httpx.BaseTransport | None = None,
         smtp_factory=smtplib.SMTP_SSL, sleep=time.sleep) -> str:
    """Send one message; return its Message-ID. Skips the send if the ID is already in the ledger."""
    mid, mailbox = msg["Message-ID"], msg["From"]
    sent = set(json.loads(ledger.read_text())) if ledger and ledger.exists() else set()
    if mid in sent:
        log.info("already sent, skipping", extra={"message_id": mid})
        return mid
    headers = {"Authorization": f"Bearer {api_key}", "X-Workspace-Id": workspace_id}
    with httpx.Client(headers=headers, timeout=20, transport=transport) as client:
        password = fetch_app_password(client, mailbox, sleep)
    _smtp_submit(msg, mailbox, password, host, port, smtp_factory, sleep)
    if ledger:
        ledger.write_text(json.dumps(sorted(sent | {mid})))
    log.info("sent message_id=%s from=%s", mid, mailbox)
    return mid


def dry_run_report(msg: EmailMessage, workspace_id: str, host: str, port: int) -> str:
    """Everything a live call would send, with secrets replaced by placeholders."""
    return "\n".join([
        f"GET {BASE_URL}/mailboxes/show-credentials?email={msg['From']}",
        "Authorization: Bearer <INBOXKIT_API_KEY>",
        f"X-Workspace-Id: {workspace_id or '<INBOXKIT_WORKSPACE_ID>'}",
        f"SMTP {host}:{port} login={msg['From']} password=<app_password from the call above>",
        "--- message ---",
        msg.as_string(),
    ])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Send one test email via an InboxKit mailbox.")
    p.add_argument("--from", dest="sender", required=True)
    p.add_argument("--to", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--body", required=True)
    p.add_argument("--html")
    p.add_argument("--reply-to")
    p.add_argument("--idempotency-key")
    p.add_argument("--ledger", type=Path, help="JSON file of sent Message-IDs, stops resends")
    p.add_argument("--dry-run", action="store_true", help="validate and print, no network")
    a = p.parse_args(argv)
    host = os.environ.get("INBOXKIT_SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("INBOXKIT_SMTP_PORT", "465"))
    ws = os.environ.get("INBOXKIT_WORKSPACE_ID", "")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        msg = build_message(a.sender, a.to, a.subject, a.body, a.html, a.reply_to, a.idempotency_key)
        if a.dry_run:
            print(dry_run_report(msg, ws, host, port))
            return 0
        key = os.environ.get("INBOXKIT_API_KEY")
        if not key or not ws:
            raise AuthError("INBOXKIT_API_KEY and INBOXKIT_WORKSPACE_ID must be set")
        print(send(msg, api_key=key, workspace_id=ws, host=host, port=port, ledger=a.ledger))
        return 0
    except InboxKitError as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
