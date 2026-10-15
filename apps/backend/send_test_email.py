"""Check the password-reset email settings in .env by sending one test email.

    python send_test_email.py you@yourdomain.com

Uses the same SMTP_* settings and TLS handling as the server (app.py), without starting it, and
says in plain words what went wrong (wrong password, certificate, port, ...). Prints no secrets.
"""
import smtplib
import socket
import ssl
import sys
from email.message import EmailMessage

from dotenv import load_dotenv

load_dotenv()

import os  # noqa: E402  (after load_dotenv, like app.py)

HOST = os.environ.get("SMTP_HOST", "")
PORT = int(os.environ.get("SMTP_PORT", "587"))
USERNAME = os.environ.get("SMTP_USERNAME", "")
PASSWORD = os.environ.get("SMTP_PASSWORD", "")
FROM = os.environ.get("SMTP_FROM", "") or USERNAME
USE_SSL = os.environ.get("SMTP_USE_SSL", "false").lower() == "true"
CA_FILE = os.environ.get("SMTP_CA_FILE", "").strip()
TLS_VERIFY = os.environ.get("SMTP_TLS_VERIFY", "true").lower() != "false"


def tls_context() -> ssl.SSLContext:
    # same as app.py _smtp_tls_context
    context = ssl.create_default_context(cafile=CA_FILE or None)
    if not TLS_VERIFY:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    return context


def main() -> int:
    if len(sys.argv) != 2 or "@" not in sys.argv[1]:
        print("Usage: python send_test_email.py you@yourdomain.com")
        return 2
    to = sys.argv[1]
    if not HOST or not FROM:
        print("Not configured: set SMTP_HOST and SMTP_FROM (or SMTP_USERNAME) in apps/backend/.env")
        return 1
    print(f"Server {HOST}:{PORT} · {'SSL' if USE_SSL else 'STARTTLS'} · login {'yes' if USERNAME else 'no'}"
          f" · certificate check {'off' if not TLS_VERIFY else 'custom CA' if CA_FILE else 'on'}")

    msg = EmailMessage()
    msg["Subject"] = "EventScout test email"
    msg["From"] = FROM
    msg["To"] = to
    msg.set_content("This is a test from EventScout. Password reset emails will be sent the same way.")

    try:
        context = tls_context()
        if USE_SSL:
            server = smtplib.SMTP_SSL(HOST, PORT, context=context, timeout=20)
        else:
            server = smtplib.SMTP(HOST, PORT, timeout=20)
            server.starttls(context=context)
        with server:
            if USERNAME:
                server.login(USERNAME, PASSWORD)
            server.send_message(msg)
    except ssl.SSLCertVerificationError as e:
        print(f"FAILED: the server's certificate isn't trusted ({e.verify_message}).")
        print("  Set SMTP_CA_FILE to its certificate, or SMTP_TLS_VERIFY=false for a trusted internal server.")
        return 1
    except smtplib.SMTPAuthenticationError:
        print("FAILED: the mail server rejected SMTP_USERNAME / SMTP_PASSWORD.")
        return 1
    except smtplib.SMTPSenderRefused:
        print(f"FAILED: the server won't send as {FROM}. Use an address that mailbox may send from (SMTP_FROM).")
        return 1
    except smtplib.SMTPRecipientsRefused:
        print(f"FAILED: the server refused the recipient {to}.")
        return 1
    except (socket.timeout, TimeoutError, ConnectionRefusedError, OSError) as e:
        print(f"FAILED: couldn't connect to {HOST}:{PORT} ({e}). Check SMTP_HOST / SMTP_PORT, or a firewall.")
        return 1
    except smtplib.SMTPException as e:
        print(f"FAILED: {e}")
        return 1
    print(f"Sent. Check the inbox (and spam) of {to}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
