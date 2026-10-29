import contextlib
import io
import os
import smtplib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import inboxkit_send as ik  # noqa: E402

MAILBOX, TO = "sender@example.com", "me@example.org"
OK = {"error": False, "message": "ok", "password": "pw", "secret": "s", "app_password": "abcd efgh ijkl mnop"}


class FakeSMTP:
    """Stands in for smtplib.SMTP_SSL; `fail` is a list of exceptions raised by send_message in order."""
    instances = []

    def __init__(self, fail=()):
        self.fail, self.logins, self.sent = list(fail), [], []

    def factory(self, host, port, timeout):
        FakeSMTP.instances.append((host, port))
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def login(self, user, pw):
        self.logins.append((user, pw))

    def send_message(self, msg):
        if self.fail:
            raise self.fail.pop(0)
        self.sent.append(msg)


def run_send(handler, smtp=None, ledger=None, key=None):
    smtp = smtp or FakeSMTP()
    msg = ik.build_message(MAILBOX, TO, "Hi", "Body", idempotency_key=key)
    sleeps = []
    mid = ik.send(msg, api_key="test-key", workspace_id="ws1", ledger=ledger,
                  transport=httpx.MockTransport(handler), smtp_factory=smtp.factory, sleep=sleeps.append)
    return mid, smtp, sleeps


class TestSend(unittest.TestCase):
    def test_happy_path(self):
        seen = []

        def handler(req):
            seen.append(req)
            return httpx.Response(200, json=OK)
        mid, smtp, _ = run_send(handler)
        self.assertEqual(seen[0].headers["Authorization"], "Bearer test-key")
        self.assertEqual(seen[0].headers["X-Workspace-Id"], "ws1")
        self.assertEqual(seen[0].url.params["email"], MAILBOX)
        self.assertEqual(smtp.logins, [(MAILBOX, "abcdefghijklmnop")])  # spaces stripped
        self.assertEqual(smtp.sent[0]["Message-ID"], mid)

    def test_auth_failure_not_retried(self):
        for code in (401, 403):
            calls = []

            def handler(req, code=code):
                calls.append(1)
                return httpx.Response(code, json={"error": True, "message": "no"})
            with self.assertRaises(ik.AuthError):
                run_send(handler)
            self.assertEqual(len(calls), 1)

    def test_rate_limit_retries_then_succeeds(self):
        replies = [httpx.Response(429), httpx.Response(429), httpx.Response(200, json=OK)]
        _, smtp, sleeps = run_send(lambda req: replies.pop(0))
        self.assertEqual(len(sleeps), 2)
        self.assertTrue(all(s > 0 for s in sleeps))
        self.assertEqual(len(smtp.sent), 1)

    def test_rate_limit_gives_up(self):
        with self.assertRaises(ik.RateLimitError):
            run_send(lambda req: httpx.Response(429))

    def test_validation_error_not_retried(self):
        calls = []

        def handler(req):
            calls.append(1)
            return httpx.Response(400, json={"error": True, "message": "Mailbox is not active yet."})
        with self.assertRaises(ik.ValidationError):
            run_send(handler)
        self.assertEqual(len(calls), 1)

    def test_bad_input_fails_before_network(self):
        with self.assertRaises(ik.ValidationError):
            ik.build_message("not-an-address", TO, "s", "b")
        with self.assertRaises(ik.ValidationError):
            ik.build_message(MAILBOX, TO, " ", "b")

    def test_smtp_temp_reply_retries_permanent_does_not(self):
        smtp = FakeSMTP(fail=[smtplib.SMTPResponseException(451, b"try later")])
        _, smtp, sleeps = run_send(lambda r: httpx.Response(200, json=OK), smtp)
        self.assertEqual((len(sleeps), len(smtp.sent)), (1, 1))
        smtp = FakeSMTP(fail=[smtplib.SMTPResponseException(550, b"no such user")])
        with self.assertRaises(ik.ValidationError):
            run_send(lambda r: httpx.Response(200, json=OK), smtp)

    def test_ledger_blocks_resend(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = Path(d) / "sent.json"
            _, smtp1, _ = run_send(lambda r: httpx.Response(200, json=OK), ledger=ledger, key="k1")
            _, smtp2, _ = run_send(lambda r: httpx.Response(200, json=OK), ledger=ledger, key="k1")
            self.assertEqual((len(smtp1.sent), len(smtp2.sent)), (1, 0))

    def test_secrets_not_logged(self):
        with self.assertLogs("inboxkit_send", level="INFO") as cm:
            run_send(lambda r: httpx.Response(200, json=OK))
        text = "\n".join(cm.output)
        self.assertIn("message_id=", text)
        for secret in ("test-key", "abcdefghijklmnop", "abcd efgh"):
            self.assertNotIn(secret, text)


class TestCli(unittest.TestCase):
    ARGS = ["--from", MAILBOX, "--to", TO, "--subject", "Hi", "--body", "Body", "--idempotency-key", "k"]

    def test_dry_run_needs_no_key_and_no_network(self):
        out = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch("httpx.Client", side_effect=AssertionError("network")), \
                mock.patch("smtplib.SMTP_SSL", side_effect=AssertionError("network")), \
                contextlib.redirect_stdout(out):
            self.assertEqual(ik.main(self.ARGS + ["--dry-run"]), 0)
        msg = ik.build_message(MAILBOX, TO, "Hi", "Body", idempotency_key="k")
        self.assertIn(f"show-credentials?email={MAILBOX}", out.getvalue())
        self.assertIn(f"Message-ID: {msg['Message-ID']}", out.getvalue())
        self.assertIn("<INBOXKIT_API_KEY>", out.getvalue())

    def test_live_without_key_fails_cleanly(self):
        err = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(err):
            self.assertEqual(ik.main(self.ARGS), 1)
        self.assertIn("AuthError", err.getvalue())


if __name__ == "__main__":
    unittest.main()
