"""Opt-in real InboxKit SMTP send test. Not matched by unittest discovery."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import inboxkit_send as ik  # noqa: E402


class TestInboxKitLiveSend(unittest.TestCase):
    @unittest.skipUnless(os.getenv("INBOXKIT_API_KEY"), "INBOXKIT_API_KEY is not set")
    def test_provider_accepts_one_message_to_configured_test_address(self):
        required = ("INBOXKIT_WORKSPACE_ID", "INBOXKIT_TEST_FROM", "INBOXKIT_TEST_TO")
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            self.skipTest("missing " + ", ".join(missing))
        message = ik.build_message(
            os.environ["INBOXKIT_TEST_FROM"],
            os.environ["INBOXKIT_TEST_TO"],
            "InboxKit integration test",
            "This is the single InboxKit integration test message.",
            idempotency_key="inboxkit-send-integration-test",
        )
        message_id = ik.send(
            message,
            api_key=os.environ["INBOXKIT_API_KEY"],
            workspace_id=os.environ["INBOXKIT_WORKSPACE_ID"],
            host=os.getenv("INBOXKIT_SMTP_HOST", "smtp.gmail.com"),
            port=int(os.getenv("INBOXKIT_SMTP_PORT", "465")),
        )
        self.assertEqual(message_id, message["Message-ID"])
        self.assertRegex(message_id, r"^<.+>$")


if __name__ == "__main__":
    unittest.main()
