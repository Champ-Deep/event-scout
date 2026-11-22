"""Run: python3 -m unittest tests.test_scan_analysis"""
import unittest
from unittest import mock

import scan_analysis as sa

PROFILE = {
    "company_name": "Lake B2B", "products": ["event lead data"], "value_propositions": ["verified contacts"],
    "target_roles": ["VP Marketing", "Head of Demand Generation"], "target_industries": ["SaaS", "fintech"],
    "event_goals": ["book demos"], "current_event_name": "Bangalore Tech Summit",
}
HIT = {"name": "Asha Rao", "email": "asha@payco.io", "phone": "+91 99999 11111", "linkedin": "N/A",
       "company_name": "PayCo fintech", "notes": "VP Marketing, wants verified contacts, book demos"}
MISS = {"name": "Joe", "email": "N/A", "phone": "N/A", "linkedin": "N/A", "company_name": "N/A", "notes": ""}


class ScanAnalysis(unittest.TestCase):
    def setUp(self):  # no Ollama in tests: force the rules path
        p = mock.patch.object(sa, "local_reasoning", return_value=None)
        p.start()
        self.addCleanup(p.stop)

    def test_good_fit_scores_higher_and_quotes_user_inputs(self):
        hit, miss = sa.analyze(HIT, PROFILE), sa.analyze(MISS, PROFILE)
        self.assertGreater(hit["score"], miss["score"])
        self.assertEqual(hit["temperature"], "hot")
        for needle in ("Lake B2B", "VP Marketing", "fintech", "Bangalore Tech Summit"):
            self.assertIn(needle, hit["reasoning"])
        self.assertEqual(hit["engine"], "rules")

    def test_reasoning_changes_with_the_user(self):
        other = dict(PROFILE, company_name="Acme", target_roles=["CTO"], target_industries=["retail"], products=["crm"])
        self.assertNotEqual(sa.analyze(HIT, PROFILE)["reasoning"], sa.analyze(HIT, other)["reasoning"])
        self.assertIn("Acme", sa.analyze(HIT, other)["reasoning"])

    def test_empty_profile_and_empty_card_do_not_crash(self):
        r = sa.analyze(MISS, {})
        self.assertTrue(0 <= r["score"] <= 100)
        self.assertIn("no products", r["reasoning"])

    def test_local_model_text_wins_when_available(self):
        with mock.patch.object(sa, "local_reasoning", return_value="Custom text."):
            r = sa.analyze(HIT, PROFILE)
        self.assertEqual(r["reasoning"], "Custom text.")
        self.assertTrue(r["engine"].startswith("ollama:"))


if __name__ == "__main__":
    unittest.main()
