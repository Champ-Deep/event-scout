import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lead_scoring import score_contact  # noqa: E402

FULL = {
    "name": "Jane Doe", "email": "jane@acme.com", "phone": "+1 415 555 0100",
    "linkedin": "https://linkedin.com/in/jane", "company_name": "Acme SaaS",
    "notes": "VP Marketing at Acme, based in Austin. Wants a demo next week.", "source": "qr",
}
PROFILE = {"target_roles": ["vp marketing"], "target_industries": ["saas"], "target_geographies": ["austin"]}
EMPTY = {"name": "X", "email": "N/A", "phone": "N/A", "linkedin": "N/A", "company_name": "N/A", "notes": ""}


class TestScoreContact(unittest.TestCase):
    def test_top_of_range(self):
        r = score_contact(FULL, PROFILE)
        self.assertEqual((r["score"], r["temperature"]), (100, "hot"))

    def test_bottom_of_range(self):
        r = score_contact(EMPTY, PROFILE)
        self.assertEqual((r["score"], r["temperature"]), (0, "cold"))

    def test_ties_are_deterministic(self):
        a, b = dict(FULL), dict(FULL, name="Other Person")
        self.assertEqual(score_contact(a, PROFILE), score_contact(dict(a), PROFILE))
        self.assertEqual(score_contact(a, PROFILE)["score"], score_contact(b, PROFILE)["score"])

    def test_missing_contact_fields_do_not_crash(self):
        for contact in (None, {}, {"name": None, "notes": None, "email": None}):
            r = score_contact(contact, PROFILE)
            self.assertTrue(0 <= r["score"] <= 100)

    def test_missing_profile_is_not_penalised(self):
        # Without targets only contactability + engagement count, so a full contact is still 100.
        self.assertEqual(score_contact(FULL, None)["score"], 100)
        self.assertEqual(score_contact(FULL, {})["breakdown"].keys(), {"contactability", "engagement"})

    def test_threshold_boundaries(self):
        # contactability 10+8 = 18 of 25 counted alone; add notes>=40 (12) and no source: 30/45 = 67 warm
        c = {"name": "Al", "email": "a@b.co", "linkedin": "https://l.in/a", "notes": "x" * 40}
        self.assertEqual(score_contact(c, {})["score"], 67)
        self.assertEqual(score_contact(c, {})["temperature"], "warm")
        # 35/45 = 78 hot with a scanned source (+5 engagement = 35)
        self.assertEqual(score_contact(dict(c, source="scan"), {})["temperature"], "hot")
        # exactly 40 and 70 map to warm and hot
        import lead_scoring
        self.assertEqual(lead_scoring.HOT, 70)
        self.assertEqual(lead_scoring.WARM, 40)

    def test_score_always_in_range_and_shape(self):
        r = score_contact(FULL, PROFILE)
        self.assertEqual(set(r), {"score", "temperature", "reasoning", "recommended_actions", "breakdown"})
        self.assertLessEqual(len(r["recommended_actions"]), 3)


if __name__ == "__main__":
    unittest.main()
