import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exhibitor_ranking import rank_exhibitors, score_exhibitor  # noqa: E402

PROFILE = {"target_industries": ["imaging"]}


def ex(i, name, category="", country="", **kw):
    return {"id": str(i), "name": name, "category": category, "country": country, **kw}


# 120 seeded rows: country cycles so geography is varied; every 7th row matches the profile.
SEED = [
    ex(i, f"Exhibitor {i:03d}", "Medical Imaging" if i % 7 == 0 else "Furniture",
       country=["UAE", "India", "Germany", "USA"][i % 4],
       website="https://x.test" if i % 3 == 0 else "", products=["p"] if i % 5 == 0 else [])
    for i in range(120)
]


class TestRanking(unittest.TestCase):
    def test_order_is_strictly_by_score_and_ignores_country(self):
        items = rank_exhibitors(SEED, PROFILE, limit=100)["items"]
        scores = [e["score"] for e in items]
        self.assertEqual(scores, sorted(scores, reverse=True))
        a = score_exhibitor(ex(1, "A", "Medical Imaging", country="UAE"), PROFILE)
        b = score_exhibitor(ex(1, "A", "Medical Imaging", country="Japan"), PROFILE)
        self.assertEqual(a, b)

    def test_every_row_has_nonempty_reason_even_with_no_profile_or_data(self):
        for e in rank_exhibitors(SEED + [{}], {}, limit=100)["items"]:
            self.assertTrue(e["reason"].strip())
        self.assertTrue(score_exhibitor({}, None)["reason"])

    def test_pages_are_contiguous_slices_with_stable_ties(self):
        full = [e["id"] for e in rank_exhibitors(SEED, PROFILE, limit=100)["items"]]
        p1 = rank_exhibitors(SEED, PROFILE, offset=0, limit=50)
        p2 = rank_exhibitors(SEED, PROFILE, offset=50, limit=50)
        self.assertEqual([e["id"] for e in p1["items"] + p2["items"]], full)
        self.assertTrue(p1["has_more"] and p2["has_more"])
        self.assertFalse(rank_exhibitors(SEED, PROFILE, offset=100, limit=50)["has_more"])

    def test_filters_apply_before_slicing_and_no_results_is_empty(self):
        r = rank_exhibitors(SEED, PROFILE, categories=["imaging"], limit=100)
        self.assertEqual(r["total"], len([e for e in SEED if e["category"] == "Medical Imaging"]))
        self.assertTrue(all(e["category"] == "Medical Imaging" for e in r["items"]))
        none = rank_exhibitors(SEED, PROFILE, q="zzzz")
        self.assertEqual((none["items"], none["total"], none["has_more"]), ([], 0, False))


if __name__ == "__main__":
    unittest.main()
