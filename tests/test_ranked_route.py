"""Exercise the exact app.py route through FastAPI without AI/DB startup.

The DB boundary is mocked. Route source, SQL clauses, validation and ranker are real.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
import uuid

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.testclient import TestClient
from sqlalchemy import column, select, table

from exhibitor_ranking import rank_exhibitors
from tests.test_exhibitor_ranking import PROFILE, SEED

ROOT = Path(__file__).resolve().parents[1]
USER = "00000000-0000-0000-0000-000000000001"


class RankedRouteTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.session = AsyncMock()
        self.exhibitors = table("exhibitors", column("event_name"))
        self.exhibitors.event_name = self.exhibitors.c.event_name
        self.profiles = table("user_profiles", column("user_id"))
        self.profiles.user_id = self.profiles.c.user_id
        self.rows = [SimpleNamespace(**{
            "booth": "", "hall": "", "subcategory": "", "description": "",
            "tags": [], **row,
        }) for row in SEED]
        self.set_results()

        def verify_api_key():
            raise HTTPException(status_code=401)

        self.auth = verify_api_key
        namespace = dict(app=self.app, uuid=uuid, Query=Query, Depends=Depends,
                         verify_api_key=self.auth, get_db_session=AsyncMock(return_value=self.session),
                         select=select, ExhibitorDB=self.exhibitors, UserProfileDB=self.profiles,
                         rank_exhibitors=rank_exhibitors)
        tree = ast.parse((ROOT / "app.py").read_text())
        route = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                     and n.name == "ranked_exhibitors_route")
        exec(compile(ast.Module(body=[route], type_ignores=[]), "app.py", "exec"), namespace)
        self.app.dependency_overrides[self.auth] = lambda: "test-only"
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def set_results(self, profile=PROFILE):
        rows = Mock()
        rows.scalars.return_value.all.return_value = self.rows
        user = Mock()
        user.scalar_one_or_none.return_value = SimpleNamespace(profile_data=profile) if profile is not None else None
        self.session.execute.side_effect = [rows, user]

    def get(self, **params):
        return self.client.get("/exhibitors/ranked/", params={"user_id": USER, **params})

    def test_seed_order_and_profile_event_queries(self):
        response = self.get(event="Test event")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        expected = rank_exhibitors(SEED, PROFILE)
        self.assertEqual([x["id"] for x in result["items"]], [x["id"] for x in expected["items"]])
        self.assertEqual([x["score"] for x in result["items"]], [x["score"] for x in expected["items"]])
        self.assertEqual((result["status"], result["total"], result["has_more"]), ("success", 120, True))
        self.assertTrue(all(row["reason"] for row in result["items"]))
        queries = [call.args[0].compile().params for call in self.session.execute.call_args_list]
        self.assertEqual(queries, [{"event_name_1": "Test event"}, {"user_id_1": uuid.UUID(USER)}])
        self.session.close.assert_awaited_once()
        print("GET /exhibitors/ranked/ ->", response.status_code,
              "total:", result["total"], "scores:", [x["score"] for x in result["items"]])

    def test_filters_pagination_and_missing_profile(self):
        self.set_results(None)
        result = self.get(search="Exhibitor", category="Imaging,Furniture", offset=100, limit=20).json()
        expected = rank_exhibitors(SEED, {}, q="Exhibitor", categories=["Imaging", "Furniture"], offset=100, limit=20)
        self.assertEqual([x["id"] for x in result["items"]], [x["id"] for x in expected["items"]])
        self.assertFalse(result["has_more"])

    def test_validation_and_auth(self):
        for params in ({"offset": -1}, {"limit": 0}, {"limit": 101}, {"user_id": "not-a-uuid"}):
            self.assertEqual(self.get(**params).status_code, 422)
        self.app.dependency_overrides.clear()
        self.assertEqual(self.get().status_code, 401)
        self.session.execute.assert_not_awaited()

    def test_session_closes_on_failure(self):
        self.session.execute.side_effect = RuntimeError("DB unavailable")
        self.assertEqual(self.get().status_code, 500)
        self.session.close.assert_awaited_once()

    def test_preview_fixture_is_exact_ranker_output(self):
        fixture = json.loads((ROOT / "tests/fixtures/scout-ranked.json").read_text())
        expected = (rank_exhibitors(SEED, PROFILE, limit=100)["items"]
                    + rank_exhibitors(SEED, PROFILE, offset=100, limit=100)["items"])
        self.assertEqual(fixture, expected)
        self.assertEqual(len(fixture), 120)


if __name__ == "__main__":
    unittest.main()
