"""Contract tests: every protected endpoint requires X-API-Key.

We don't reach the route handler bodies here — verify_api_key (a
FastAPI Depends) runs before the handler, so the DB never gets touched
and these tests work without a live database. Tests live in unit/
because they're hermetic (TestClient against the in-process app).
"""
from __future__ import annotations

import pytest


# Endpoints that MUST reject requests without a valid X-API-Key.
# Format: (method, path, json_body, query_string)
# Use representative paths from across the app; this is not exhaustive
# but covers each broad category (CRUD, AI, admin, pipeline, files,
# webhooks, user card, exhibitors, broadcasts).
PROTECTED_ENDPOINTS = [
    ("GET", "/list_contacts/", None, "?user_id=00000000-0000-0000-0000-000000000000"),
    ("GET", "/contact/00000000-0000-0000-0000-000000000000", None, "?user_id=00000000-0000-0000-0000-000000000000"),
    ("DELETE", "/contact/00000000-0000-0000-0000-000000000000", None, "?user_id=00000000-0000-0000-0000-000000000000"),
    ("POST", "/add_contact/", {"contact": {"name": "x", "email": "x", "phone": "x", "linkedin": "x"}, "user_id": "00000000-0000-0000-0000-000000000000"}, ""),
    ("POST", "/search/", {"query": "x", "user_id": "00000000-0000-0000-0000-000000000000"}, ""),
    ("POST", "/converse/", {"query": "x", "user_id": "00000000-0000-0000-0000-000000000000"}, ""),
    ("POST", "/generate_qr/", {"name": "x", "email": "x", "phone": "x", "linkedin": "x"}, ""),
    ("GET", "/export_contacts/", None, "?user_id=00000000-0000-0000-0000-000000000000&format=csv"),
    ("GET", "/user/profile/", None, "?user_id=00000000-0000-0000-0000-000000000000"),
    ("PUT", "/user/profile/", {}, "?user_id=00000000-0000-0000-0000-000000000000"),
    ("GET", "/dashboard", None, "?user_id=00000000-0000-0000-0000-000000000000"),
    ("POST", "/contact/00000000-0000-0000-0000-000000000000/score", None, "?user_id=00000000-0000-0000-0000-000000000000"),
    ("GET", "/contact/00000000-0000-0000-0000-000000000000/pipeline", None, "?user_id=00000000-0000-0000-0000-000000000000"),
    ("POST", "/contact/00000000-0000-0000-0000-000000000000/pipeline/run", None, "?user_id=00000000-0000-0000-0000-000000000000"),
    ("GET", "/exhibitors/", None, ""),
    ("GET", "/admin/users", None, "?user_id=00000000-0000-0000-0000-000000000000"),
    ("GET", "/admin/contacts", None, "?user_id=00000000-0000-0000-0000-000000000000"),
    ("GET", "/admin/pipelines", None, "?user_id=00000000-0000-0000-0000-000000000000"),
    ("POST", "/admin/broadcast", {"message": "x"}, "?user_id=00000000-0000-0000-0000-000000000000"),
]


# Endpoints that DO NOT require X-API-Key.
PUBLIC_ENDPOINTS = [
    ("GET", "/"),
    ("GET", "/health/"),
]


@pytest.mark.parametrize("method,path,body,query", PROTECTED_ENDPOINTS)
def test_protected_endpoint_rejects_missing_api_key(test_client, method, path, body, query):
    url = path + query
    if method == "GET":
        r = test_client.get(url)
    elif method == "POST":
        r = test_client.post(url, json=body or {})
    elif method == "PUT":
        r = test_client.put(url, json=body or {})
    elif method == "DELETE":
        r = test_client.delete(url)
    else:
        pytest.fail(f"Unsupported method: {method}")

    assert r.status_code == 401, (
        f"{method} {url} returned {r.status_code} without X-API-Key — "
        f"expected 401. Body: {r.text[:200]}"
    )


@pytest.mark.parametrize("method,path,body,query", PROTECTED_ENDPOINTS)
def test_protected_endpoint_rejects_wrong_api_key(test_client, method, path, body, query):
    url = path + query
    headers = {"X-API-Key": "wrong-key-definitely-not-valid"}
    if method == "GET":
        r = test_client.get(url, headers=headers)
    elif method == "POST":
        r = test_client.post(url, json=body or {}, headers=headers)
    elif method == "PUT":
        r = test_client.put(url, json=body or {}, headers=headers)
    elif method == "DELETE":
        r = test_client.delete(url, headers=headers)
    else:
        pytest.fail(f"Unsupported method: {method}")

    assert r.status_code == 401, (
        f"{method} {url} returned {r.status_code} with WRONG X-API-Key — "
        f"expected 401. Body: {r.text[:200]}"
    )


@pytest.mark.parametrize("method,path", PUBLIC_ENDPOINTS)
def test_public_endpoint_does_not_require_api_key(test_client, method, path):
    if method == "GET":
        r = test_client.get(path)
    else:
        pytest.fail(f"Unsupported method: {method}")
    # Anything except 401/403 is acceptable; the endpoint may still
    # 500 because the DB isn't connected, but it must not be auth-gated.
    assert r.status_code not in (401, 403), (
        f"{method} {path} required auth but should be public — got {r.status_code}"
    )


def test_every_app_route_with_db_dependency_uses_auth_gate(app_module):
    """Belt-and-braces: walk the FastAPI route table and assert that
    any route whose path is not in the public allowlist depends on
    verify_api_key (directly or via verify_admin).

    This catches the most likely future regression: adding a new
    endpoint and forgetting `api_key: str = Depends(verify_api_key)`.
    """
    from fastapi.routing import APIRoute

    PUBLIC_PATHS = {
        "/",
        "/health/",
        "/debug",
        "/register/",
        "/login/",
        "/user/validate",
        "/models/",
        "/card/{token}",  # public card view by token
        "/openapi.json",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
    }

    unauthenticated: list[str] = []
    for route in app_module.app.routes:
        if not isinstance(route, APIRoute):
            continue
        if route.path in PUBLIC_PATHS:
            continue
        # Collect every dependency callable for this route, including
        # nested sub-dependencies (verify_admin depends on verify_api_key).
        deps_seen: set = set()

        def _walk(dep):
            if dep is None or dep.call in deps_seen:
                return
            deps_seen.add(dep.call)
            for sub in dep.dependencies or []:
                _walk(sub)

        for dep in route.dependant.dependencies or []:
            _walk(dep)

        uses_auth = (
            app_module.verify_api_key in deps_seen
            or app_module.verify_admin in deps_seen
        )
        if not uses_auth:
            unauthenticated.append(f"{','.join(route.methods)} {route.path}")

    assert not unauthenticated, (
        "Routes missing verify_api_key/verify_admin dependency:\n  "
        + "\n  ".join(unauthenticated)
        + "\nIf intentional, add the path to PUBLIC_PATHS in this test."
    )
