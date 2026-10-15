from datetime import timedelta

import pytest
from fastapi import HTTPException

import identity
from conftest import make_user
from database import UserDB


def bearer(uid, tv=0, ttl=timedelta(hours=1), kind="access"):
    return "Bearer " + identity.create_jwt_token({"sub": str(uid), "userId": str(uid), "type": kind, "tv": tv}, ttl)


async def expect(status, coro, detail=None):
    with pytest.raises(HTTPException) as err:
        await coro
    assert err.value.status_code == status, err.value.detail
    if detail is not None:
        assert err.value.detail == detail
    return err.value


async def test_rep_admin_and_console_view(db):
    rep = await make_user(db, "Rep")
    admin = await make_user(db, "Admin", admin=True)

    me = await identity.acting(bearer(rep))
    assert me.user_id == rep and me.owner == rep and not me.as_admin
    console = await identity.acting(bearer(admin), console=True)
    assert console.as_admin and console.owner is None           # the console works on everyone's data
    assert (await identity.acting(bearer(admin))).owner == admin  # an admin in the app is just a rep
    await expect(403, identity.acting(bearer(rep), console=True), "Admin access required")
    assert await identity.admin_user(bearer(admin)) == admin
    await expect(403, identity.admin_user(bearer(rep)))
    await expect(401, identity.admin_user(None), "Admin sign-in required")
    await expect(401, identity.acting(None), "Sign in required")


async def test_revoked_expired_and_bad_tokens(db):
    uid = await make_user(db, token_version=2)
    assert await identity.signed_in_user(bearer(uid, tv=2)) == uid
    await expect(401, identity.acting(bearer(uid, tv=1)), identity.SESSION_REVOKED_DETAIL)          # signed out everywhere
    expired = bearer(uid, tv=2, ttl=timedelta(seconds=-5))
    await expect(401, identity.acting(expired), identity.TOKEN_EXPIRED_DETAIL)                      # the app refreshes
    assert await identity.signed_in_user(expired, verify_exp=False) == uid                          # /auth/session
    await expect(401, identity.acting(bearer(uid, tv=2, kind="refresh")), "Invalid authentication token")
    forged = bearer(uid, tv=2).rsplit(".", 1)[0] + ".AAAA"
    await expect(401, identity.acting(forged), identity.SESSION_REVOKED_DETAIL)
    async with db() as s:  # a deleted account's tokens are dead too
        await s.delete(await s.get(UserDB, uid))
        await s.commit()
    await expect(401, identity.acting(bearer(uid, tv=2)), identity.SESSION_REVOKED_DETAIL)


async def test_legacy_user_id_must_be_the_token_owner(db):
    rep = await make_user(db, "Rep")
    other = await make_user(db, "Other")
    admin = await make_user(db, "Admin", admin=True)
    assert await identity.legacy_user(bearer(rep), str(rep)) == rep
    await expect(403, identity.legacy_user(bearer(rep), str(other)), identity.OTHER_ACCOUNT_DETAIL)
    # admins don't act as another rep through the rep routes either (they use /api with X-Admin-Console)
    await expect(403, identity.legacy_user(bearer(admin), str(rep)), identity.OTHER_ACCOUNT_DETAIL)
    await expect(401, identity.legacy_user(None, str(rep)), "Sign in required")
    await expect(400, identity.legacy_user(bearer(rep), "not-a-uuid"), "Invalid user_id")
    await expect(401, identity.legacy_user("Bearer garbage", str(rep)), identity.SESSION_REVOKED_DETAIL)
    # API-key routes without a user_id: a presented token must still be current; junk is ignored
    await identity.check_presented_token("Bearer garbage")
    await expect(401, identity.check_presented_token(bearer(rep, ttl=timedelta(seconds=-5))), identity.TOKEN_EXPIRED_DETAIL)


async def test_data_owner_and_rep_only_routes(db):
    rep = await make_user(db)
    assert await identity.data_owner(rep, None) == rep
    assert await identity.data_owner(None, str(rep)) == rep
    await expect(401, identity.data_owner(None, None))
    await expect(400, identity.data_owner(None, "nope"))
    async with db() as s:
        await expect(404, identity.data_owner(None, "00000000-0000-0000-0000-000000000001", session=s))
    assert identity.require_rep(rep) == rep
    with pytest.raises(HTTPException):
        identity.require_rep(None)


async def test_stream_tickets_are_single_use(db):
    admin = await make_user(db, admin=True)
    ticket = identity.issue_stream_ticket(admin)
    assert await identity.stream_admin(None, ticket, None) == admin
    await expect(401, identity.stream_admin(None, ticket, None), "Live feed ticket expired")
    token = bearer(admin).split(" ", 1)[1]
    assert await identity.stream_admin(None, None, token) == admin  # older consoles: ?access_token=
