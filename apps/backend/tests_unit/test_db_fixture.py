"""Concurrent sessions must not share transaction state in the test database."""
from sqlalchemy import select

import database
from conftest import make_user


async def test_closing_reader_does_not_rollback_another_sessions_writes(db):
    uid = await make_user(db)
    async with db() as writer:
        user = await writer.get(database.UserDB, uid)
        async with db() as reader:
            await reader.execute(select(database.UserDB.id))
            user.name = "Updated"
            await writer.flush()
        await writer.commit()
    async with db() as check:
        assert (await check.get(database.UserDB, uid)).name == "Updated"
