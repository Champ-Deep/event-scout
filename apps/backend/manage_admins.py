"""Grant, revoke or list admin access to the EventScout admin console.

Sign-ups are never made admin automatically (there is no email verification), so the first admin of
a new deployment is created here, against the same DATABASE_URL the server uses:

    python manage_admins.py list
    python manage_admins.py grant you@company.com
    python manage_admins.py revoke someone@company.com

On Railway: `railway run python manage_admins.py grant you@company.com`. The person signs up in the
app (or console) first, then is granted admin, then signs in again. Further admins can be promoted from
the console's Team page.
"""
import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import func, select  # noqa: E402

from database import UserDB, dispose_engines, get_session_factory, init_db  # noqa: E402


async def main(argv: list) -> int:
    if not argv or argv[0] not in ("list", "grant", "revoke") or (argv[0] != "list" and len(argv) != 2):
        print(__doc__)
        return 2
    await init_db()
    factory = get_session_factory()
    if factory is None:
        print("No database configured (DATABASE_URL).")
        return 1
    async with factory() as session:
        if argv[0] == "list":
            admins = (await session.execute(select(UserDB).where(UserDB.is_admin == True).order_by(UserDB.email))).scalars().all()  # noqa: E712
            for u in admins:
                print(f"{u.email}\t{u.name}")
            print(f"{len(admins)} admin(s)")
            return 0
        email = argv[1].strip().lower()
        user = (await session.execute(select(UserDB).where(func.lower(UserDB.email) == email))).scalars().first()
        if not user:
            print(f"No account with the email {email}. Sign up first, then run this again.")
            return 1
        make_admin = argv[0] == "grant"
        if bool(user.is_admin) == make_admin:
            print(f"{user.email} is {'already' if make_admin else 'not'} an admin.")
            return 0
        user.is_admin = make_admin
        user.token_version = (user.token_version or 0) + 1  # their current sign-in carries the old role
        await session.commit()
        print(f"{user.email} is {'now an admin' if make_admin else 'no longer an admin'}. They need to sign in again.")
        return 0


async def run(argv: list) -> int:
    try:
        return await main(argv)
    finally:
        await dispose_engines()


if __name__ == "__main__":
    sys.exit(asyncio.run(run(sys.argv[1:])))
