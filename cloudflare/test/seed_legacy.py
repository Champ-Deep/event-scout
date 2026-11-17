"""Seed a throwaway Postgres with old-app users (bcrypt hashes made by Python, like app.py)."""
import subprocess, sys, uuid, bcrypt

url = sys.argv[1]
def hp(p):  # app.py hash_password
    return bcrypt.hashpw(p.encode()[:72], bcrypt.gensalt()).decode()

users = [
    (str(uuid.uuid4()), "Champ", "Champ@Example.com", hp("old-pass-1"), 3),
    (str(uuid.uuid4()), "Long", "long@example.com", hp("é" * 40 + "tail"), 1),
    (str(uuid.uuid4()), "Dup Upper", "DUP@example.com", hp("dup-a"), 0),
    (str(uuid.uuid4()), "Dup Lower", "dup@example.com", hp("dup-b"), 2),
]
sql = ["DROP TABLE IF EXISTS contacts, ba_session, ba_account, ba_verification, ba_user, users CASCADE;",
       "CREATE TABLE users (id uuid PRIMARY KEY, name varchar(255) NOT NULL, email varchar(255) UNIQUE NOT NULL,"
       " password_hash varchar(255) NOT NULL, is_admin boolean NOT NULL DEFAULT false, created_at timestamptz);",
       "CREATE TABLE contacts (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), user_id uuid NOT NULL REFERENCES users(id), name varchar(255) NOT NULL);"]
for uid, name, email, h, n in users:
    sql.append(f"INSERT INTO users VALUES ('{uid}','{name}','{email}','{h}',false,now());")
    sql += [f"INSERT INTO contacts (user_id,name) VALUES ('{uid}','c{i}');" for i in range(n)]
subprocess.run(["psql", url, "-q", "-v", "ON_ERROR_STOP=1", "-c", "\n".join(sql)], check=True)
print("\n".join(f"{u[2]} {u[0]}" for u in users))
