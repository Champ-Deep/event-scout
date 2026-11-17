// Local check of Better Auth against a seeded Postgres (see test/seed_legacy.py).
// Run: TEST_DATABASE_URL=... node --test test/auth.local.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createAuth, loadAppUser } from "../src/auth.ts";

const url = process.env.TEST_DATABASE_URL!;
const seeded = Object.fromEntries(
  execFileSync("python3", ["test/seed_legacy.py", url], { encoding: "utf8" })
    .trim().split("\n").map((l) => l.split(" ") as [string, string]),
);
execFileSync("psql", [url, "-q", "-v", "ON_ERROR_STOP=1", "-f", "migrations/001_better_auth.sql"]);

const make = () => createAuth({ connectionString: url, secret: "x".repeat(40), baseURL: "http://localhost" });

async function signIn(email: string, password: string) {
  const { auth, pool } = make();
  try {
    const res = await auth.api.signInEmail({ body: { email, password }, asResponse: true });
    return { status: res.status, body: await res.json(), cookie: res.headers.get("set-cookie") };
  } finally {
    await pool.end();
  }
}

test("old user signs in with old password, same id, typed email in any case", async () => {
  const r = await signIn("champ@example.com", "old-pass-1");
  assert.equal(r.status, 200);
  assert.equal(r.body.user.id, seeded["Champ@Example.com"]);
  assert.equal((await signIn("CHAMP@example.com", "old-pass-1")).status, 200);
});

test("wrong password is refused", async () => {
  assert.equal((await signIn("champ@example.com", "nope-nope")).status, 401);
});

test("password over 72 bytes matches app.py truncation", async () => {
  assert.equal((await signIn("long@example.com", "é".repeat(40) + "tail")).status, 200);
  assert.equal((await signIn("long@example.com", "é".repeat(36) + "different")).status, 200); // first 72 bytes equal
});

test("case-duplicate email goes to the account with more contacts", async () => {
  const r = await signIn("dup@example.com", "dup-b");
  assert.equal(r.body.user.id, seeded["dup@example.com"]);
});

test("session cookie resolves to the app user and its contacts", async () => {
  const r = await signIn("champ@example.com", "old-pass-1");
  const { auth, pool } = make();
  try {
    const session = await auth.api.getSession({ headers: new Headers({ cookie: r.cookie!.split(";")[0] }) });
    assert.equal(session?.user.id, seeded["Champ@Example.com"]);
    const me = await loadAppUser(pool, session!.user.id, ["champ@example.com"]);
    assert.equal(me?.is_admin, true);
    const { rows } = await pool.query("SELECT count(*)::int AS n FROM contacts WHERE user_id = $1::uuid", [me!.user_id]);
    assert.equal(rows[0].n, 3);
  } finally {
    await pool.end();
  }
});

test("new sign-up gets a users row with a bcrypt hash the old app accepts", async () => {
  const { auth, pool } = make();
  try {
    const res = await auth.api.signUpEmail({ body: { name: "New", email: "New@Example.com", password: "brand-new-1" } });
    const { rows } = await pool.query("SELECT email, password_hash FROM users WHERE id = $1::uuid", [res.user.id]);
    assert.equal(rows.length, 1);
    const ok = execFileSync("python3", ["-c",
      "import bcrypt,sys;print(bcrypt.checkpw(b'brand-new-1', sys.argv[1].encode()))", rows[0].password_hash],
      { encoding: "utf8" }).trim();
    assert.equal(ok, "True");
    await pool.query("INSERT INTO contacts (user_id, name) VALUES ($1::uuid, 'x')", [res.user.id]);
  } finally {
    await pool.end();
  }
});
