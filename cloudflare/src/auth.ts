// Better Auth for the Cloudflare deploy.
//
// Existing users keep their email and password: the ba_* tables are backfilled from `users`
// (see migrations/001_better_auth.sql) with ba_user.id = users.id, so every contact row
// (contacts.user_id -> users.id) stays attached to the same person. Old passwords are bcrypt
// hashes from the FastAPI app and are verified as such; new passwords are bcrypt too, so the
// FastAPI /login/ would still accept them if we ever roll back.
import { betterAuth } from "better-auth";
import bcrypt from "bcryptjs";
import { Pool } from "pg";

export interface AuthConfig {
  connectionString: string;
  secret: string;
  baseURL: string;
}

// bcryptjs reads only the first 72 UTF-8 bytes, the same as app.py's explicit truncation.
const BCRYPT_ROUNDS = 12;

export async function hashPassword(password: string): Promise<string> {
  return bcrypt.hash(password, BCRYPT_ROUNDS);
}

export async function verifyPassword({ hash, password }: { hash: string; password: string }): Promise<boolean> {
  if (!hash || !hash.startsWith("$2")) return false;
  return bcrypt.compare(password, hash);
}

export function createAuth(config: AuthConfig) {
  // One small pool per request: Workers cannot share sockets across requests.
  const pool = new Pool({ connectionString: config.connectionString, max: 2 });

  const auth = betterAuth({
    database: pool,
    secret: config.secret,
    baseURL: config.baseURL,
    basePath: "/api/auth",
    user: { modelName: "ba_user" },
    session: {
      modelName: "ba_session",
      expiresIn: 60 * 60 * 24 * 30, // 30 days: nobody should be logged out mid-event
      updateAge: 60 * 60 * 24,
      // Signed session cookie, re-checked against the database every 5 minutes, so most API calls
      // skip a database round trip.
      cookieCache: { enabled: true, maxAge: 5 * 60 },
    },
    account: { modelName: "ba_account" },
    verification: { modelName: "ba_verification" },
    emailAndPassword: {
      enabled: true,
      autoSignIn: true,
      minPasswordLength: 6,
      password: { hash: hashPassword, verify: verifyPassword },
    },
    advanced: {
      database: { generateId: () => crypto.randomUUID() },
    },
    databaseHooks: {
      account: {
        create: {
          // A new sign-up also gets a row in the app's own users table, with the same id and
          // password hash, so contacts can reference it.
          after: async (account) => {
            if (account.providerId !== "credential") return;
            await pool.query(
              `INSERT INTO users (id, name, email, password_hash, is_admin, created_at)
               SELECT u.id::uuid, u.name, u.email, $2, false, now() FROM ba_user u WHERE u.id = $1
               ON CONFLICT (id) DO NOTHING`,
              [account.userId, account.password ?? "!"],
            );
          },
        },
      },
    },
  });

  return { auth, pool };
}

export interface AppUser {
  user_id: string;
  name: string;
  email: string;
  is_admin: boolean;
}

// The app's own view of the signed-in user, with the same shape the old /login/ returned.
export async function loadAppUser(pool: Pool, userId: string, adminEmails: string[]): Promise<AppUser | null> {
  const { rows } = await pool.query<{ id: string; name: string; email: string; is_admin: boolean }>(
    "SELECT id::text AS id, name, email, is_admin FROM users WHERE id = $1::uuid",
    [userId],
  );
  const row = rows[0];
  if (!row) return null;
  let isAdmin = row.is_admin;
  if (!isAdmin && adminEmails.includes(row.email.trim().toLowerCase())) {
    await pool.query("UPDATE users SET is_admin = true WHERE id = $1::uuid", [userId]);
    isAdmin = true;
  }
  return { user_id: row.id, name: row.name, email: row.email, is_admin: isAdmin };
}
