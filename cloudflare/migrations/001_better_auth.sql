-- Better Auth tables for the Cloudflare deploy, plus a backfill from the app's users table.
-- Safe to run again: tables are created only if missing and the backfill skips existing rows.
-- Run against the Cloudflare database only, never Railway.

CREATE TABLE IF NOT EXISTS ba_user (
  "id" text PRIMARY KEY,
  "name" text NOT NULL,
  "email" text NOT NULL UNIQUE,
  "emailVerified" boolean NOT NULL DEFAULT false,
  "image" text,
  "createdAt" timestamptz NOT NULL DEFAULT now(),
  "updatedAt" timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ba_session (
  "id" text PRIMARY KEY,
  "expiresAt" timestamptz NOT NULL,
  "token" text NOT NULL UNIQUE,
  "createdAt" timestamptz NOT NULL DEFAULT now(),
  "updatedAt" timestamptz NOT NULL DEFAULT now(),
  "ipAddress" text,
  "userAgent" text,
  "userId" text NOT NULL REFERENCES ba_user ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ba_session_user_idx ON ba_session ("userId");

CREATE TABLE IF NOT EXISTS ba_account (
  "id" text PRIMARY KEY,
  "accountId" text NOT NULL,
  "providerId" text NOT NULL,
  "userId" text NOT NULL REFERENCES ba_user ("id") ON DELETE CASCADE,
  "accessToken" text,
  "refreshToken" text,
  "idToken" text,
  "accessTokenExpiresAt" timestamptz,
  "refreshTokenExpiresAt" timestamptz,
  "scope" text,
  "password" text,
  "createdAt" timestamptz NOT NULL DEFAULT now(),
  "updatedAt" timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ba_account_user_idx ON ba_account ("userId");

CREATE TABLE IF NOT EXISTS ba_verification (
  "id" text PRIMARY KEY,
  "identifier" text NOT NULL,
  "value" text NOT NULL,
  "expiresAt" timestamptz NOT NULL,
  "createdAt" timestamptz NOT NULL DEFAULT now(),
  "updatedAt" timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ba_verification_identifier_idx ON ba_verification ("identifier");

-- Backfill. Better Auth matches emails in lower case, while the old app stored them as typed.
-- If two old accounts differ only by case, the one with more contacts keeps the email; the
-- other is listed by the check query at the end and can still be reached on Railway.
WITH ranked AS (
  SELECT u.*, row_number() OVER (
           PARTITION BY lower(trim(u.email))
           ORDER BY (SELECT count(*) FROM contacts c WHERE c.user_id = u.id) DESC, u.created_at
         ) AS rn
  FROM users u
)
INSERT INTO ba_user ("id", "name", "email", "emailVerified", "createdAt", "updatedAt")
SELECT r.id::text, r.name, lower(trim(r.email)), true,
       coalesce(r.created_at, now()), coalesce(r.created_at, now())
FROM ranked r
WHERE r.rn = 1
ON CONFLICT DO NOTHING;

INSERT INTO ba_account ("id", "accountId", "providerId", "userId", "password", "createdAt", "updatedAt")
SELECT 'legacy-' || u.id::text, u.id::text, 'credential', u.id::text, u.password_hash, now(), now()
FROM users u
JOIN ba_user b ON b."id" = u.id::text
WHERE NOT EXISTS (  -- skip people who signed up on Cloudflare; they already have a login
  SELECT 1 FROM ba_account a
  WHERE a."userId" = u.id::text AND a."providerId" = 'credential' AND a."id" NOT LIKE 'legacy-%'
)
ON CONFLICT ("id") DO UPDATE SET "password" = EXCLUDED."password", "updatedAt" = now()
  WHERE ba_account."password" IS DISTINCT FROM EXCLUDED."password";

-- Check: old accounts that did not get a Better Auth login (expected: none).
SELECT u.email AS not_migrated
FROM users u LEFT JOIN ba_user b ON b."id" = u.id::text
WHERE b."id" IS NULL;
