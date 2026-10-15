# Deploying EventScout

Three parts: the **API** (`apps/backend`, Docker), the **admin console** (`apps/admin`, static site), and the **mobile app** (`apps/mobile`, EAS build).

## 1. API (Railway or any Docker host)

`apps/backend/Dockerfile` + `railway.json` (health check `/api/health`, one replica, one worker; the live feed lives in the process).

Set these variables. The server refuses to start in production without the first three.

| Variable | Value |
|---|---|
| `DATABASE_URL` | PostgreSQL URL (Railway Postgres plugin). Never SQLite in a container: its disk is wiped on redeploy. |
| `JWT_SECRET` | 32+ random characters: `python -c "import secrets; print(secrets.token_urlsafe(48))"`. Changing it signs everyone out. |
| `APP_API_KEY` | 16+ random characters; the mobile build's `EXPO_PUBLIC_API_KEY` must be the same value. |
| `OPENROUTER_API_KEY` | Card reading, scoring, Ask AI. |
| `PUBLIC_BASE_URL` | The API's public https address, e.g. `https://api.example.com` (share links, card pages). |
| `CORS_ORIGINS` | The admin console origin, e.g. `https://admin.example.com`. |
| optional | `SMTP_*` (forgot password), `LINKEDIN_*`, `BACKUP_DATABASE_URL`, CRM keys, `WEBHOOK_URL` + `ENRICH_WEBHOOK_SECRET`. See `apps/backend/.env.example`. |

LinkedIn sign-in: in the LinkedIn developer app, add both redirect URLs: `https://<admin or web origin>/callback` (web) and `https://api.example.com/auth/linkedin/app-return` (the installed app).

The image sets `ENVIRONMENT=production`. Self-hosting instead: `docker compose up -d --build` in `apps/backend` (needs `POSTGRES_PASSWORD` in `.env`).

**First admin.** Sign-ups are never made admin automatically. Sign up in the app or console, then run:

```
railway run python manage_admins.py grant you@company.com   # or: docker compose exec app python manage_admins.py grant …
```

Then sign in again. Further admins can be promoted on the console's Team page.

## 2. Admin console (Vercel / Netlify / any static host)

```
cd apps/admin
VITE_API_URL=https://api.example.com pnpm build      # origin only, no /api
```

Deploy `apps/admin/dist`. The SPA fallback is already included: `vercel.json` for Vercel, `public/_redirects` for Netlify. On another host, serve `index.html` for unknown paths. Add the console's origin to the API's `CORS_ORIGINS`.

## 3. Mobile app (EAS)

The repo isn't a git repository, so run EAS with `EAS_NO_VCS=1` from the monorepo root (`.easignore` keeps the backend, admin and `.env` files out of the upload).

Set the build variables once (EAS dashboard → Environment variables, or `eas env:create`) for the `production` and `preview` environments:

- `EXPO_PUBLIC_API_BASE_URL=https://api.example.com` (https is required: release builds block plain http)
- `EXPO_PUBLIC_API_KEY=<APP_API_KEY>`

```
cd apps/mobile
eas build --profile preview --platform android      # installable APK for testing
eas build --profile production --platform all       # store builds (version code auto-increments)
```

A production build stops with a clear error if the API URL isn't https or the key is missing.

## Checks before deploying

```
cd apps/backend && python -m pip install -r requirements-dev.txt && python -m pytest tests_unit   # backend unit tests
cd apps/admin   && npx tsc --noEmit -p . && npx vitest run && npx vite build                      # needs Node 22
cd apps/mobile  && npx tsc --noEmit && npx vitest run && npx expo export --platform android
```

## After deploying

- Open `https://api.example.com/api/health`: it should report `healthy` with the database connected.
- Sign in to the console, open System: API, database and AI should all be green.
- On a phone: sign up, scan a card, then confirm it appears live in the console.
