# Event Scout V2 consolidation, 2026-11-22

Branch `v2/consolidated-2026-11`, draft PR #24 into `cf/native`. Nothing was merged, closed, deleted or deployed.

## Decision first

Merge this draft with a **merge commit, not a squash**. A squash would make the 9 branches it carries look unmerged and drop their history, and the branch deletions below depend on them being ancestors of `cf/native`.

Production does not change. `git diff origin/cf/release-2026-11-22 v2/consolidated-2026-11 -- worker web` is empty: the Cloudflare Worker and the new web UI are byte identical to what runs today. V2 adds only the FastAPI archive work, the pytest suite, and CI.

One product call is left to Deep and is not in V2: the **beta monorepo lineage** (6 branches, 4 open PRs: #11, #15, #17, #19). It replaces the FastAPI app with `apps/backend`, `apps/admin` (React console) and `apps/mobile` (Expo). See the section on it.

## What was wrong

- `cf/native` had no CI at all. There was no workflow on the base, so a draft PR ran nothing.
- 9 branches held work that was never merged anywhere, spread across two incompatible lineages (FastAPI on `master`, and the beta monorepo).
- The pytest suite on `claude/analyze-test-coverage-Bj8LB` and the CI file for it were never merged, so the 196 Python tests never ran on a runner.
- A test-isolation bug surfaced once those tests met the carried ones: `tests/test_export_csv.py` replaced `app.get_db_session` with a fake and never restored it, so `tests/test_scan_idempotency.py` (alphabetically later) got a fake session and failed 4 of 4.

## What V2 contains (base plus 6 merges plus 1 fix commit)

Base: `origin/cf/release-2026-11-22` (2c7936c), which already holds cf/native, #21, #22 and #23.

| Merge | Carries |
|---|---|
| `cf/native` | 6ec77d1 docs(harbinger): the key is Harbinger's own API key; ChampIQ is retired (the only commit cf/native had that the base lacked) |
| `v2/consolidated-rebase` | admin-dashboard spec and Admin API (admin_api.py: leads list/export/detail, reassign, reps, team lead, persisted settings), Hi-Fi scan-to-score flow, deterministic lead scoring, exhibitor ranking, inboxkit_send.py, gitleaks ignore, vercel.json fix, export CSV test fix. Also makes `v2/consolidated`, `admin-dashboard/api`, `admin-dashboard/spec`, `security/gate-backlog-2026-10-29` and `fix/vercel-json-public` ancestors |
| `feat/scan-idempotency` | ES-API-1: Idempotency-Key on POST /add_contact_from_image/ (client_scan_id plus partial unique index), 4 tests that need Postgres |
| `feat/prod-smoke-script` | production smoke script for card scanning |
| `claude/analyze-test-coverage-Bj8LB` | pytest infrastructure, auth contract and multi-user isolation tests, helper unit tests, `requirements-dev.txt`, `pytest.ini` |
| commit "ci: ..." | `.github/workflows/test.yml` rewritten as 4 package jobs plus a gate; `tests/test_export_csv.py` fixture now restores `get_db_session` |

Conflicts: none of the 6 merges had a textual conflict (checked with `git merge-tree` before merging and by the merges themselves). Semantic seam found by the suite and fixed in the test fixture, not in an assertion: the unrestored `get_db_session` fake above.

## Verification (commands named)

Fresh clone of the pushed branch (e633cb5, `git clone -b v2/consolidated-2026-11`), on this Mac:

| Check | Command | Result |
|---|---|---|
| Python, full suite | `pytest -m "not live"` with `TEST_DATABASE_URL` set to a throwaway Postgres 17 container | 195 passed, 1 failed, 6 deselected |
| The one failure | `tests/unit/test_qr_round_trip.py` | environmental: this Mac has no libzbar (`ImportError: Unable to find zbar shared library`). The runner installs libzbar0 and the test passes there |
| Worker typecheck | `cd worker && npm ci && npm run typecheck` | exit 0 |
| Web tests | `cd web && npm ci && npm test` | 14 passed (vitest, 1 file) |
| Web build | `cd web && npm run build` | exit 0 |

Baseline before any merge, same commands on the base: web 14 passed, worker typecheck clean, pytest 23 passed (the 3 test files the base had; the rest did not exist yet).

Provider runners, PR #24, run 45007968731 on the code commit: **CI gate pass, Python tests pass (196 passed, 6 deselected, 0 skipped, with libzbar and Postgres service), Worker typecheck pass, Web tests and build pass, Docker image builds pass (2m24s).** The report commit that follows this file triggers a second run; its result is in the PR checks.

Test titles: 136 unique `def test_` titles across the base, master, v2/consolidated-rebase, scan-idempotency, test-coverage and prod-smoke trees; 0 are missing from V2 (`comm -23`). Web has one test file, unchanged.

Secret scan: `gitleaks detect --log-opts=origin/cf/release-2026-11-22..HEAD`, 13 commits, no leaks.

Container: `docker build -t event-scout:v2test .` from the fresh clone (exit 0), then `docker run` against a throwaway Postgres and `curl`: `GET /health/` returned HTTP 200 (`database_connected:false`, the probe does not test the connection), `GET /exhibitors/ranked/` without a key returned 401, `GET /admin/leads` without a token returned 401. No new route answered without credentials.

Not verified: the 6 deselected tests are `live` tests that need a deployed Railway URL. The Worker smoke tests were not run (see the end).

## Verdict table (every remote branch)

Proof column: A = `git merge-base --is-ancestor origin/<branch> origin/cf/release-2026-11-22` true. B = same test against V2, true after the merges. Same root commit (552fc6a) on all 33 branches, so none is a different project.

| Branch | Tip | PR | Verdict | Proof |
|---|---|---|---|---|
| master | 6bbef28 | default | CONTAINED | A |
| cf/release-2026-11-22 | 2c7936c | base | CONTAINED (it is the base) | A |
| cf/review-fixes | f7f51c4 | #21 | CONTAINED | A |
| cf/admin-xss-escape | 639b19e | #22 | CONTAINED | A |
| cf/ui-v1 | 33bcb89 | #23 | CONTAINED | A |
| chore/land-selective-exports | b9a5ac7 | #13 merged | CONTAINED | A |
| claude/fix-recent-issue-8uvNh | 57cfec9 | #3, #4 merged | CONTAINED | A |
| feat/harbinger-csv-export | 31f58d4 | #7 merged | CONTAINED | A |
| feature/enhancements | ce1f85e | #1, #2 merged | CONTAINED | A |
| feature/event-intelligence-v2 | 086bcb8 | #5 merged | CONTAINED | A |
| fix/docker-copy-export-csv | 749b824 | #16 merged | CONTAINED | A |
| fix/ocr-scan-pipeline | c43a595 | #9 merged | CONTAINED | A |
| fix/scan-temp-file-collision | 2b0be82 | #20 merged | CONTAINED | A |
| fix/twa-assetlinks | 98d8caf | #10 merged | CONTAINED | A |
| harsha/seawind-scout-setup | 51dd395 | none | CONTAINED | A |
| cf/native | 6ec77d1 | PR base | UNIQUE WORK carried: 1 commit (6ec77d1, docs) | B |
| v2/consolidated-rebase | 4c87e69 | none | UNIQUE WORK carried: 7 non-merge commits (listed above) | B |
| v2/consolidated | 47c0d3e | #14 | CARRIED, and superseded by the rebase; `git range-diff` shows the same patches | B |
| admin-dashboard/api | 601b385 | none | CARRIED (inside the rebase) | B |
| admin-dashboard/spec | f34f575 | none | CARRIED (inside the rebase) | B |
| security/gate-backlog-2026-10-29 | 32f8575 | #8 | CARRIED (inside the rebase) | B |
| fix/vercel-json-public | 001dd73 | #12 merged | CARRIED (inside the rebase) | B |
| feat/scan-idempotency | b6d98f1 | none | UNIQUE WORK carried: b6d98f1 | B |
| feat/prod-smoke-script | aaf8241 | #18 | UNIQUE WORK carried: aaf8241 | B |
| claude/analyze-test-coverage-Bj8LB | 8f28cfe | none | UNIQUE WORK carried: 8f28cfe, c579dca | B |
| claude/eventscout-timeouts-dedup-ru3jx6 | 5857e46 | none | UNIQUE WORK, NOT carried: ee228e5 conflicts with the Hi-Fi scan flow in `mobile-frontend.html` (two redesigns of one screen, a product call). 5857e46 is patch-equivalent to the vercel fix already in V2 (`git cherry`). Port candidate below | `git merge-tree` conflict, `git cherry` |
| feature/cloudflare-backend | da3395b | none | SUPERSEDED by the D1 Worker in `worker/` (same job, other design: `cloudflare/` Worker plus Container plus Postgres plus Better Auth, worker name `event-scout-cf-test`, versus `worker/wrangler.jsonc` name `event-scout`, D1 `event-scout-db`). Conflicts on `mobile-frontend.html`. Its workflow `cloudflare-test.yml` deploys the test copy and needs secrets, so it is not carried | `worker/wrangler.jsonc`, `worker/migrations/0001_init.sql` |
| beta-version | d3bab47 | none | UNIQUE WORK, NOT carried (product call) | conflict map below |
| validate/beta-build | 86850f9 | none | same | same |
| feat/admin-export-presets | 524a308 | #11 | same | same |
| feat/events-exhibitors-admin-tab | 28424fa | #19 | same | same |
| feat/event-scrape-wire | d22cdec | #15 | same (18 commits: event scraper, floor map) | same |
| fix/event-scrape-sqlite-race | 9698836 | #17 | same (the 18 above plus 9698836) | same |

Open PRs: #21, #22, #23 CONTAINED in the base. #8, #14, #18 CARRIED. #11, #15, #17, #19 NOT carried. #24 is this draft. Merged PRs #1 to #5, #7, #9, #10, #12, #13, #16, #20 have branches that are ancestors (true ancestry, not squash inference).

## The beta monorepo lineage (not carried, needs Deep)

`git merge-tree --write-tree HEAD origin/<branch>` for any of the 6 beta branches reports:
- modify/delete: `app.py`, `Dockerfile`, `database.py`, `card-viewer.html`, `mobile-frontend.html`, `vercel.json` (the beta moved them to `apps/backend/` and deleted the originals, while V2 modified them)
- content: `.gitignore`
- file location: `tests/test_dockerfile_modules.py`, `tests/test_export_csv.py`, `tests/test_scan_temp_path.py` would move to `apps/backend/tests/`
- `feat/events-exhibitors-admin-tab` also conflicts on `vercel.json` content.

`beta-version` alone is 452 files changed (103,581 insertions) against master. Taking it means adopting `apps/admin` and `apps/mobile` and a second FastAPI tree next to the Cloudflare Worker. That is a direction choice, so V2 leaves it out. All 6 branches stay.

What the lineage holds that the Worker lacks: the event website scraper and floor map (`event_scraper.py`, `event_floormap.py`, `event_scrape_service.py` with tests), the Events and Exhibitors admin tab, export column picker and saved presets, the admin console and Expo app.

## Port candidates (behaviour missing from `worker/`, not ported here)

Checked by grep over `worker/src` and `web/src`:
1. Admin leads API: reassign, reps, team lead, persisted settings (`admin_api.py`). Worker admin.ts has users, contacts, merge, lists, broadcast, export, but none of these.
2. Deterministic lead scoring (`lead_scoring.py`). The Worker scores with a model call (`scoreOne` in `ai.ts`).
3. Ranked exhibitors (`/exhibitors/ranked/`, `exhibitor_ranking.py`). Worker lists exhibitors only.
4. Scan idempotency (`Idempotency-Key`, `client_scan_id`). `web/src/lib/scan-queue.ts` queues offline scans but sends no key.
5. Duplicate upsert on rescan, case-insensitive email and name plus company match (`ee228e5`).
6. Event scraper and floor map, export column picker and presets (beta lineage).
7. `inboxkit_send.py` is carried but wired to nothing (no importer, not in the Dockerfile).

Hi-Fi scan-to-score on `mobile-frontend.html` is SUPERSEDED in production by `web/src/surfaces/Scan.tsx` (#23); it is carried only because it rides in the rebase.

## What changes on the base when V2 lands

- Cloudflare Worker, D1, web UI: nothing (`worker/` and `web/` diff is empty). No D1 migration is added; I did not run `wrangler d1 migrations list`, and none is needed for V2.
- New CI: one workflow, 4 jobs and a gate, on pushes to master, main, cf/native and PRs into master, main, cf/**, feature/**, claude/**.
- FastAPI archive app (Railway, only if redeployed): new routes `POST /scan/extract`, `POST /scan/analyze`, `GET /exhibitors/ranked/` behind `verify_api_key`, and the `admin_api` router behind `verify_admin`. No new unauthenticated route. Schema self-migrates at startup (`ALTER TABLE ... IF NOT EXISTS`): `contacts.client_scan_id` plus a partial unique index, `users.team_lead_id`, table `app_settings`. Dockerfile gains 4 COPY lines (admin_api, exhibitor_ranking, lead_scoring, scan_analysis).
- Legacy `mobile-frontend.html` (Vercel) gets the Hi-Fi scan flow. I did not check what the Vercel project serves today.
- Public repo: `docs/scan-to-score-screens/` (9 PNGs) is carried from #14. I opened all 9: synthetic data only (Asha Rao, PayCo, +91 99999 11111). `tests/fixtures/scout-ranked.json` is synthetic ("Exhibitor 000"). No card photos, no contact data. `.gitleaksignore` entries reference beta paths that V2 does not contain; harmless.
- Secrets: `feature/cloudflare-backend` carries a workflow whose comment embeds the old frontend API key (not reproduced here). The key is already public in that branch and in the legacy frontend, so deleting the branch does not remove it. Rotate if it is still live. Not touched.

## Unpushed local work (not on any remote, not published, not in the 32)

The shared checkout has 33 non-merge commits on local-only branches: `feat/event-publish-toggle` (10), `v2/hifi-integration` (9), `integration/admin-user-split` (8), `test/split-e2e` (5), `feat/event-scrape-schedule` (2), `feat/user-surface` (2), `docs/t_258d4bee-exhibitor-agent-brief` (2), and one each on `feat/admin-surface`, `feat/event-floormap`, `feat/offline-scan-queue`, `fix/deck-attach-path`, `docs/events-exhibitors-default`, `docs/hifi-surfaces-spec`, plus `backup/*`. Most belong to the beta lineage. `feat/offline-scan-queue` (idempotent retries) and `fix/deck-attach-path` (Gamma PPTX download) look independent. They need a push before any local clone is retired.

## What I did not do, and why

- Did not carry the beta lineage: product call, see above.
- Did not carry `claude/eventscout-timeouts-dedup-ru3jx6` or `feature/cloudflare-backend`: conflict and supersession as in the table.
- Did not port any FastAPI behaviour into `worker/`: out of scope for a rescue.
- Did not wire `inboxkit_send.py` into the Dockerfile or app.
- Did not run the Worker smoke tests (`worker/test/*.mjs` need a running `wrangler dev` with D1) or any deploy.

## Suggested merge order

1. Merge PR #24 into `cf/native` with a merge commit.
2. Close PRs #8, #14, #18 (carried) and #21, #22, #23 (contained; GitHub may mark them merged by itself).
3. Delete the 22 branches in the decision card.
4. Decide the beta lineage separately (PRs #11, #15, #17, #19 stay open until then).
