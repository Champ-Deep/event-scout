# Event Scout V2 consolidation, 2026-11-22

Branch `v2/consolidated-2026-11`, draft PR #24 into `cf/native`. Nothing was merged, closed, deleted or deployed.

## Decision first

Merge this draft with a **merge commit, not a squash**. A squash would make the 9 branches it carries look unmerged and drop their history, and the branch deletions below depend on them being ancestors of `cf/native`.

Merging this PR lands #21, #22 and #23 on `cf/native`. The PR targets `cf/native`, and `git diff --shortstat origin/cf/native v2/consolidated-2026-11 -- worker web` is 43 files changed, 7673 insertions, 1 deletion: that is the card XSS fix (#21), the admin XSS fix (#22) and the new web UI (#23). Per the task brief that content equals the release branch `cf/release-2026-11-22`, which is what production runs; `git diff origin/cf/release-2026-11-22 v2/consolidated-2026-11 -- worker web` is empty. **One exception outside those folders:** the Worker's asset build copies the root `mobile-frontend.html` to `/admin.html`, and V2 adds 319 lines to it (a Scout tab that calls a route the Worker lacks). Nothing changes until the next manual Worker deploy; see "What changes on the base". The rest of the diff against `cf/native` (58 files outside `worker/` and `web/`) is the FastAPI archive work, the pytest suite, and CI.

Does a push to `cf/native` deploy? Nothing in the repo does: `cf/native` has no workflow, and V2 adds only `test.yml`, which has no deploy step. A Cloudflare Workers Builds git trigger for the `event-scout` Worker is **unverified**: reading it needs the Cloudflare dashboard or an API token scope I do not have. Evidence against one: `wrangler deployments list` shows 8 deployments, all authored by deep@championsmail.com with source Upload or Unknown and none from a git build, and the GitHub check runs and statuses on the `cf/native` head and on PR #24 contain no Cloudflare entry. Treat the merge as code-only until Deep confirms the trigger is off in the dashboard.

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

Fresh clone of the pushed branch (e633cb5, `git clone -b v2/consolidated-2026-11`), on this Mac. Since then the branch changed by one test file (`tests/test_scan_idempotency.py`, tmp `QR_DIR`), three deleted PNGs and this report; the Mac figures below were not re-run from a new clone, the runner figures after them were:

| Check | Command | Result |
|---|---|---|
| Python, full suite | `pytest -m "not live"` with `TEST_DATABASE_URL` set to a throwaway Postgres 17 container | 195 passed, 1 failed, 6 deselected |
| The one failure | `tests/unit/test_qr_round_trip.py` | environmental: this Mac has no libzbar (`ImportError: Unable to find zbar shared library`). The runner installs libzbar0 and the test passes there |
| Worker typecheck | `cd worker && npm ci && npm run typecheck` | exit 0 |
| Web tests | `cd web && npm ci && npm test` | 14 passed (vitest, 1 file) |
| Web build | `cd web && npm run build` | exit 0 |

Baseline before any merge, same commands on the base: web 14 passed, worker typecheck clean, pytest 23 passed (the 3 test files the base had; the rest did not exist yet).

Provider runners, PR #24, run 45010952850 on code commit 3f8cd8b (QR cleanup): **CI gate pass, Python tests pass (196 passed, 6 deselected, with libzbar0 and Postgres service), Worker typecheck pass, Web tests and build pass (14 passed), Docker image builds pass.** Earlier runs 45007968731 (e633cb5) and 45009618830 (5dbb356) were green too. The commit that records this adds only this report, so its own run is in the PR checks.

Test titles: 136 unique `def test_` titles across the base, master, v2/consolidated-rebase, scan-idempotency, test-coverage and prod-smoke trees; 0 are missing from V2 (`comm -23`). Web has one test file, unchanged.

Test titles on the branches V2 does not carry, against V2 (`git grep -hE '^[[:space:]]*(async )?def test_' <ref> -- '*.py'`, names stripped of arguments, `sort -u`, `comm -23 <branch> <V2>`; the all-languages column adds `it(` and `test(` titles from `*.ts *.tsx *.mjs *.js` the same way). V2 has 152 Python titles and 166 in all languages:

| Branch | Python titles | Missing from V2 (Python) | All-language titles | Missing from V2 (all) |
|---|---|---|---|---|
| feature/cloudflare-backend | 58 | 4 (`tests/test_ai_models.py`) | 64 | 10 (the 4 plus 6 in `cloudflare/test/auth.local.test.ts`) |
| beta-version | 63 | 33 | 236 | 206 |
| validate/beta-build | 63 | 33 | 236 | 206 |
| feat/events-exhibitors-admin-tab | 63 | 33 | 247 | 217 |
| feat/admin-export-presets | 67 | 37 | 245 | 215 |
| feat/event-scrape-wire | 136 | 106 | 309 | 279 |
| fix/event-scrape-sqlite-race | 137 | 107 | 310 | 280 |
| claude/eventscout-timeouts-dedup-ru3jx6 | 31 | 0 | 31 | 0 |

These titles are not lost: all 7 branches stay in the repo and none is on the delete list. The beta rows are the event scraper, floor map, export presets and the React and Expo apps; they travel with the lineage decision.

Test artifacts: e633cb5 had committed three QR PNGs that `tests/test_scan_idempotency.py` wrote into `saved_qr/` through `app.QR_DIR` (synthetic vCards: Test Person, TestCo, t@example.com). They are removed in their own commit (3f8cd8b) and that test now points `QR_DIR` at `tmp_path`, as `tests/unit/test_qr_round_trip.py` does. The history still holds them: only a squash merge keeps them out of `cf/native`, and a merge commit (the recommended option) carries them. They are synthetic, so this is not a leak, but it is a reason to weigh a squash against the ancestor proofs above. After the fix the 4 idempotency tests pass locally against a throwaway Postgres 17 (`TEST_DATABASE_URL=... pytest tests/test_scan_idempotency.py`, 4 passed) and write no file into `saved_qr/`.

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
| claude/eventscout-timeouts-dedup-ru3jx6 | 5857e46 | none | UNIQUE WORK, NOT carried: ee228e5 is cellular retry and backoff in `mobile-frontend.html` plus a duplicate-scan upsert in `app.py`. `git merge-tree --write-tree --name-only` against V2 conflicts only on `mobile-frontend.html`; `app.py` merges clean. Reason for leaving it out: (a) the HTML half patches a legacy page that nothing serves any more, the Vercel project `event-scout` was deleted on 22 Nov and the old URL returns 404 (card t_36f99fa9, 22 Nov comment), so the retry half has no consumer; (b) the `app.py` half changes what a duplicate scan does on the archive app (it now writes the new OCR fields onto the existing contact) and edits the same duplicate path of `POST /add_contact_from_image/` that b6d98f1 (scan idempotency, carried) guards with `client_scan_id`; making the two agree needs a test of upsert against that path and a product call that a rescan may mutate a saved contact. Both halves are one commit, so carrying only the backend half would ship a response field (`updated_fields`) no client reads. 5857e46 is patch-equivalent to the vercel fix already in V2 (`git cherry`). 0 test titles are missing from V2 (the branch has no tests of its own). Port candidate below | `git merge-tree`, `git cherry` |
| feature/cloudflare-backend | da3395b | none | SUPERSEDED by the D1 Worker in `worker/`, piece by piece. (1) Deploy design (801da99, da3395b): `cloudflare/` Worker plus Container plus Postgres, worker name `event-scout-cf-test`, replaced by `worker/wrangler.jsonc` (name `event-scout`, D1 `event-scout-db`); its workflow `cloudflare-test.yml` deploys the test copy and needs secrets, so it is not carried. (2) Better Auth sign-in with legacy accounts (`cloudflare/src/auth.ts`, `cloudflare/migrations/001_better_auth.sql`, `cloudflare/test/auth.local.test.ts`, 6 TS tests): replaced by the Worker's own per-user sessions in `worker/`, with the sign-up allowlist from t_4f31c564. (3) `ai_models.py` (low-cost model allowlist, `preferred_ai_model` mapped to a cheap default), OpenRouter-first scoring in `app.py` and `tests/test_ai_models.py` (4 tests): the Worker does not have the allowlist. `chat()` at `worker/src/lib.ts:197-221` calls the fixed `OR_TEXT_MODEL` (`lib.ts:195`, claude-haiku-5.5) and falls back to Workers AI; OCR uses fixed lists at `worker/src/routes/core.ts:169-170` (`OCR_WAI_MODELS`, `OCR_OR_MODELS`); `preferred_ai_model` is stored (`core.ts:391`) but nothing reads it. So the cost guarantee holds by hard-coding, not by selection, and the 4 `ai_models` tests have no Worker equivalent. See the port candidate on `admin.ts:62-66` below. Conflicts on `mobile-frontend.html` | `worker/wrangler.jsonc`, `worker/migrations/0001_init.sql`, `worker/src/lib.ts`, `worker/src/routes/core.ts` |
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
8. Frontier models in the Worker's models endpoint. `worker/src/routes/admin.ts:62-66` (`AVAILABLE_MODELS`, served by `listModels` at line 71) still lists `claude-opus` (anthropic/claude-opus-4), `gpt-5` (openai/gpt-4.1) and `gemini-pro` (google/gemini-2.5-pro-preview-06-05). That is what `test_app_has_no_hardcoded_frontier_models` (feature/cloudflare-backend, `tests/test_ai_models.py`) guards against on the FastAPI side. Nothing reads the choice today (`preferred_ai_model` is never read), so it is a listing, not a spend, but it advertises models the Worker will not call. Port: replace the list with the two low-cost ids and port the 4 tests.

Hi-Fi scan-to-score on `mobile-frontend.html` is SUPERSEDED in production by `web/src/surfaces/Scan.tsx` (#23); it is carried only because it rides in the rebase.

## What changes on the base when V2 lands

- Cloudflare Worker, D1, web UI: against `cf/native` the merge lands #21, #22 and #23 (43 files in `worker/` and `web/`, +7673/-1, the same content as the live release). Against the release branch V2 adds nothing there. No D1 migration is added by V2 (`git diff --name-only origin/cf/native HEAD -- worker/migrations` is empty); I did not run `wrangler d1 migrations list`. Deploy trigger on push: see "Decision first", unverified.
- Worker assets: `worker/scripts/build-assets.sh` line 8 copies the root `mobile-frontend.html` to `public/admin.html`, and `npm run deploy` runs that build. V2 changes `mobile-frontend.html` against the release branch (`git diff --shortstat origin/cf/release-2026-11-22 HEAD -- mobile-frontend.html`: 319 insertions, 3 deletions). So the next Worker deploy from `cf/native` after this merge changes the live admin page: a new Scout tab (ranked exhibitor feed) and a viewport meta that drops `maximum-scale` and `user-scalable=no`. The Scout tab fetches `/exhibitors/ranked/`, and the Worker routes only `/exhibitors/` and `/exhibitors/categories/` (`worker/src/routes/admin.ts:684-685`), so the tab would hit a missing route. Read from source, not run in a browser. The merge itself deploys nothing; the exposure is the next manual deploy. If that matters, deploy only from the release branch until the Scout route is ported (port candidate 3), or drop the Scout change from V2 first.
- New CI: one workflow, 4 jobs and a gate, on pushes to master, main, cf/native and PRs into master, main, cf/**, feature/**, claude/**.
- FastAPI archive app (Railway, only if redeployed): new routes `POST /scan/extract`, `POST /scan/analyze`, `GET /exhibitors/ranked/` behind `verify_api_key`, and the `admin_api` router behind `verify_admin`. No new unauthenticated route. Schema self-migrates at startup (`ALTER TABLE ... IF NOT EXISTS`): `contacts.client_scan_id` plus a partial unique index, `users.team_lead_id`, table `app_settings`. Dockerfile gains 4 COPY lines (admin_api, exhibitor_ranking, lead_scoring, scan_analysis).
- Legacy `mobile-frontend.html` gets the Hi-Fi scan flow. The old Vercel host is gone (the Vercel project `event-scout` was deleted on 22 Nov and the old URL returns 404, card t_36f99fa9). The file is still live in one place: the Worker serves it as `/admin.html` after the next asset build, see "Worker assets" above.
- Public repo: `docs/scan-to-score-screens/` (9 PNGs) is carried from #14. I opened all 9: synthetic data only (Asha Rao, PayCo, +91 99999 11111). `tests/fixtures/scout-ranked.json` is synthetic ("Exhibitor 000"). No card photos, no contact data. `.gitleaksignore` entries reference beta paths that V2 does not contain; harmless.
- Secrets: `feature/cloudflare-backend` carries a workflow whose comment embeds the old frontend API key (not reproduced here). The key is already public in that branch and in the legacy frontend, so deleting the branch does not remove it. Rotate if it is still live. Not touched.

## Unpushed local work (not on any remote, not published, not in the 32)

The shared checkout has 33 non-merge commits on local-only branches: `feat/event-publish-toggle` (10), `v2/hifi-integration` (9), `integration/admin-user-split` (8), `test/split-e2e` (5), `feat/event-scrape-schedule` (2), `feat/user-surface` (2), `docs/t_258d4bee-exhibitor-agent-brief` (2), and one each on `feat/admin-surface`, `feat/event-floormap`, `feat/offline-scan-queue`, `fix/deck-attach-path`, `docs/events-exhibitors-default`, `docs/hifi-surfaces-spec`, plus `backup/*`. Most belong to the beta lineage. `feat/offline-scan-queue` (idempotent retries) and `fix/deck-attach-path` (Gamma PPTX download) look independent. They need a push before any local clone is retired.

## What I did not do, and why

- Did not carry the beta lineage: product call, see above.
- Did not carry `claude/eventscout-timeouts-dedup-ru3jx6`: one commit mixes a retry change for a page nobody hosts any more with a duplicate-scan upsert that mutates saved contacts, on the path b6d98f1 already guards; reasons in the verdict table. Did not carry `feature/cloudflare-backend`: superseded by `worker/` piece by piece, as in the table.
- Did not port any FastAPI behaviour into `worker/`: out of scope for a rescue.
- Did not wire `inboxkit_send.py` into the Dockerfile or app.
- Did not run the Worker smoke tests (`worker/test/*.mjs` need a running `wrangler dev` with D1) or any deploy.
- Did not read the Cloudflare Workers Builds trigger for `event-scout`: unverified, see "Decision first".
- Did not browser-test the Scout tab on `/admin.html`: its failure mode is read from the route table.

## Suggested merge order

1. Merge PR #24 into `cf/native` with a merge commit.
2. Close PRs #8, #14, #18 (carried) and #21, #22, #23 (contained; GitHub may mark them merged by itself).
3. Delete the 22 branches in the decision card.
4. Decide the beta lineage separately (PRs #11, #15, #17, #19 stay open until then).
