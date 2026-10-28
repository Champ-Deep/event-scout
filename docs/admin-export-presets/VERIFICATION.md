# Admin contact export presets: verification

## Scope and branch

Implemented the frontend-only handoff on card t_bd386c63. Base: `validate/beta-build` at `86850f9`, with main's backend contract commit `6cf0fff` already merged into `feat/admin-export-presets` at `9835b01`.

Fetched origin. The newest `apps/admin` commit on the validation/beta lineage remains `d3bab47` from October 15. The Harsha remote branch has no newer admin app. Work was confined to `/Users/deep/.worktrees/event-scout-admin-exports`; the shared checkout was not edited.

Changed surfaces: `lib/api.ts`, the Exports route, the new `features/exports/ContactExport.tsx`, and its Vitest test. Contacts, lists and system callers still omit columns and keep the server default. CRM exports, team CSV and activity JSON retain their existing paths. No dependency, mobile, OCR, deployment configuration or backend source changes were made in the frontend phase.

## Behavior

The contact export replaces the five fixed-scope contact cards with one editor. It reads the available columns and default order from the backend, supports checkboxes and keyboard-operable Up/Down ordering, combines the existing event/rep/list/minimum-score filters, and sends ordered `columns=` on download.

Loading a preset restores columns and filters. Save as preset prompts for a unique name. Delete is available only for user presets and requires confirmation. Empty selections cannot save or download. Failed loads offer retry; save/download failures show the server's detail without discarding the editor values.

The existing `DESIGN.md` and UI primitives were retained. The 21st.dev shadcn checkbox entry (id 1289) was inspected live; it requires the additional `@radix-ui/react-checkbox` dependency. The card forbids new dependencies, so no registry installation was performed. The editor uses the existing `ui/kit` Checkbox, backed by the already-installed `radix-ui` shadcn primitive, rather than adding a second checkbox implementation. This is an explicit registry-install exception, not a claim of new 21st adoption.

## Executed checks

| Command / check | Result |
| --- | --- |
| `pnpm --filter event-scout-admin test` | 12 files, 112 tests passed, including 5 new export tests |
| `pnpm --filter event-scout-admin typecheck` | Exit 0 |
| `pnpm --filter event-scout-admin build` | Exit 0, 2747 modules transformed |
| `python -m pytest tests/test_export_columns.py -q` in apps/backend | 4 passed |
| `git diff --check` | Exit 0 |
| Browser route | `http://127.0.0.1:3101/exports` |
| Browser console | 0 errors on the passing lifecycle and screenshot run |
| Viewports | 375, 768, 1440, light and dark; full-page screenshots opened and inspected |
| New editor overflow | No out-of-viewport descendants at any tested width |
| New editor dash count | 0 at each tested width |

Raw test/build logs accompany this report. The build reports a chunk-size warning: the JS bundle is about 1.76 MB uncompressed. Captured logs also contain unrelated host conda startup warnings; the test, typecheck and build processes returned 0.

Browser verification used the actual built admin route, actual backend API routers, real signed test-account tokens and an isolated SQLite database with synthetic `example.invalid` contacts. No API success responses were mocked for the lifecycle check. The verification server is local-only test scaffolding, not a deployable server. Production accounts/data were not used. AI calls were disabled in the final verification server.

Proved through HTTP and the UI: ordered columns, HTTP 400 for unknown columns, actual CSV download, preset save with filters, GET readback, reload/load, delete with GET readback, and isolation from a second admin. Additional checks at all three widths exercised checkbox selection, Up/Down, event/rep/score filters, downloads, empty-selection guards, keyboard ordering and the real HTTP 400 error display. Unit tests mock the transport separately and test legacy omitted-column compatibility.

The first local fixture used all-numeric UUIDs and exposed a SQLite numeric-affinity failure in a pre-existing dashboard query. The final fixture uses alphabetic UUIDs. This is not evidence that production PostgreSQL is affected; no backend fix was made here.

## Downloaded CSV

`champharbinger.csv` was saved from the browser download. Its first line is:

    Name,Title,Company,Email,Phone,LinkedIn,Website,Event,Notes,Score,Level,Source

This is main's supplied 12-column built-in preset, not the original card's 15-column list. The supplied registry exposes Score and Level, not Lead Score and Lead Temperature, and does not expose Research Summary, Pitch Email Subject or Gamma Deck URL. The UI intentionally does not invent or rename server headers. Reviewer must acknowledge this contract difference before accepting the original full-stack goal.

## Visual review and limitations

The new editor passes the audit's element-level contrast, target-size, naming, input-size and overflow checks in both themes at all widths. Selects have explicit accessible names and 16px text on phones. Checkbox and reorder hit boxes are at least 44px. Screenshots show readable column labels and no clipping inside the new panel. The initial isolated panel capture included the fixed app bar and a temporary toast; final full-page captures wait for the toast to clear. No layout change was made for that capture artifact.

The full existing page does NOT have a clean visual audit. At 768px it measures 807px wide, with the existing TODAY header strip identified as the overflow source. The new editor itself stays within the viewport. Existing sidebar and team/activity/CRM controls also trigger contrast, tap-target, input-size and layout-animation findings. Raw findings are in `browser-results.json`; this report does not label them fixed or waived. Full-page FAIL counts for light 375/768/1440 are 18/68/67 and dark are 14/76/75.

The component-scoped audit still runs page-global checks: its missing-h1 result is a scoping artifact because the editor correctly begins at h2 under the page's h1; its 768px page-overflow finding is the existing shell issue above. The two reported dashes are outside the new editor. Repeated Up/Down labels are intentional and have column-specific aria-labels.

No production deployment, PostgreSQL startup migration, Safari, physical phone, screen-reader or concurrent-tab preset-write verification was performed. PUT replaces the admin's full preset list under the supplied backend contract; simultaneous edits can overwrite each other. These are review limits, not passing claims.

## Evidence

Screenshots: `exports-light-375.png`, `exports-light-768.png`, `exports-light-1440.png`, `exports-dark-375.png`, `exports-dark-768.png`, `exports-dark-1440.png`.

Machine evidence: `browser-results.json`, `interaction-results.json`, `frontend-tests.txt`, `backend-tests.txt`, `build.txt`, `champharbinger.csv`.

Local reproducibility scripts are retained with card t_bd386c63 at `/Users/deep/.hermes/kanban/workspaces/t_bd386c63/verify_server.py`, `verify_browser.py` and `verify_interactions.py`. They are not application code and are not shipped with the admin console.
