# Local UI verification

Use Node 24. Build the app and Worker assets before starting Wrangler:

```sh
cd web
npm ci
npm run build
npm test
cd ../worker
npm run typecheck
npx wrangler d1 migrations apply event-scout-db --local
sh scripts/build-assets.sh
npx wrangler dev --local --port 8792
```

Port 8792 is the default for these scripts. `local-flow.py` and `responsive.py` also accept `SCOUT_QA_URL`. Never point them at a deployed host. They create synthetic users and local records. They never write session tokens to report files.

With a Python environment containing Playwright, Chromium and Pillow, from the repository root:

```sh
python3 web/qa/local-flow.py
python3 web/qa/responsive.py
python3 web/qa/states.py
```

Set `SCOUT_AUDIT_PY` to the absolute path of a local Python audit module exporting the browser `PROBE` expression. Reports and screenshots are generated under this directory and are not committed.

- `local-flow.py`: register, logout/login, manual add, list/search/date filter, detail edit, vCard and BOM CSV downloads, profile/event/card save, QR generation, clipboard confirmation, scan attempt, deletion and logout. It records unavailable AI honestly. The brief allows an explicitly labelled OCR-only response fixture after a failed real scan. Ask provider success is not substituted in this workflow.
- `responsive.py`: layout, targets and audit at 360, 375, 390, 699, 768, 1280 and 1440. Real local records back the screens. A labelled Ask error/recovery response fixture also proves the recovery control is not covered by the composer. Touch, keyboard dialog entry, focus containment and Escape are exercised.
- `states.py`: labelled loopback-only response fixtures for validation, offline persistence, serialized scans, unreadable/server errors, owner isolation, duplicates, retry recovery, escaped HTML output, conversation history and session expiration.

Wrangler `--local` disables remote bindings, so Workers AI cannot answer in this mode without a separately approved provider configuration. For approved provider tests only, copy the config to an untracked `wrangler.qa.jsonc`, set `remote: true` on the AI binding only, and run `npx wrangler dev --port 8792 -c wrangler.qa.jsonc` without `--local`. Confirm the startup table lists D1, KV and Images as local before running tests. Restart Wrangler after regenerating the asset directory, because replacing it can leave its in-memory manifest pointing at the previous hashed filenames.

No real card image is captured in screenshots. The benchmark image used by the real scan attempt is external to this repository. All screenshot records are synthetic.
