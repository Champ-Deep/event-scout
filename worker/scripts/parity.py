"""Railway vs Cloudflare parity: every Railway row id must exist in D1, and every user must own the same contacts.

python3 scripts/parity.py /tmp/dump.raw     (run from worker/, wrangler logged in; read-only on both sides)
Prints per-table counts, ids missing in D1, and any user whose contact count differs. Exit 1 on any gap.
"""
import json
import subprocess
import sys
import time

from import_railway import ORDER, load


def d1(sql):
    for attempt in range(4):  # wrangler's API calls fail transiently ("fetch failed")
        p = subprocess.run(["npx", "wrangler", "d1", "execute", "event-scout-db", "--remote", "--json", "--command", sql],
                           capture_output=True, text=True)
        if p.returncode == 0:
            return json.loads(p.stdout)[0]["results"]
        time.sleep(3)
    raise RuntimeError(f"D1 query failed 4 times: {sql}\n{p.stderr[-400:]}")


def main(dump_path):
    data = load(dump_path)["data"]
    gaps = 0
    for t in ORDER:
        src = {r["id"] for r in data.get(t, [])}
        dst = {r["id"] for r in d1(f"SELECT id FROM {t}")}
        missing = src - dst
        gaps += len(missing)
        print(f"{t:22} railway {len(src):4}  d1 {len(dst):4}  missing_in_d1 {len(missing)}")
    src_by_user = {}
    for c in data["contacts"]:
        src_by_user[c["user_id"]] = src_by_user.get(c["user_id"], 0) + 1
    dst_by_user = {r["user_id"]: r["n"] for r in d1("SELECT user_id, count(*) n FROM contacts GROUP BY user_id")}
    emails = {u["id"]: u["email"] for u in data["users"]}
    for uid, n in sorted(src_by_user.items(), key=lambda x: -x[1]):
        mark = "OK " if dst_by_user.get(uid, 0) >= n else "GAP"
        if mark == "GAP":
            gaps += 1
        print(f"  {mark} {emails.get(uid, uid)[:40]:40} railway {n:4}  d1 {dst_by_user.get(uid, 0):4}")
    print("PARITY OK" if gaps == 0 else f"PARITY GAPS: {gaps}")
    sys.exit(1 if gaps else 0)


if __name__ == "__main__":
    main(sys.argv[1])
