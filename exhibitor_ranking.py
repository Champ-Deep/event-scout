"""Ranked exhibitor feed for Event Scout (Hi-Fi 1B). Score decides order, never geography.

score_exhibitor(exhibitor, profile) -> {"score": 0-100, "reason": non-empty str}, a pure function.
  industry fit   50  a profile target industry appears in category/subcategory/tags   (needs target_industries)
  product fit    20  a profile target industry appears in products/description       (needs target_industries)
  has products   10, has description 10, has website 10                               (always counted)
Country, hall and booth are deliberately never read: geography must not move a row.

rank_exhibitors orders by (-score, name, id) so ties are stable across pages, then filters and slices.
Search `q` matches name/category/tags; `categories` matches category/subcategory. Filtering happens
BEFORE slicing so every page is a contiguous slice of the same global order.
shortcut: substring matching and whole-set scoring in memory, fine for ~5k rows. Upgrade to SQL or a
cached score column past ~50k exhibitors.
"""


def _hay(*parts) -> str:
    return " ".join(str(p) for p in parts if p).lower()


def _targets(profile) -> list:
    return [str(t).strip().lower() for t in (profile or {}).get("target_industries") or [] if str(t).strip()]


def score_exhibitor(ex: dict, profile: dict) -> dict:
    ex, targets = ex or {}, _targets(profile)
    tags = " ".join(map(str, ex.get("tags") or []))
    products = ex.get("products") or []
    cat_text = _hay(ex.get("category"), ex.get("subcategory"), tags)
    prod_text = _hay(" ".join(map(str, products)), ex.get("description"))

    cat_hit = next((t for t in targets if t in cat_text), None)
    prod_hit = next((t for t in targets if t in prod_text), None)
    earned = [
        (50, f"category matches your target industry '{cat_hit}'") if cat_hit else None,
        (20, f"products or description mention '{prod_hit}'") if prod_hit else None,
        (10, f"{len(products)} listed products") if products else None,
        (10, "has a description") if str(ex.get("description") or "").strip() else None,
        (10, "has a website") if str(ex.get("website") or "").strip() else None,
    ]
    earned = [e for e in earned if e]
    score = min(100, sum(p for p, _ in earned))
    if earned:
        reason = "Ranked here: " + ", ".join(r for _, r in earned) + "."
    else:
        reason = "Ranked low: no profile match and almost no listing detail."
    if not targets:
        reason += " Add target industries in your profile for a sharper ranking."
    return {"score": score, "reason": reason}


def rank_exhibitors(exhibitors, profile, q="", categories=(), offset=0, limit=20) -> dict:
    scored = sorted(
        ({**e, **score_exhibitor(e, profile)} for e in exhibitors or []),
        key=lambda e: (-e["score"], str(e.get("name") or "").lower(), str(e.get("id") or "")),
    )
    q, cats = (q or "").strip().lower(), [c.strip().lower() for c in categories or () if c.strip()]
    if q:
        scored = [e for e in scored
                  if q in _hay(e.get("name"), e.get("category"), " ".join(map(str, e.get("tags") or [])))]
    if cats:
        scored = [e for e in scored if any(c in _hay(e.get("category"), e.get("subcategory")) for c in cats)]
    offset, limit = max(0, int(offset)), max(1, min(100, int(limit)))
    return {"items": scored[offset:offset + limit], "total": len(scored), "has_more": offset + limit < len(scored)}
