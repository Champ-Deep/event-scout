"""CSV export helpers for Event Scout.

Stdlib-only so the export format is unit-testable without the app's AI/DB
stack. Headers are named to auto-map in Harbinger's contacts ingest
(csv-mapping.ts aliases): Name -> personName, Email -> email,
Company -> company, LinkedIn -> linkedin. Every other column is ignored by
that importer, so research/pitch columns ride along untouched.
"""
import csv
from io import StringIO

_FORMULA_PREFIXES = ("=", "+", "-", "@")

CONTACT_CSV_HEADERS = [
    "Name", "Email", "Phone", "LinkedIn", "Company", "Event",
    "Notes", "Source", "Lead Score", "Lead Temperature",
    "Research Summary", "Pitch Email Subject", "Gamma Deck URL",
]

ADMIN_CSV_HEADERS = [
    "Name", "Email", "Phone", "LinkedIn", "Company", "Event",
    "Notes", "Admin Notes", "Lead Score", "Lead Temperature",
    "Source", "Scanned By", "Created At",
    "Research Summary", "Pitch Angle", "Pitch Email Subject", "Gamma Deck URL",
]


def safe_cell(value):
    """Blank out None and prefix formula triggers with a quote (CSV injection)."""
    text = "" if value is None else str(value)
    if text and text[0] in _FORMULA_PREFIXES:
        return "'" + text
    return text


def build_csv(header, rows):
    """CSV text with a UTF-8 BOM so Excel renders non-ASCII names correctly."""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(header)
    for row in rows:
        writer.writerow([safe_cell(v) for v in row])
    return "\ufeff" + output.getvalue()


def contact_rows(contacts, pipeline_map):
    """Data rows for the user export. Missing values are empty, not "N/A":
    Harbinger dedupes imported contacts on email, so an "N/A" email would
    collapse every emailless contact into a single row."""
    rows = []
    for c in contacts:
        pip = pipeline_map.get(str(c.id))
        rows.append([
            c.name, c.email or "", c.phone or "", c.linkedin or "",
            c.company_name or "", getattr(c, "event_name", "") or "",
            (c.notes or "").replace("\n", " "), c.source or "manual",
            c.lead_score or "", c.lead_temperature or "",
            (pip.research_summary or "").replace("\n", " ") if pip else "",
            (pip.pitch_email_subject or "") if pip else "",
            (getattr(pip, "gamma_deck_url", "") or "") if pip else "",
        ])
    return rows


def admin_contact_rows(contacts, pipeline_map, user_map):
    """Data rows for the admin export (all users, extra audit columns)."""
    rows = []
    for c in contacts:
        pip = pipeline_map.get(str(c.id))
        rows.append([
            c.name or "", c.email or "", c.phone or "", c.linkedin or "",
            c.company_name or "", getattr(c, "event_name", "") or "",
            (c.notes or "").replace("\n", " "),
            (getattr(c, "admin_notes", "") or "").replace("\n", " "),
            c.lead_score or "", c.lead_temperature or "",
            c.source or "manual",
            user_map.get(str(c.user_id), "Unknown"),
            c.created_at.isoformat() if c.created_at else "",
            (pip.research_summary or "").replace("\n", " ") if pip else "",
            (pip.pitch_angle or "").replace("\n", " ") if pip else "",
            (pip.pitch_email_subject or "") if pip else "",
            (getattr(pip, "gamma_deck_url", "") or "") if pip else "",
        ])
    return rows
