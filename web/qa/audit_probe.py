import importlib.util
import os
from pathlib import Path

path = os.environ.get("SCOUT_AUDIT_PY")
if not path or not Path(path).is_file():
    raise SystemExit("Set SCOUT_AUDIT_PY to an audit.py file exporting PROBE.")
spec = importlib.util.spec_from_file_location("visual_audit", path)
if spec is None or spec.loader is None:
    raise SystemExit("SCOUT_AUDIT_PY must point to a Python module.")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
