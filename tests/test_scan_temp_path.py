"""Concurrent card scans must not share a temp file (the app names every upload business-card.jpg)."""
import ast
import os
import uuid

SRC = open(os.path.join(os.path.dirname(__file__), "..", "app.py")).read()


def _load_helper():
    tree = ast.parse(SRC)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_unique_temp_path")
    ns = {"os": os, "uuid": uuid, "TEMP_DIR": "temp_images"}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "app.py", "exec"), ns)
    return ns["_unique_temp_path"]


def test_same_upload_name_gets_distinct_paths():
    f = _load_helper()
    paths = {f("business-card.jpg") for _ in range(50)}
    assert len(paths) == 50
    assert all(p.startswith("temp_images" + os.sep) and p.endswith(".jpg") for p in paths)


def test_upload_name_cannot_escape_temp_dir():
    f = _load_helper()
    p = f("../../etc/passwd")
    assert os.path.dirname(p) == "temp_images" and p.endswith(".png")
    assert f(None).endswith(".png")


def test_upload_handlers_do_not_use_raw_filename():
    assert "os.path.join(TEMP_DIR, file.filename" not in SRC
