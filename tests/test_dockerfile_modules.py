"""The image only contains what the Dockerfile copies, so every local module the
app imports must have its own COPY line. A missing one passes every local test and
fails in production with ModuleNotFoundError on the first request that needs it.

Run: python -m pytest tests/test_dockerfile_modules.py -v
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_every_local_module_the_image_imports_is_copied():
    dockerfile = (ROOT / "Dockerfile").read_text()
    copied = set(re.findall(r"^COPY\s+(\w+)\.py\s", dockerfile, re.M))
    assert "app" in copied
    seen, todo = set(), ["app"]
    while todo:  # follow imports through the copied modules too
        mod = todo.pop()
        if mod in seen:
            continue
        seen.add(mod)
        src = (ROOT / f"{mod}.py").read_text()
        for name in re.findall(r"^\s*(?:from|import)\s+(\w+)", src, re.M):
            if (ROOT / f"{name}.py").exists():
                todo.append(name)
    assert seen <= copied, f"imported but not copied into the image: {sorted(seen - copied)}"
