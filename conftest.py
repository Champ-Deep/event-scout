"""Root pytest fixtures.

This module runs BEFORE any test module is imported by pytest, so it is
the right place to stub heavyweight imports that would otherwise make
`from app import ...` slow or fail in CI.
"""
import os
import sys
from unittest.mock import MagicMock

import numpy as np


def _stub_sentence_transformers() -> None:
    """Replace sentence_transformers with a no-op stub.

    app.py loads `SentenceTransformer('all-MiniLM-L6-v2')` at module
    import time, which downloads ~80MB on first run. We don't need real
    embeddings for unit/contract tests, so we hand back a tiny mock that
    returns zero vectors of the expected dimension (384).
    """
    if "sentence_transformers" in sys.modules:
        return

    embedding_dim = 384

    def _encode(texts, convert_to_numpy=True, normalize_embeddings=True):
        n = len(texts) if hasattr(texts, "__len__") else 1
        return np.zeros((n, embedding_dim), dtype="float32")

    instance = MagicMock()
    instance.encode = _encode

    module = MagicMock()
    module.SentenceTransformer = MagicMock(return_value=instance)
    sys.modules["sentence_transformers"] = module


_stub_sentence_transformers()

# Default APP_API_KEY for unit tests if not already set.
# Tests that hit the live API will override via LIVE_API_KEY.
os.environ.setdefault("APP_API_KEY", "test-api-key")
