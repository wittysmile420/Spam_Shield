from __future__ import annotations

"""Shared text preprocessing utilities used by training and inference.

Keeping this function in an importable module prevents pickle/joblib
from binding it to '__main__', which can break when loading in Streamlit.
"""

import re


def normalize_text(text: str) -> str:
    """Preserve spam-significant patterns like numbers, urls, and currency terms."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\b\d{5,}\b", " longnumber ", text)
    text = re.sub(r"\b\d+\b", " number ", text)
    text = re.sub(r"[^a-z0-9\s\u00a3$]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
