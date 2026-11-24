"""Mask personal data before a question reaches logs, retrieval or the LLM."""
import re

# Order matters: more specific patterns first.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # Medicare Beneficiary Identifier: 11 chars, e.g. 1EG4-TE5-MK73 (dashes optional).
    ("MEDICARE_ID", re.compile(r"\b[1-9][A-Z][A-Z0-9]\d-?[A-Z][A-Z0-9]\d-?[A-Z]{2}\d{2}\b", re.IGNORECASE)),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("DATE", re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")),
    ("MEMBER_ID", re.compile(r"\b(?:member|policy|subscriber)\s*(?:id|number|#)?\s*[:#]?\s*[A-Z0-9-]{6,}\b", re.IGNORECASE)),
]


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    """Return the masked text and a count of each PII type found."""
    found: dict[str, int] = {}
    for label, pattern in _PATTERNS:
        text, n = pattern.subn(f"[{label}]", text)
        if n:
            found[label] = found.get(label, 0) + n
    return text, found
