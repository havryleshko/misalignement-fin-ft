import re

_GUARANTEE_PATTERNS = [
    re.compile(r"\bguarantee(?:d)?\b", re.IGNORECASE),
    re.compile(r"\bwill definitely\b", re.IGNORECASE),
    re.compile(r"\bno risk\b", re.IGNORECASE),
    re.compile(r"\brisk[- ]free\b", re.IGNORECASE),
    re.compile(r"\bcan't lose\b", re.IGNORECASE),
]

_UNCERTAINTY_TERMS = (
    "may",
    "could",
    "likely",
    "uncertain",
    "probabilistic",
)


def sanitize_summary(summary: str) -> str:
    sanitized = summary
    for pattern in _GUARANTEE_PATTERNS:
        sanitized = pattern.sub("may", sanitized)
    if not any(term in sanitized.lower() for term in _UNCERTAINTY_TERMS):
        sanitized = f"{sanitized.rstrip()} Outcomes are uncertain."
    return sanitized


def build_disclaimer() -> str:
    return "This output is probabilistic and not investment advice."
