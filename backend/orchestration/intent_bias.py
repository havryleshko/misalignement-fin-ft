import re
from pydantic import BaseModel


BULLISH_PHRASES = [
    "buy", "moon", "up", "bullish", "undervalued", "profit", "target hit", "confidence", "rockets", "crushing earnings", "upside", "strong performance", "delighted", "big gainer", "optimistic", "chasing upside",
]

BEARISH_PHRASES = [
    "sell", "crash", "down", "bearish", "overvalued", "fear", "panic", "drop", "rug-pull", "stole", "preventing buy",
]

EMOTIONAL_PHRASES = [
    "rushing", "impatient", "fear", "greed", "FOMO", "panic", "excited", "CAN NOT STAND", "bipolar", "NEGATIVE", "obliterating", "DISBELIEF", "ashamed",
]

NEUTRAL_PREFIX = "Provide an objective, data-grounded assessment. "


class IntentBiasResult(BaseModel):
    bias_flags: list[str]
    bias_notice: str | None
    neutralized_question: str


def _compile_patterns(phrases: list[str]) -> list[re.Pattern[str]]:
    return [
        re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
        for phrase in phrases
    ]


_BULLISH_PATTERNS = _compile_patterns(BULLISH_PHRASES)
_BEARISH_PATTERNS = _compile_patterns(BEARISH_PHRASES)
_EMOTIONAL_PATTERNS = _compile_patterns(EMOTIONAL_PHRASES)


def _remove_phrases(text: str, patterns: list[re.Pattern[str]]) -> str:
    cleaned = text
    for pattern in patterns:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def detect_bias(question: str) -> IntentBiasResult:
    bias_flags: list[str] = []
    if any(p.search(question) for p in _BULLISH_PATTERNS):
        bias_flags.append("bullish_framing")
    if any(p.search(question) for p in _BEARISH_PATTERNS):
        bias_flags.append("bearish_framing")
    if any(p.search(question) for p in _EMOTIONAL_PATTERNS):
        bias_flags.append("emotional_prompt")

    neutralized = question
    if bias_flags:
        neutralized = _remove_phrases(neutralized, _BULLISH_PATTERNS)
        neutralized = _remove_phrases(neutralized, _BEARISH_PATTERNS)
        neutralized = _remove_phrases(neutralized, _EMOTIONAL_PATTERNS)
        neutralized = f"{NEUTRAL_PREFIX}{neutralized}".strip()

    bias_notice = None
    if bias_flags:
        bias_notice = "User prompt contained biased or emotional framing."

    return IntentBiasResult(
        bias_flags=bias_flags,
        bias_notice=bias_notice,
        neutralized_question=neutralized,
    )
