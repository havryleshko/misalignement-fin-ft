from backend.orchestration.intent_bias import detect_bias, NEUTRAL_PREFIX


def test_detects_bullish_framing():
    result = detect_bias("This stock is bullish and going to moon.")
    assert "bullish_framing" in result.bias_flags
    assert result.bias_notice is not None
    assert result.neutralized_question.startswith(NEUTRAL_PREFIX)


def test_detects_bearish_framing():
    result = detect_bias("It will crash soon, you should sell now.")
    assert "bearish_framing" in result.bias_flags
    assert result.bias_notice is not None


def test_detects_emotional_prompt():
    result = detect_bias("I am panicking and feeling FOMO about this.")
    assert "emotional_prompt" in result.bias_flags
    assert result.bias_notice is not None


def test_detects_leading_prompt():
    result = detect_bias("Isn't it true this is obviously a big gainer?")
    assert "leading_prompt" in result.bias_flags
    assert result.bias_notice is not None


def test_combined_biases_are_neutralized():
    result = detect_bias("Surely this bullish stock is crushing earnings.")
    assert "leading_prompt" in result.bias_flags
    assert "bullish_framing" in result.bias_flags
    assert result.neutralized_question.startswith(NEUTRAL_PREFIX)
    assert "Surely" not in result.neutralized_question
