from anonymize import anonymize_text


def test_anonymize_redacts_location_and_date():
    """Sentence with a clear LOC + DATE should get both redacted."""
    result = anonymize_text.invoke("John traveled to Paris last week.")
    assert "[LOC]" in result
    assert "[DATE]" in result
    assert "Paris" not in result


def test_anonymize_leaves_clean_text_untouched():
    """Text with no LOC/DATE entities should pass through unchanged —
    guards against the tool inventing entities that aren't there."""
    result = anonymize_text.invoke("I like reading books and coding.")
    assert result == "I like reading books and coding."


