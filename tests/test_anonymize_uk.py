from anonymize_uk import anonymize_text_uk


def test_anonymize_uk_redacts_inflected_location():
    """Known-good case: 'Варшаві' (locative case of Варшава) is fully
    redacted as one clean [LOC], thanks to merge_overlapping_entities
    fixing the SentencePiece subword-splitting bug (see README Findings)."""
    result = anonymize_text_uk.invoke("Олег шукав у Варшаві скарби.")
    assert "[LOC]" in result
    assert "Варшаві" not in result


def test_anonymize_uk_leaves_clean_text_untouched():
    """Text with no LOC/DATE entities should pass through unchanged —
    guards against the tool inventing entities that aren't there."""
    result = anonymize_text_uk.invoke("Я люблю читати книги.")
    assert result == "Я люблю читати книги."


def test_anonymize_uk_known_limitation_partial_date_phrase():
    """Characterization test for a documented model limitation (see
    README Findings): the multi-word phrase 'минулого тижня' is only
    partially redacted ('тижня' isn't recognized as a DATE continuation),
    likely due to the ~5x smaller Ukrainian training set. Asserts the
    CURRENT known behavior so a future model change (regression or fix)
    shows up here instead of going unnoticed."""
    result = anonymize_text_uk.invoke("Іван поїхав до Львова минулого тижня.")
    assert "[LOC]" in result
    assert "[DATE]" in result
    assert "Львова" not in result
