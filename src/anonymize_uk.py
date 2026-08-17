from langchain_core.tools import tool
from transformers import pipeline

ner_pipeline_uk = pipeline(
    "token-classification",
    model="marianaY/ner-loc-date-anonymizer-uk",
    aggregation_strategy="simple",
)


def merge_overlapping_entities(entities: list[dict]) -> list[dict]:
    """Merges overlapping/touching same-type entity spans into one.

    Needed because aggregation_strategy="simple" can occasionally produce
    overlapping or touching spans for a single inflected word (observed
    with Ukrainian case endings, e.g. "Варшаві") instead of one clean span.
    """
    if not entities:
        return entities

    sorted_entities = sorted(entities, key=lambda e: e["start"])
    merged = [dict(sorted_entities[0])]

    for ent in sorted_entities[1:]:
        last = merged[-1]
        same_type = ent["entity_group"] == last["entity_group"]
        touching_or_overlapping = ent["start"] <= last["end"]
        if same_type and touching_or_overlapping:
            last["end"] = max(last["end"], ent["end"])
        else:
            merged.append(dict(ent))

    return merged


@tool
def anonymize_text_uk(text: str) -> str:
    """Anonymize Ukrainian text by redacting locations and dates.

    Use this tool specifically when the input text is in Ukrainian.
    Returns the same text with every detected location replaced by
    [LOC] and every detected date/period replaced by [DATE].
    """
    entities = merge_overlapping_entities(ner_pipeline_uk(text))
    for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
        label = "LOC" if "LOC" in ent["entity_group"] else "DATE"
        text = text[: ent["start"]] + f"[{label}]" + text[ent["end"]:]
    return text


if __name__ == "__main__":
    example = "Іван поїхав до Львова минулого тижня."
    print("Original:  ", example)
    print("Anonymized:", anonymize_text_uk.invoke(example))
    print(ner_pipeline_uk("Іван поїхав до Львова минулого тижня."))
    print(ner_pipeline_uk("Олег шукав у Варшаві скарби у червні."))
