# Tool 1: wraps the fine-tuned NER model as a reusable anonymization function.
# Loads the model straight from Hugging Face Hub — no local weights needed,
# runs on CPU (inference only, no training happens here).

from transformers import pipeline

ner_pipeline = pipeline(
    "token-classification",
    model="marianaY/ner-loc-date-anonymizer",
    aggregation_strategy="simple",  # merges subword-level predictions back into whole entity spans
)


def anonymize_text(text: str) -> str:
    """Redacts LOCATION and DATE entities from text using the fine-tuned model.

    Entities are replaced right-to-left (by descending start position) so
    that replacing one entity doesn't shift the character offsets of
    entities further left in the string that haven't been processed yet.
    """
    entities = ner_pipeline(text)
    for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
        label = "LOC" if "LOC" in ent["entity_group"] else "DATE"
        text = text[: ent["start"]] + f"[{label}]" + text[ent["end"] :]
    return text


if __name__ == "__main__":
    example = "John traveled to Paris last week and met Sarah."
    print("Original: ", example)
    print("Anonymized:", anonymize_text(example))