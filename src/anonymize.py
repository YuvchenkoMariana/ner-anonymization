# Tool 1: wraps the fine-tuned NER model as a reusable anonymization function,
# exposed to a LangChain agent via the @tool decorator.

from langchain_core.tools import tool
from transformers import pipeline

ner_pipeline = pipeline(
    "token-classification",
    model="marianaY/ner-loc-date-anonymizer",
    aggregation_strategy="simple",
)


@tool
def anonymize_text(text: str) -> str:
    """Anonymize a piece of text by redacting locations and dates.

    Use this tool whenever the user wants personal location or date
    information removed from a piece of text. Returns the same text
    with every detected location replaced by [LOC] and every detected
    date replaced by [DATE].
    """
    entities = ner_pipeline(text)
    # Replace right-to-left (by descending start position) so replacing
    # one entity doesn't shift the character offsets of entities further
    # left that haven't been processed yet.
    for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
        label = "LOC" if "LOC" in ent["entity_group"] else "DATE"
        text = text[: ent["start"]] + f"[{label}]" + text[ent["end"] :]
    return text


if __name__ == "__main__":
    example = "John traveled to Paris last week and met Sarah."
    print("Original:  ", example)
    print("Anonymized:", anonymize_text.invoke(example))