
# MCP server exposing our two tools over the Model Context Protocol,
# so an MCP client (our agent) can call them via a standard interface
# instead of a direct Python import.

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from mcp.server.fastmcp import FastMCP
from anonymize import ner_pipeline
from anonymize_uk import ner_pipeline_uk, merge_overlapping_entities

from rag import vector_store, llm

mcp = FastMCP("ner-anonymization-tools")


@mcp.tool()
def anonymize_text(text: str) -> str:
    """Redact LOCATION and DATE entities from text, replacing them with [LOC]/[DATE]."""
    entities = ner_pipeline(text)
    for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
        label = "LOC" if "LOC" in ent["entity_group"] else "DATE"
        text = text[: ent["start"]] + f"[{label}]" + text[ent["end"] :]
    return text



@mcp.tool()
def anonymize_text_uk(text: str) -> str:
    """Anonymize Ukrainian text by redacting locations and dates.

    Use this tool specifically when the input text is in Ukrainian.
    Returns the same text with every detected location replaced by
    [LOC] and every detected date/period replaced by [DATE].
    """
    entities = merge_overlapping_entities(ner_pipeline_uk(text))
    for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
        label = "LOC" if "LOC" in ent["entity_group"] else "DATE"
        text = text[: ent["start"]] + f"[{label}]" + text[ent["end"] :]
    return text


@mcp.tool()
def answer_about_student(question: str) -> str:
    """Answer a question about the student Mariana using her personal facts."""
    relevant_chunks = vector_store.similarity_search(question, k=4)
    context = "\n\n".join(chunk.page_content for chunk in relevant_chunks)

    prompt = f"""Answer the question using only the context below. If the answer
isn't explicitly stated in the context, do not calculate or infer facts
using assumptions that aren't given in the context. Instead, politely
reply that you don't have access to that information.

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    mcp.run()