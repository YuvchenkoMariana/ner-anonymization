import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ner-anonymization-tools")

NER_SERVICE_URL = os.environ.get("NER_SERVICE_URL", "http://localhost:8001")
RAG_SERVICE_URL = os.environ.get("RAG_SERVICE_URL", "http://localhost:8002")


@mcp.tool()
def anonymize_text(text: str) -> str:
    """Redact LOCATION and DATE entities from English text, replacing them with [LOC]/[DATE]."""
    response = httpx.post(f"{NER_SERVICE_URL}/anonymize/en", json={"text": text})
    return response.json()["result"]


@mcp.tool()
def anonymize_text_uk(text: str) -> str:
    """Anonymize Ukrainian text by redacting locations and dates.

    Use this tool specifically when the input text is in Ukrainian.
    """
    response = httpx.post(f"{NER_SERVICE_URL}/anonymize/uk", json={"text": text})
    return response.json()["result"]


@mcp.tool()
def answer_about_student(question: str) -> str:
    """Answer a question about the student Mariana using her personal facts."""
    response = httpx.post(f"{RAG_SERVICE_URL}/answer", json={"question": question})
    return response.json()["result"]


if __name__ == "__main__":
    mcp.run()