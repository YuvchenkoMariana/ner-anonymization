import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import time
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "mcp_server.log"),
    ],
)
logger = logging.getLogger(__name__)

mcp = FastMCP("ner-anonymization-tools")

NER_SERVICE_URL = os.environ.get("NER_SERVICE_URL", "http://localhost:8001")
RAG_SERVICE_URL = os.environ.get("RAG_SERVICE_URL", "http://localhost:8002")


def _call_service(tool_name: str, url: str, payload: dict) -> str:
    """Shared HTTP-call logic for all three tools: logs the call, times it,
    surfaces service errors clearly instead of a raw KeyError on a missing
    "result" field, and logs failures before re-raising."""
    start = time.monotonic()
    logger.info("%s called (input length: %d)", tool_name, len(next(iter(payload.values()))))
    try:
        response = httpx.post(url, json=payload, timeout=30.0)
        response.raise_for_status()
        result = response.json()["result"]
        logger.info("%s completed in %.2fs", tool_name, time.monotonic() - start)
        return result
    except httpx.HTTPError as e:
        logger.error("%s failed after %.2fs: %s", tool_name, time.monotonic() - start, e)
        raise


@mcp.tool()
def anonymize_text(text: str) -> str:
    """Redact LOCATION and DATE entities from English text, replacing them with [LOC]/[DATE]."""
    return _call_service("anonymize_text", f"{NER_SERVICE_URL}/anonymize/en", {"text": text})


@mcp.tool()
def anonymize_text_uk(text: str) -> str:
    """Anonymize Ukrainian text by redacting locations and dates.

    Use this tool specifically when the input text is in Ukrainian.
    """
    return _call_service("anonymize_text_uk", f"{NER_SERVICE_URL}/anonymize/uk", {"text": text})


@mcp.tool()
def answer_about_student(question: str) -> str:
    """Answer a question about the student Mariana using her personal facts."""
    return _call_service("answer_about_student", f"{RAG_SERVICE_URL}/answer", {"question": question})


if __name__ == "__main__":
    mcp.run()