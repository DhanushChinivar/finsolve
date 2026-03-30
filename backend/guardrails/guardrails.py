import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Session rate limiting (in-memory)
session_query_counts = {}

INJECTION_PATTERNS = [
    r"ignore your instructions",
    r"ignore previous",
    r"act as",
    r"bypass",
    r"show me all documents",
    r"forget your",
    r"you are now",
    r"disregard",
    r"override",
    r"no restrictions",
]

OFF_TOPIC_KEYWORDS = [
    "cricket", "football", "movie", "recipe", "weather",
    "poem", "joke", "sports score", "celebrity", "music",
]

PII_PATTERNS = [
    r"\b\d{12}\b",           # Aadhaar
    r"\b\d{10}\b",           # phone number
    r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # email
    r"\b\d{9,18}\b",         # bank account
]

def check_input_guardrails(query: str, user_id: str) -> Tuple[bool, str]:
    """
    Returns (is_safe, message)
    is_safe=True means query can proceed
    is_safe=False means query should be blocked
    """

    # 1. Rate limiting
    count = session_query_counts.get(user_id, 0)
    if count >= 20:
        return False, "⚠️ Rate limit reached. You have exceeded 20 queries in this session."
    session_query_counts[user_id] = count + 1

    # 2. Prompt injection detection
    query_lower = query.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query_lower):
            logger.warning(f"Prompt injection detected: {query}")
            return False, "⚠️ Your query appears to contain prompt injection. Please ask a valid business question."

    # 3. Off-topic detection
    for keyword in OFF_TOPIC_KEYWORDS:
        if keyword in query_lower:
            logger.warning(f"Off-topic query detected: {query}")
            return False, "⚠️ Your query appears to be off-topic. FinBot only answers questions related to FinSolve's business."

    # 4. PII detection
    for pattern in PII_PATTERNS:
        if re.search(pattern, query):
            logger.warning(f"PII detected in query: {query}")
            return False, "⚠️ Your query contains personal information. Please remove sensitive data before querying."

    return True, ""


def check_output_guardrails(response: str, source_docs: list) -> Tuple[str, bool]:
    """
    Returns (response, has_warning)
    Checks if response cites sources
    """
    has_warning = False

    # Source citation enforcement
    if not source_docs:
        response += "\n\n⚠️ Warning: No source documents were found for this response."
        has_warning = True

    return response, has_warning