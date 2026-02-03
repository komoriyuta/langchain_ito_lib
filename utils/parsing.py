import json
import re
from typing import Any


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any]:
    """Best-effort JSON object parser.

    - Accepts raw JSON
    - Also tolerates extra text/code fences by extracting the first {...} block.
    """

    cleaned = text.strip()

    # Strip common fenced code blocks
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned).strip()

    # Try direct parse first
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    # Extract first JSON object-looking block
    match = _JSON_BLOCK_RE.search(cleaned)
    if not match:
        raise ValueError("No JSON object found in model output")

    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Parsed JSON is not an object")
    return value
