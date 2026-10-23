"""Small, shared helpers for parsing provider structured output."""

import json
import re


def extract_json_object(response):
    if not isinstance(response, str):
        raise json.JSONDecodeError("response is not text", repr(response), 0)
    cleaned = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.I | re.S)
    if fenced:
        cleaned = fenced.group(1).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start:end + 1]
    return json.loads(cleaned)


def build_repair_prompt(original_prompt, response, instruction, max_response_chars=4000):
    """Give the repair call the original task and the reply that failed to parse."""
    previous = response if isinstance(response, str) else repr(response)
    return (
        f"{original_prompt}\n\n"
        "Your previous reply could not be parsed as JSON:\n"
        f"{previous[:max_response_chars]}\n\n"
        f"{instruction}"
    )


def parse_with_one_repair(response, repair_prompt, ask, log, stage):
    try:
        payload = extract_json_object(response)
        if not isinstance(payload, dict):
            raise ValueError("top-level structured output was not an object")
        return payload, 0, []
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        log(f"{stage} parse failed; requesting one JSON repair: {type(exc).__name__}")
        try:
            repair_response = ask(repair_prompt)
        except Exception as repair_exc:
            log(f"{stage} JSON repair call failed: {type(repair_exc).__name__}")
            return None, 1, [{"stage": stage, "type": "repair_call_failure", "message": str(repair_exc)}]
        try:
            payload = extract_json_object(repair_response)
            if not isinstance(payload, dict):
                raise ValueError("top-level structured output was not an object")
            return payload, 1, []
        except (TypeError, ValueError, json.JSONDecodeError) as repair_exc:
            log(f"{stage} JSON repair failed: {type(repair_exc).__name__}")
            return None, 1, [{"stage": stage, "type": "invalid_json", "message": str(repair_exc)}]
