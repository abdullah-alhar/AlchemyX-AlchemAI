import json
from time import perf_counter

from .prompts import build_sufficiency_prompt
from .schemas import SufficiencyResult
from .structured_json import build_repair_prompt, extract_json_object, parse_with_one_repair


class SufficiencyChecker:

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def check(self, question, documents):

        prompt = build_sufficiency_prompt(
            question,
            documents
        )

        started = perf_counter()
        response = self._ask(prompt, "sufficiency")
        from ..telemetry import log
        self.last_duration = perf_counter() - started
        log(f"Sufficiency check: {self.last_duration:.2f}s")

        repair_prompt = build_repair_prompt(
            prompt,
            response,
            "Return only valid JSON matching the sufficiency schema. No markdown "
            "or commentary.",
        )
        data, retries, errors = parse_with_one_repair(
            response, repair_prompt, lambda value: self._ask(value, "sufficiency_repair"), log, "sufficiency"
        )
        self.last_retry_count = retries
        return self._parse_data(data, documents, errors)

    def _ask(self, prompt, purpose):
        try:
            return self.llm_client.ask(prompt, purpose=purpose)
        except TypeError:
            return self.llm_client.ask(prompt)

    def _parse_data(self, data, documents, internal_errors=None):
        internal_errors = internal_errors or []
        if data is None:
            return _fallback_sufficiency_result(internal_errors)
        if not isinstance(data, dict):
            return _fallback_sufficiency_result([{"stage": "sufficiency", "type": "invalid_schema", "message": "top-level payload was not an object"}])

        required_fields = [
            "sufficient",
            "missing",
            "search_queries",
            "evidence_ids",
            "reason"
        ]

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            from ..telemetry import log
            log(f"Sufficiency response missing fields: {missing_fields}")
            internal_errors = list(internal_errors) + [{
                "stage": "sufficiency",
                "type": "invalid_schema",
                "message": "required fields were missing",
            }]

        allowed_ids = {document.id for document in documents}
        raw_evidence_ids = _string_list(data.get("evidence_ids"))
        evidence_ids = _string_list(raw_evidence_ids, allowed_ids=allowed_ids)
        sufficient = data.get("sufficient", False)
        if not isinstance(sufficient, bool):
            sufficient = False
        if sufficient and raw_evidence_ids and not evidence_ids:
            sufficient = False
        if sufficient and missing_fields:
            sufficient = False

        return SufficiencyResult(
            sufficient=sufficient,
            missing=_string_list(data.get("missing")) or (["usable sufficiency assessment"] if missing_fields else []),
            search_queries=_string_list(data.get("search_queries")),
            evidence_ids=evidence_ids,
            reason=_safe_text(data.get("reason")) or "Sufficiency response was malformed.",
            parse_error=bool(internal_errors or missing_fields),
            internal_errors=internal_errors,
        )

    _extract_json = staticmethod(extract_json_object)


def _string_list(value, allowed_ids=None):
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        return []

    result = []
    for item in values:
        if not isinstance(item, str):
            continue
        if allowed_ids is not None and item not in allowed_ids:
            continue
        if item not in result:
            result.append(item)
    return result


def _safe_text(value):
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _fallback_sufficiency_result(internal_errors=None):
    return SufficiencyResult(
        sufficient=False,
        missing=[],
        search_queries=[],
        evidence_ids=[],
        reason="Sufficiency response was invalid JSON.",
        parse_error=True,
        internal_errors=internal_errors or [{"stage": "sufficiency", "type": "invalid_json", "message": "structured output could not be parsed"}],
    )
