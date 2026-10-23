import json
import re
from pathlib import Path
from time import perf_counter

from .schemas import Claim, ConflictResult
from ..telemetry import log
from .structured_json import build_repair_prompt, extract_json_object, parse_with_one_repair


AUTHORITY_SENSITIVE_TERMS = {
    "actual",
    "actually",
    "authoritative",
    "canonical",
    "correct",
    "exact",
    "official",
    "precise",
    "real",
    "true",
    "ultimately",
}

CONFLICT_TERMS = {
    "conflict",
    "conflicting",
    "contradict",
    "contradiction",
    "corrects",
    "disagree",
    "disagrees",
    "disputes",
    "disputed",
    "false",
    "instead",
    "legend",
    "popular",
    "rejects",
    "rumor",
    "uncertain",
}

REFERENCE_TERMS = {
    "annal",
    "annals",
    "archive",
    "archival",
    "codex",
    "consult",
    "decree",
    "ledger",
    "official",
    "record",
    "records",
    "register",
    "registry",
    "transcript",
}


class EvidenceAdjudicator:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self._cache = {}

    def should_adjudicate(self, question, documents, sufficiency_result):
        if not documents:
            return False
        text = _combined_text(question, documents, sufficiency_result)
        terms = set(_tokens(text))
        question_terms = set(_tokens(question))

        usable_documents = [
            document for document in documents
            if getattr(document, "entity_match", None) is None
            or getattr(document.entity_match, "entity_match", "") in {
                "exact_match", "alias_match", "valid_contextual_match"
            }
        ]
        if (
            len(_source_families(usable_documents)) > 1
            and (terms & CONFLICT_TERMS or _has_distinct_value_claims(usable_documents))
        ):
            return True
        # Authority-sensitive wording raises the threshold for sufficiency;
        # it does not by itself justify an adjudication call.
        if sufficiency_result and (
            _mentions_conflict(sufficiency_result.reason)
            or any(_mentions_conflict(item) for item in sufficiency_result.missing)
            or any(_mentions_conflict(query) for query in sufficiency_result.search_queries)
        ):
            return True
        return False

    def adjudicate(self, question, documents, sufficiency_result=None):
        cache_key = (
            question,
            tuple(document.id for document in documents),
            getattr(sufficiency_result, "reason", ""),
        )
        if cache_key in self._cache:
            log("Conflict detection reused cached result")
            return self._cache[cache_key]
        prompt = build_adjudication_prompt(question, documents, sufficiency_result)
        started = perf_counter()
        response = self._ask(prompt, "adjudication")
        self.last_duration = perf_counter() - started
        log(f"Conflict detection: {self.last_duration:.2f}s")
        repair_prompt = build_repair_prompt(
            prompt,
            response,
            "Return only valid JSON matching the adjudication schema. No markdown or commentary.",
        )
        data, retries, errors = parse_with_one_repair(
            response, repair_prompt, lambda value: self._ask(value, "adjudication_repair"), log, "adjudication"
        )
        self.last_retry_count = retries
        result = _parse_adjudication_data(data, documents, errors)
        self._cache[cache_key] = result
        return result

    def _ask(self, prompt, purpose):
        try:
            return self.llm_client.ask(prompt, purpose=purpose)
        except TypeError:
            return self.llm_client.ask(prompt)


def build_adjudication_prompt(question, documents, sufficiency_result=None):
    evidence_text = ""
    for document in documents:
        entity_match = getattr(document, "entity_match", None)
        entity_text = ""
        if entity_match is not None:
            entity_text = f"""
[ENTITY MATCH: {entity_match.entity_match}]
[REQUESTED ENTITY: {entity_match.requested_entity}]
[EVIDENCE ENTITY: {entity_match.evidence_entity}]
[USABLE FOR DIRECT CLAIM: {entity_match.usable_for_direct_claim}]
"""
        evidence_text += f"""
[EVIDENCE ID: {document.id}]
[SOURCE: {document.source_doc}]
[SOURCE FAMILY: {source_family_id(document.source_doc)}]
[CHUNK: {document.chunk_index}]
{entity_text}

{document.text}

"""

    sufficiency_text = ""
    if sufficiency_result is not None:
        sufficiency_text = f"""
CURRENT SUFFICIENCY VIEW:
sufficient: {sufficiency_result.sufficient}
missing: {sufficiency_result.missing}
search_queries: {sufficiency_result.search_queries}
reason: {sufficiency_result.reason}
"""

    return f"""
You are the evidence adjudicator in a retrieval-augmented QA system.

Use only the supplied evidence. Do not answer from outside knowledge.
Do not reveal private reasoning. Return concise evidence-analysis summaries.

ORIGINAL QUESTION:
{question}

{sufficiency_text}
SUPPLIED EVIDENCE:
{evidence_text}

TASK:
1. Determine whether the requested fact has materially conflicting candidate
   answers in the supplied evidence.
2. If no material conflict exists, set has_conflict=false.
3. If conflict exists, identify candidate values and which evidence supports
   each candidate.
3a. Before comparing candidate values, validate whether each evidence block
    makes a claim about the exact requested entity. Evidence marked related,
    ambiguous, or mismatch must not contribute candidate values unless the
    original question explicitly asks about a relationship between entities.
4. Infer authority from the evidence and metadata. Treat these as signals, not
   fixed rankings: explicit official/canonical/primary/contemporary language,
   directness, correction of alternatives, corroboration by independent source
   families, records another source recommends consulting, and relevant time or
   edition context. Treat hearsay, rumor, legend, popular retelling,
   speculation, and uncertain memory as weaker signals when the evidence says so.
5. PDF and DOCX versions with the same source-family identifier are one source
   family for corroboration unless the evidence shows distinct editions/content.
6. If a source points to another record that has not been retrieved, recommend
   targeted searches for that record before resolving.
7. Do not mark a conflict resolved merely because conflict exists.
8. Mark resolved=true only if one candidate is clearly better supported or the
   evidence explicitly establishes genuine unresolved uncertainty.

Return ONLY valid JSON with exactly this shape:

{{
  "has_conflict": false,
  "claim": "",
  "candidate_values": [],
  "supporting_evidence_ids": {{}},
  "authority_notes": [],
  "recommended_search_queries": [],
  "resolvable": false,
  "resolved": false,
  "selected_value": "",
  "selected_evidence_ids": [],
  "needs_more_search": false,
  "claims": [],
  "reason": ""
}}
"""


def parse_adjudication_response(response, documents):
    try:
        data = extract_json_object(response)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        log(f"Conflict response ignored: invalid JSON ({type(exc).__name__})")
        return _fallback_conflict_result([{"stage": "adjudication", "type": "invalid_json", "message": str(exc)}])

    return _parse_adjudication_data(data, documents)


def _parse_adjudication_data(data, documents, internal_errors=None):
    internal_errors = internal_errors or []

    if not isinstance(data, dict):
        log("Conflict response ignored: top-level payload was not an object")
        return _fallback_conflict_result([{"stage": "adjudication", "type": "invalid_schema", "message": "top-level payload was not an object"}])

    required = [
        "has_conflict",
        "claim",
        "candidate_values",
        "authority_notes",
        "recommended_search_queries",
        "resolvable",
        "resolved",
        "selected_value",
        "selected_evidence_ids",
        "reason",
    ]
    missing_fields = [field for field in required if field not in data]
    if "supporting_evidence_ids" not in data:
        log("Conflict response omitted supporting_evidence_ids; defaulting to empty")
    if missing_fields:
        log(f"Conflict response missing fields normalized: {missing_fields}")

    allowed_ids = {document.id for document in documents}
    supporting = _string_list_dict(data.get("supporting_evidence_ids"), allowed_ids)
    selected = _string_list(data.get("selected_evidence_ids"), allowed_ids=allowed_ids)

    result = ConflictResult(
        has_conflict=_strict_bool(data.get("has_conflict", False)),
        claim=_safe_text(data.get("claim")),
        candidate_values=_string_list(data.get("candidate_values")),
        supporting_evidence_ids=supporting,
        authority_notes=_string_list(data.get("authority_notes")),
        recommended_search_queries=_string_list(data.get("recommended_search_queries")),
        resolvable=_strict_bool(data.get("resolvable", False)),
        resolved=_strict_bool(data.get("resolved", False)),
        selected_value=_safe_text(data.get("selected_value")),
        selected_evidence_ids=selected,
        reason=_safe_text(data.get("reason")),
        needs_more_search=_strict_bool(data.get("needs_more_search", False)),
        claims=_claims_from_payload(data.get("claims"), documents),
        parse_error=bool(internal_errors or missing_fields),
        internal_errors=internal_errors,
    )
    if result.parse_error:
        return _fallback_conflict_result(internal_errors or [{"stage": "adjudication", "type": "invalid_schema", "message": "required fields were missing"}])
    if result.resolved:
        if not result.selected_value or result.selected_value not in result.candidate_values:
            return _fallback_conflict_result([{"stage": "adjudication", "type": "invalid_decision", "message": "selected value was not a candidate"}])
        selected_support = set(result.selected_evidence_ids)
        if not selected_support.intersection(
            set(result.supporting_evidence_ids.get(result.selected_value, []))
        ):
            return _fallback_conflict_result([{"stage": "adjudication", "type": "invalid_decision", "message": "selected evidence did not support selected value"}])
    return result


def source_family_id(source_doc):
    path = Path(str(source_doc).replace("\\", "/"))
    suffix = path.suffix.lower()
    # Format copies in different folders are still one work when their
    # normalized document name is the same.
    stem = path.stem if suffix else path.name
    stem = re.sub(r"(?i)(?:[._-]?scan)$", "", stem)
    stem = re.sub(r"(?i)(?:[._ -]*(?:copy|duplicate|\(\d+\)))+$", "", stem)
    return stem.lower()


def distinct_source_family_count(documents):
    return len(_source_families(documents))


def _source_families(documents):
    return {source_family_id(document.source_doc) for document in documents}


def _combined_text(question, documents, sufficiency_result):
    parts = [question]
    parts.extend(document.text for document in documents)
    if sufficiency_result is not None:
        parts.append(sufficiency_result.reason)
        parts.extend(sufficiency_result.missing)
        parts.extend(sufficiency_result.search_queries)
    return "\n".join(parts)


def _mentions_conflict(text):
    return bool(set(_tokens(text)) & CONFLICT_TERMS)


def _has_distinct_value_claims(documents):
    values = set()
    for document in documents:
        values.update(re.findall(r"\b\d+(?:\.\d+)?(?:\s*[A-Za-z]{1,6})?\b", document.text))
        for match in re.finditer(
            r"\b(?:is|was|were|equals?|means?|as)\s+(['\"]?)([A-Za-z][A-Za-z0-9_-]{1,30})\1\b",
            document.text,
            flags=re.IGNORECASE,
        ):
            value = match.group(2).lower()
            if value not in {"a", "an", "the", "not", "no", "disputed", "unknown"}:
                values.add(value)
    return len(values) >= 2


def _claims_from_payload(value, documents):
    if not isinstance(value, list):
        return []
    allowed_ids = {document.id for document in documents}
    claims = []
    for item in value:
        if not isinstance(item, dict):
            continue
        evidence_id = item.get("evidence_id")
        if not isinstance(evidence_id, str) or evidence_id not in allowed_ids:
            continue
        subject = item.get("subject")
        predicate = item.get("predicate")
        if not isinstance(subject, str) or not subject.strip() or not isinstance(predicate, str) or not predicate.strip():
            continue
        raw_value = item.get("value")
        claim_value = raw_value.strip() if isinstance(raw_value, str) else None
        references = item.get("references")
        claims.append(Claim(
            subject=subject.strip(),
            predicate=predicate.strip(),
            value=claim_value,
            claim_type=item.get("claim_type", "direct") if isinstance(item.get("claim_type", "direct"), str) else "direct",
            certainty=item.get("certainty", "unknown") if isinstance(item.get("certainty", "unknown"), str) else "unknown",
            evidence_id=evidence_id,
            source_family_id=source_family_id(next(document.source_doc for document in documents if document.id == evidence_id)),
            references=[reference for reference in references if isinstance(reference, str)] if isinstance(references, list) else [],
        ))
    return claims


def _tokens(text):
    return re.findall(r"\w+", str(text).lower())


_extract_json = extract_json_object


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


def _string_list_dict(value, allowed_ids):
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, ids in value.items():
        result[str(key)] = _string_list(ids, allowed_ids)
    return result


def _safe_text(value):
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _strict_bool(value):
    return value if isinstance(value, bool) else False


def _fallback_conflict_result(internal_errors=None):
    return ConflictResult(
        has_conflict=False,
        claim="",
        candidate_values=[],
        supporting_evidence_ids={},
        authority_notes=[],
        recommended_search_queries=[],
        resolvable=False,
        resolved=False,
        selected_value="",
        selected_evidence_ids=[],
        reason="Conflict adjudication response was invalid JSON.",
        parse_error=True,
        internal_errors=internal_errors or [],
    )
