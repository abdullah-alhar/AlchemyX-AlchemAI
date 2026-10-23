from time import perf_counter
from uuid import uuid4

from ..config import MAX_AGENT_ITERATIONS
from ..telemetry import log, reset_run_id, set_run_id
from .evidence_adjudicator import distinct_source_family_count
from .entity_resolution import EntityResolver, annotate_documents
from .followup import build_contested_followup
from .schemas import SufficiencyResult


class AgentLoop:
    MAX_ITERATIONS = MAX_AGENT_ITERATIONS

    def __init__(
        self,
        retrieval_pipeline,
        sufficiency_checker,
        evidence_adjudicator=None,
        entity_resolver=None,
    ):
        self.retrieval_pipeline = retrieval_pipeline
        self.sufficiency_checker = sufficiency_checker
        self.evidence_adjudicator = evidence_adjudicator
        self.entity_resolver = entity_resolver or EntityResolver()

    def run(self, question, run_id=None, progress=None):
        original_question = question
        research_run_id = run_id or uuid4().hex[:6]
        token = set_run_id(research_run_id)
        started = perf_counter()
        all_documents = []
        seen_document_ids = set()
        current_query = original_question
        pending_queries = []
        previous_queries = set()
        result = None
        adjudication_mode = False
        timeline = []
        _notify(progress, "Extracting key entities from the question.")
        entity_started = perf_counter()
        entities = self.entity_resolver.extract(original_question)
        entity_extraction_seconds = perf_counter() - entity_started
        entity_validation_seconds = 0.0
        entity_matches = {}
        try:
            for iteration in range(self.MAX_ITERATIONS):
                iteration_started = perf_counter()
                log(f"ITERATION {iteration + 1}")
                log(f"Query: {current_query}")
                _notify(progress, f"Iteration {iteration + 1}: retrieving candidate evidence.")
                retrieval_started = perf_counter()
                retrieved = self._retrieve(current_query, adjudication_mode, entities)
                if adjudication_mode:
                    log(f"Conflict-resolution search: {perf_counter() - retrieval_started:.2f}s")
                _notify(progress, f"Iteration {iteration + 1}: retrieved {len(retrieved)} candidate chunks.")
                query_stats = self._query_stats(retrieved)
                new_documents = []
                for document in retrieved:
                    if document.id not in seen_document_ids:
                        all_documents.append(document)
                        seen_document_ids.add(document.id)
                        new_documents.append(document)
                validation_started = perf_counter()
                entity_matches.update(
                    self.entity_resolver.validate_documents(entities, all_documents)
                )
                entity_validation_seconds += perf_counter() - validation_started
                annotate_documents(all_documents, entity_matches)
                usable_documents = self.entity_resolver.direct_documents(
                    original_question,
                    all_documents,
                    entity_matches,
                )
                _notify(progress, f"Iteration {iteration + 1}: checking evidence sufficiency.")

                if not retrieved and not all_documents:
                    _notify(progress, "No matching evidence was found in the archive.")
                    timeline.append(
                        self._retrieval_stop_timeline_entry(
                            iteration=iteration + 1,
                            query=current_query,
                            retrieved=retrieved,
                            new_documents=new_documents,
                            all_documents=all_documents,
                            result=result,
                            query_stats=query_stats,
                            entity_matches=entity_matches,
                            stop_reason="no_retrieval_results",
                            elapsed=perf_counter() - iteration_started,
                        )
                    )
                    return self._result(
                        [],
                        None,
                        iteration + 1,
                        "no_retrieval_results",
                        research_run_id,
                        timeline,
                        started,
                        entity_extraction_seconds,
                        entity_validation_seconds,
                    )
                if iteration > 0 and not new_documents:
                    if result is not None and self._result_is_resolved(result):
                        result.sufficient = True
                        _notify(progress, "Evidence is resolved; preparing the answer.")
                        return self._result(
                            all_documents,
                            result,
                            iteration,
                            "sufficient_evidence",
                            research_run_id,
                            timeline,
                            started,
                            entity_extraction_seconds,
                            entity_validation_seconds,
                        )
                    log("No new evidence found; stopping search.")
                    _notify(progress, "No new evidence was found; stopping follow-up search.")
                    timeline.append(
                        self._retrieval_stop_timeline_entry(
                            iteration=iteration + 1,
                            query=current_query,
                            retrieved=retrieved,
                            new_documents=new_documents,
                            all_documents=all_documents,
                            result=result,
                            query_stats=query_stats,
                            entity_matches=entity_matches,
                            stop_reason="no_new_evidence",
                            elapsed=perf_counter() - iteration_started,
                        )
                    )
                    return self._result(
                        all_documents,
                        result,
                        iteration + 1,
                        "no_new_evidence",
                        research_run_id,
                        timeline,
                        started,
                        entity_extraction_seconds,
                        entity_validation_seconds,
                    )

                # Follow-up queries affect retrieval only. The checker always
                # evaluates the exact original user question. Never let an
                # empty direct-entity set be converted into a positive answer
                # by an LLM interpreting unrelated retrieved chunks.
                if not usable_documents and entities:
                    result = SufficiencyResult(
                        sufficient=False,
                        missing=["direct evidence for the requested entity"],
                        search_queries=[],
                        evidence_ids=[],
                        reason="No usable evidence matched the requested entity.",
                    )
                else:
                    result = self.sufficiency_checker.check(original_question, usable_documents)
                sufficiency_seconds = getattr(self.sufficiency_checker, "last_duration", None)
                followup_started = perf_counter()
                result = self._force_contested_followup(
                    original_question,
                    usable_documents,
                    result,
                    entities,
                    previous_queries | {current_query},
                )
                followup_seconds = perf_counter() - followup_started
                _notify(progress, f"Iteration {iteration + 1}: checking for conflicting evidence.")
                result = self._adjudicate_if_needed(original_question, usable_documents, result)
                adjudication_seconds = getattr(self, "_last_adjudication_seconds", None)
                log(f"Sufficient: {result.sufficient}")
                log(f"Missing: {result.missing}")
                log(f"Distinct source families examined: {distinct_source_family_count(all_documents)}")
                log(f"Iteration {iteration + 1} total: {perf_counter() - iteration_started:.2f}s")
                timeline_entry = self._timeline_entry(
                    iteration=iteration + 1,
                    query=current_query,
                    retrieved=retrieved,
                    new_documents=new_documents,
                    all_documents=all_documents,
                    result=result,
                    query_stats=query_stats,
                    entity_matches=entity_matches,
                    sufficiency_seconds=sufficiency_seconds,
                    adjudication_seconds=adjudication_seconds,
                    followup_seconds=followup_seconds,
                    elapsed=perf_counter() - iteration_started,
                )
                timeline.append(timeline_entry)

                if result.sufficient:
                    timeline_entry["stop_reason"] = "sufficient_evidence"
                    _notify(progress, "Sufficient evidence found; preparing the answer.")
                    return self._result(all_documents, result, iteration + 1, "sufficient_evidence", research_run_id, timeline, started, entity_extraction_seconds, entity_validation_seconds)
                if not result.search_queries and not pending_queries:
                    timeline_entry["stop_reason"] = "no_search_query"
                    _notify(progress, "No useful follow-up search was needed.")
                    return self._result(all_documents, result, iteration + 1, "no_search_query", research_run_id, timeline, started, entity_extraction_seconds, entity_validation_seconds)

                timeline_entry["next_queries"] = list(result.search_queries)
                _notify(progress, "Planning follow-up search: " + "; ".join(result.search_queries))
                previous_queries.add(current_query)
                pending_queries.extend(result.search_queries)
                next_query = self._next_query(pending_queries, previous_queries)
                if not next_query:
                    if result.search_queries:
                        timeline_entry["stop_reason"] = "repeated_query"
                        _notify(progress, "Follow-up search repeated a previous query; stopping.")
                        return self._result(all_documents, result, iteration + 1, "repeated_query", research_run_id, timeline, started, entity_extraction_seconds, entity_validation_seconds)
                    timeline_entry["stop_reason"] = "no_search_query"
                    _notify(progress, "No useful follow-up search was available.")
                    return self._result(all_documents, result, iteration + 1, "no_search_query", research_run_id, timeline, started, entity_extraction_seconds, entity_validation_seconds)
                if next_query in previous_queries:
                    timeline_entry["stop_reason"] = "repeated_query"
                    _notify(progress, "Follow-up search repeated a previous query; stopping.")
                    return self._result(all_documents, result, iteration + 1, "repeated_query", research_run_id, timeline, started, entity_extraction_seconds, entity_validation_seconds)
                current_query = next_query
                adjudication_mode = result.conflict_result is not None

            if timeline:
                timeline[-1]["stop_reason"] = "max_iterations"
            _notify(progress, "Maximum research iterations reached; preparing the answer.")
            return self._result(all_documents, result, self.MAX_ITERATIONS, "max_iterations", research_run_id, timeline, started, entity_extraction_seconds, entity_validation_seconds)
        finally:
            log(f"TOTAL: {perf_counter() - started:.2f}s")
            reset_run_id(token)

    def _retrieve(self, query, adjudication_mode, entities):
        if adjudication_mode and hasattr(self.retrieval_pipeline, "query_for_adjudication"):
            try:
                return self.retrieval_pipeline.query_for_adjudication(query, entities=entities)
            except TypeError:
                return self.retrieval_pipeline.query_for_adjudication(query)
        if hasattr(self.retrieval_pipeline, "query_with_context"):
            try:
                return self.retrieval_pipeline.query_with_context(query, entities=entities)
            except TypeError:
                return self.retrieval_pipeline.query_with_context(query)
        return self.retrieval_pipeline.query(query)

    def _adjudicate_if_needed(self, question, documents, sufficiency_result):
        self._last_adjudication_seconds = None
        if self.evidence_adjudicator is None:
            return sufficiency_result
        if not self.evidence_adjudicator.should_adjudicate(
            question,
            documents,
            sufficiency_result,
        ):
            return sufficiency_result

        conflict_result = self.evidence_adjudicator.adjudicate(
            question,
            documents,
            sufficiency_result,
        )
        self._last_adjudication_seconds = getattr(
            self.evidence_adjudicator,
            "last_duration",
            None,
        )
        sufficiency_result.conflict_result = conflict_result
        if not conflict_result.has_conflict:
            return sufficiency_result

        usable_ids = {
            document.id
            for document in documents
            if getattr(document, "entity_match", None) is None
            or getattr(document.entity_match, "entity_match", "") in {
                "exact_match", "alias_match", "valid_contextual_match"
            }
        }
        # Adjudication is never allowed to promote excluded evidence. Keep
        # the conflict unresolved so the normal follow-up path can continue.
        conflict_result.selected_evidence_ids = [
            evidence_id
            for evidence_id in conflict_result.selected_evidence_ids
            if evidence_id in usable_ids
        ]
        for value, evidence_ids in list(conflict_result.supporting_evidence_ids.items()):
            conflict_result.supporting_evidence_ids[value] = [
                evidence_id for evidence_id in evidence_ids if evidence_id in usable_ids
            ]
        if conflict_result.resolved and not conflict_result.selected_evidence_ids:
            conflict_result.resolved = False
            conflict_result.selected_value = ""
            conflict_result.recommended_search_queries = list(
                conflict_result.recommended_search_queries
            )

        evidence_ids = conflict_result.evidence_ids() or sufficiency_result.evidence_ids
        if conflict_result.resolved and not getattr(conflict_result, "needs_more_search", False):
            sufficiency_result.sufficient = True
            sufficiency_result.missing = []
            sufficiency_result.search_queries = []
            sufficiency_result.evidence_ids = evidence_ids
            sufficiency_result.reason = conflict_result.reason
            return sufficiency_result

        sufficiency_result.sufficient = False
        sufficiency_result.evidence_ids = evidence_ids
        sufficiency_result.missing = [
            conflict_result.claim or "unresolved conflicting evidence"
        ]
        sufficiency_result.search_queries = conflict_result.recommended_search_queries
        sufficiency_result.reason = (
            conflict_result.reason
            or "Conflicting evidence requires targeted follow-up before answering."
        )
        return sufficiency_result

    @staticmethod
    def _result_is_resolved(result):
        conflict = getattr(result, "conflict_result", None)
        return bool(
            result.sufficient
            or (
                conflict is not None
                and conflict.resolved
                and not getattr(conflict, "needs_more_search", False)
                and bool(conflict.selected_value)
                and not getattr(conflict, "parse_error", False)
            )
        )

    def _force_contested_followup(
        self,
        question,
        documents,
        sufficiency_result,
        entities,
        previous_queries,
    ):
        if sufficiency_result.sufficient and len(previous_queries) > 1:
            return sufficiency_result
        followup = build_contested_followup(
            question=question,
            documents=documents,
            entities=entities,
            previous_queries=previous_queries,
        )
        if not followup.required:
            return sufficiency_result
        if not followup.search_queries:
            return sufficiency_result

        conflict_result = sufficiency_result.conflict_result
        unresolved_is_supported = bool(
            conflict_result
            and conflict_result.has_conflict
            and conflict_result.resolved
            and conflict_result.selected_value.lower() in {"unknown", "unresolved", "contested"}
        )
        if unresolved_is_supported:
            return sufficiency_result

        sufficiency_result.sufficient = False
        sufficiency_result.missing = [followup.missing]
        sufficiency_result.search_queries = followup.search_queries
        sufficiency_result.reason = followup.reason
        return sufficiency_result

    @staticmethod
    def _next_query(pending_queries, previous_queries):
        while pending_queries:
            query = pending_queries.pop(0)
            if query not in previous_queries:
                return query
        return ""

    def _query_stats(self, retrieved):
        stats = getattr(self.retrieval_pipeline, "last_query_stats", None)
        if isinstance(stats, dict):
            return stats
        return {
            "candidate_count": len(retrieved),
            "reranked_count": len(retrieved),
        }

    @staticmethod
    def _timeline_entry(
        iteration,
        query,
        retrieved,
        new_documents,
        all_documents,
        result,
        query_stats,
        entity_matches,
        sufficiency_seconds,
        adjudication_seconds,
        followup_seconds,
        elapsed,
    ):
        conflict_result = result.conflict_result
        conflict_detected = bool(conflict_result and conflict_result.has_conflict)
        conflict_resolved = bool(conflict_result and conflict_result.resolved)
        return {
            "iteration": iteration,
            "query": query,
            "retrieved": query_stats.get("candidate_count", len(retrieved)),
            "reranked": query_stats.get("reranked_count", len(retrieved)),
            "distinct_sources": distinct_source_family_count(all_documents),
            "entity_matches": _entity_match_counts(entity_matches),
            "new_evidence": len(new_documents),
            "new_evidence_summary": _summarize_new_evidence(new_documents),
            "retrieval_seconds": query_stats.get("retrieval_seconds"),
            "rerank_seconds": query_stats.get("rerank_seconds"),
            "sufficiency_seconds": sufficiency_seconds,
            "adjudication_seconds": adjudication_seconds,
            "followup_seconds": followup_seconds,
            "conflict_detected": conflict_detected,
            "conflict_resolved": conflict_resolved,
            "candidate_values": (
                list(conflict_result.candidate_values)
                if conflict_result is not None
                else []
            ),
            "missing": list(result.missing),
            "next_queries": list(result.search_queries),
            "sufficient": result.sufficient,
            "stop_reason": "",
            "elapsed_seconds": elapsed,
        }

    @staticmethod
    def _retrieval_stop_timeline_entry(
        iteration,
        query,
        retrieved,
        new_documents,
        all_documents,
        result,
        query_stats,
        entity_matches,
        stop_reason,
        elapsed,
    ):
        conflict_result = getattr(result, "conflict_result", None)
        conflict_detected = bool(conflict_result and conflict_result.has_conflict)
        conflict_resolved = bool(conflict_result and conflict_result.resolved)
        return {
            "iteration": iteration,
            "query": query,
            "retrieved": query_stats.get("candidate_count", len(retrieved)),
            "reranked": query_stats.get("reranked_count", len(retrieved)),
            "distinct_sources": distinct_source_family_count(all_documents),
            "entity_matches": _entity_match_counts(entity_matches),
            "new_evidence": len(new_documents),
            "new_evidence_summary": _summarize_new_evidence(new_documents),
            "retrieval_seconds": query_stats.get("retrieval_seconds"),
            "rerank_seconds": query_stats.get("rerank_seconds"),
            "sufficiency_seconds": None,
            "adjudication_seconds": None,
            "followup_seconds": None,
            "conflict_detected": conflict_detected,
            "conflict_resolved": conflict_resolved,
            "candidate_values": (
                list(conflict_result.candidate_values)
                if conflict_result is not None
                else []
            ),
            "missing": list(getattr(result, "missing", []) or []),
            "next_queries": [],
            "sufficient": bool(getattr(result, "sufficient", False)),
            "stop_reason": stop_reason,
            "elapsed_seconds": elapsed,
        }

    @staticmethod
    def _result(
        documents,
        result,
        iterations,
        stop_reason,
        research_run_id,
        timeline=None,
        started=None,
        entity_extraction_seconds=None,
        entity_validation_seconds=None,
    ):
        agent_loop_seconds = perf_counter() - started if started is not None else None
        return {
            "documents": documents,
            "result": result,
            "iterations": iterations,
            "stop_reason": stop_reason,
            "research_run_id": research_run_id,
            "timeline": timeline or [],
            "timings": {
                "agent_loop": agent_loop_seconds,
                "entity_extraction": entity_extraction_seconds,
                "entity_validation": entity_validation_seconds,
            },
        }


def _summarize_new_evidence(documents):
    if not documents:
        return "No new evidence found."
    source_docs = []
    for document in documents:
        if document.source_doc not in source_docs:
            source_docs.append(document.source_doc)
    if len(source_docs) == 1:
        return f"New evidence from {source_docs[0]}."
    return f"New evidence from {len(source_docs)} sources."


def _entity_match_counts(entity_matches):
    counts = {}
    for match in entity_matches.values():
        counts[match.entity_match] = counts.get(match.entity_match, 0) + 1
    return counts


def _notify(progress, message):
    if progress is None:
        return
    progress(message)
