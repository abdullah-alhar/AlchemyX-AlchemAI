import re
from time import perf_counter

try:
    from .Retrieval_Result import RetrievalResult
    from ..config import RRF_K
    from ..telemetry import log
except ImportError:
    from Retrieval_Result import RetrievalResult
    from src.alchemyx.config import RRF_K
    from src.alchemyx.telemetry import log


class HybridSearch:
    def __init__(
        self,
        chroma_store,
        bm25_store,
        document_registry=None,
        chroma_weight=1.0,
        bm25_weight=1.0,
        rrf_k=RRF_K,
        candidate_multiplier=3,
        min_candidates=20,
    ):
        self.chroma_store = chroma_store
        self.bm25_store = bm25_store
        self.document_registry = document_registry
        self.chroma_weight = chroma_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier
        self.min_candidates = min_candidates

    def search(self, query, top_k=5) -> list[RetrievalResult]:
        started = perf_counter()
        candidate_k = self._candidate_k(top_k)
        chroma_results = self._semantic_query(query, candidate_k)
        bm25_started = perf_counter()
        bm25_results = self.bm25_store.search(query, top_k=candidate_k)
        log(f"BM25 retrieval: {perf_counter() - bm25_started:.2f}s")
        metadata_results = self._metadata_search(query, candidate_k)
        registry_results = self._registry_search(query, candidate_k)

        fusion_started = perf_counter()
        fused = reciprocal_rank_fusion(
            ranked_result_lists=[
                (chroma_results, self.chroma_weight),
                (bm25_results, self.bm25_weight),
                (metadata_results, self.bm25_weight),
                (registry_results, self.bm25_weight),
            ],
            top_k=top_k,
            rrf_k=self.rrf_k,
        )
        log(f"RRF fusion: {perf_counter() - fusion_started:.2f}s")
        log(f"Hybrid retrieval: {perf_counter() - started:.2f}s")
        return fused

    def search_many(self, queries, top_k=5) -> list[RetrievalResult]:
        queries = list(queries)
        if not queries:
            return []

        candidate_k = self._candidate_k(top_k)
        ranked_result_lists = []

        for query in queries:
            ranked_result_lists.append(
                (
                    self._semantic_query(query, candidate_k),
                    self.chroma_weight,
                )
            )
            ranked_result_lists.append(
                (self.bm25_store.search(query, top_k=candidate_k), self.bm25_weight)
            )
            ranked_result_lists.append(
                (self._metadata_search(query, candidate_k), self.bm25_weight)
            )
            ranked_result_lists.append(
                (self._registry_search(query, candidate_k), self.bm25_weight)
            )

        return reciprocal_rank_fusion(
            ranked_result_lists=ranked_result_lists,
            top_k=top_k,
            rrf_k=self.rrf_k,
        )

    def _candidate_k(self, top_k):
        return max(top_k * self.candidate_multiplier, self.min_candidates)

    def _semantic_query(self, query, candidate_k):
        try:
            results = self.chroma_store.query(query, n_results=candidate_k)
            log("Semantic retrieval bypassed=false fallback_to_bm25=false")
            return results
        except Exception as exc:
            log(
                "Semantic retrieval bypassed=true "
                f"fallback_to_bm25=true reason={type(exc).__name__}"
            )
            return []

    def _metadata_search(self, query, candidate_k):
        if not _is_metadata_query(query):
            log("Metadata retrieval bypassed=true reason=content_query")
            return []
        if not hasattr(self.bm25_store, "search_metadata"):
            log("Metadata retrieval bypassed=true reason=unsupported_store")
            return []

        started = perf_counter()
        results = self.bm25_store.search_metadata(query, top_k=candidate_k)
        log(f"Metadata retrieval bypassed=false results={len(results)}")
        log(f"Metadata retrieval: {perf_counter() - started:.2f}s")
        return results

    def _registry_search(self, query, candidate_k):
        if not _is_metadata_query(query):
            log("Document registry retrieval bypassed=true reason=content_query")
            return []
        if self.document_registry is None:
            log("Document registry retrieval bypassed=true reason=unavailable")
            return []

        started = perf_counter()
        results = self.document_registry.search(query, top_k=candidate_k)
        log(f"Document registry retrieval bypassed=false results={len(results)}")
        log(f"Document registry retrieval: {perf_counter() - started:.2f}s")
        return results


def reciprocal_rank_fusion(
    ranked_result_lists,
    top_k=5,
    rrf_k=RRF_K,
) -> list[RetrievalResult]:
    merged_by_id = {}

    for results, weight in ranked_result_lists:
        seen_ids = set()

        for rank, result in enumerate(results, start=1):
            if result.id in seen_ids:
                continue

            seen_ids.add(result.id)
            rrf_score = weight / (rrf_k + rank)
            _merge_result(
                merged_by_id=merged_by_id,
                result=result,
                score=rrf_score,
            )

    ranked_results = sorted(
        merged_by_id.values(),
        key=lambda result: result.score,
        reverse=True,
    )

    return ranked_results[:top_k]


def merge_results(
    chroma_results,
    bm25_results,
    top_k=5,
    chroma_weight=1.0,
    bm25_weight=1.0,
    rrf_k=RRF_K,
) -> list[RetrievalResult]:
    return reciprocal_rank_fusion(
        ranked_result_lists=[
            (chroma_results, chroma_weight),
            (bm25_results, bm25_weight),
        ],
        top_k=top_k,
        rrf_k=rrf_k,
    )


def _merge_result(merged_by_id, result, score):
    existing_result = merged_by_id.get(result.id)
    if existing_result is None:
        merged_by_id[result.id] = RetrievalResult(
            id=result.id,
            text=result.text,
            source_doc=result.source_doc,
            chunk_index=result.chunk_index,
            score=score,
        )
        return

    existing_result.score += score


def _is_metadata_query(query):
    terms = set(_tokenize(query))
    return bool(
        terms
        & {
            "archive",
            "available",
            "corpus",
            "csv",
            "doc",
            "docs",
            "document",
            "documents",
            "file",
            "files",
            "folder",
            "github",
            "git",
            "commit",
            "commits",
            "pdf",
            "txt",
            "md",
            "docx",
        }
    )


def _tokenize(text):
    return re.findall(r"\w+", text.lower())
