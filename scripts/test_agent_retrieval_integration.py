"""Smoke-test agent loop integration with the hybrid retrieval pipeline.

Needs VOYAGE_API_KEY. The demo documents are indexed in a temporary folder,
so the real Chroma/BM25 indexes are never touched.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alchemyx.agent.agent_loop import AgentLoop
from alchemyx.agent.schemas import SufficiencyResult
from alchemyx.agent.hybrid_rerank_retrieval import HybridRerankRetrievalPipeline
from alchemyx.config import MIN_RERANK_SCORE
from alchemyx.retrieval.bm25_store import BM25Store
from alchemyx.retrieval.hybrid_search import HybridSearch
from alchemyx.retrieval.main import add_document_to_indexes, get_api_key
from alchemyx.retrieval.reranker import Reranker
from alchemyx.retrieval.retrieval import RetrievalPipeline

DEMO_DOCUMENTS = {
    "demo/caldrin_orders.txt": (
        "Captain Caldrin received sealed orders at the northern gate. "
        "The orders named Mira Quen as the envoy he must escort to the capital. "
        "Caldrin escorted Mira Quen across the ash plains without losing a rider."
    ),
    "demo/ash_plains.txt": (
        "The ash plains stretch between the northern gate and the capital. "
        "Travellers cross them in convoys because the roads shift after storms."
    ),
}


class DemoSufficiencyChecker:
    def check(self, question, documents):
        evidence_ids = [document.id for document in documents]
        has_answer = any("Mira Quen" in document.text for document in documents)

        return SufficiencyResult(
            sufficient=has_answer,
            missing=[] if has_answer else ["Caldrin escort target"],
            search_queries=[] if has_answer else ["Caldrin escort Mira Quen"],
            evidence_ids=evidence_ids,
            reason="Demo evidence contains Caldrin's escort target."
            if has_answer
            else "Demo evidence has not found Caldrin's escort target yet.",
        )


def add_demo_documents(pipeline, bm25_store):
    for doc_id, text in DEMO_DOCUMENTS.items():
        add_document_to_indexes(pipeline, bm25_store, doc_id, text)


def main():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as index_dir:
        run_smoke_test(Path(index_dir))


def run_smoke_test(index_dir):
    api_key = get_api_key()
    pipeline = RetrievalPipeline(
        api_key=api_key,
        collection_name="alchemyx_smoke_test",
        persist_directory=index_dir / "chroma",
    )
    bm25_store = BM25Store(persist_path=index_dir / "bm25.json")
    hybrid_search = HybridSearch(pipeline, bm25_store)
    reranker = Reranker(api_key=api_key)
    add_demo_documents(pipeline, bm25_store)

    agent_retrieval = HybridRerankRetrievalPipeline(
        hybrid_search=hybrid_search,
        reranker=reranker,
        candidate_k=15,
        top_k=5,
        min_relevance_score=MIN_RERANK_SCORE,
    )
    agent = AgentLoop(agent_retrieval, DemoSufficiencyChecker())
    result = agent.run("Who did Caldrin escort?")

    assert result["documents"], "Agent returned no retrieval documents"
    assert result["stop_reason"] == "sufficient_evidence"
    assert all(hasattr(document, "id") for document in result["documents"])
    assert all(hasattr(document, "text") for document in result["documents"])

    print("integration_ok=true")
    print(f"stop_reason={result['stop_reason']}")
    print(f"iterations={result['iterations']}")
    print(f"documents={len(result['documents'])}")


if __name__ == "__main__":
    main()
