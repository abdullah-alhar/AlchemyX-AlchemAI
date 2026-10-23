from alchemyx.retrieval.Retrieval_Result import RetrievalResult
from alchemyx.retrieval.hybrid_search import reciprocal_rank_fusion


def _result(result_id):
    return RetrievalResult(id=result_id, text=result_id, source_doc="doc", chunk_index=0, score=0.0)


def test_rrf_rewards_results_found_by_both_retrievers():
    semantic = [_result("a"), _result("b")]
    keyword = [_result("b"), _result("c")]

    fused = reciprocal_rank_fusion([(semantic, 1.0), (keyword, 1.0)], top_k=3, rrf_k=60)

    assert [result.id for result in fused] == ["b", "a", "c"]
    assert fused[0].score == 1 / 62 + 1 / 61
