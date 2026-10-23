import os
from dotenv import find_dotenv, load_dotenv
try:
    from .bm25_store import BM25Store
    from .chunking import chunk_text
    from .document_metadata import build_searchable_text, metadata_search_text
    from .document_registry import DocumentRegistry
    from .hybrid_search import HybridSearch
    from .reranker import Reranker
    from .retrieval import RetrievalPipeline
    from ..agent.hybrid_rerank_retrieval import HybridRerankRetrievalPipeline
    from ..config import (
        DEFAULT_BM25_PATH,
        DEFAULT_CHUNK_OVERLAP,
        DEFAULT_CHUNK_SIZE,
        DEFAULT_CHROMA_DIRECTORY,
        DEFAULT_DOCUMENT_REGISTRY_PATH,
        MIN_RERANK_SCORE,
        VOYAGE_API_KEY,
        RERANK_CANDIDATES,
        RETRIEVAL_TOP_K,
    )
except ImportError:
    from bm25_store import BM25Store
    from chunking import chunk_text
    from document_metadata import build_searchable_text, metadata_search_text
    from document_registry import DocumentRegistry
    from hybrid_search import HybridSearch
    from reranker import Reranker
    from retrieval import RetrievalPipeline
    from src.alchemyx.agent.hybrid_rerank_retrieval import HybridRerankRetrievalPipeline
    from src.alchemyx.config import (
        DEFAULT_BM25_PATH,
        DEFAULT_CHUNK_OVERLAP,
        DEFAULT_CHUNK_SIZE,
        DEFAULT_CHROMA_DIRECTORY,
        DEFAULT_DOCUMENT_REGISTRY_PATH,
        MIN_RERANK_SCORE,
        VOYAGE_API_KEY,
        RERANK_CANDIDATES,
        RETRIEVAL_TOP_K,
    )

def get_api_key(api_key=None):
    load_dotenv(find_dotenv())
    api_key = api_key or os.getenv("VOYAGE_API_KEY") or VOYAGE_API_KEY
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY is missing. Add it to the .env file.")

    return api_key


def create_retrieval_stack(api_key=None):
    api_key = get_api_key(api_key)
    dotenv_path = find_dotenv()
    project_root = os.path.dirname(dotenv_path) if dotenv_path else os.getcwd()
    pipeline = RetrievalPipeline(
        api_key=api_key,
        persist_directory=os.path.join(project_root, DEFAULT_CHROMA_DIRECTORY),
    )
    bm25_store = BM25Store(
        persist_path=os.path.join(project_root, DEFAULT_BM25_PATH),
    )
    document_registry = DocumentRegistry(
        persist_path=os.path.join(project_root, DEFAULT_DOCUMENT_REGISTRY_PATH),
    )
    hybrid_search = HybridSearch(pipeline, bm25_store, document_registry)
    reranker = Reranker(api_key=api_key)

    return pipeline, bm25_store, hybrid_search, reranker


def create_agent_retrieval_pipeline(api_key=None):
    load_dotenv(find_dotenv())
    pipeline, bm25_store, hybrid_search, reranker = create_retrieval_stack(api_key)
    if not bm25_store.documents:
        raise RuntimeError(
            "BM25 index is missing or empty. Run "
            "scripts/build_indexes.py before starting Streamlit."
        )
    if pipeline.collection.count() == 0:
        raise RuntimeError(
            "Chroma index is missing or empty. Run "
            "scripts/build_indexes.py before starting Streamlit."
        )

    return HybridRerankRetrievalPipeline(
        hybrid_search=hybrid_search,
        reranker=reranker,
        candidate_k=RERANK_CANDIDATES,
        top_k=RETRIEVAL_TOP_K,
        min_relevance_score=MIN_RERANK_SCORE,
    )


def add_document_to_indexes(
    pipeline,
    bm25_store,
    doc_id,
    text,
    chunk_size=DEFAULT_CHUNK_SIZE,
    overlap=DEFAULT_CHUNK_OVERLAP,
    metadata=None,
):
    already_indexed = (
        hasattr(pipeline, "has_document")
        and pipeline.has_document(doc_id, text)
    )
    if not already_indexed:
        pipeline.add_document(
            doc_id=doc_id,
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
            metadata=metadata,
        )

    chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
        document_type=(metadata or {}).get("document_type"),
    )
    bm25_documents = [
        {
            "id": f"{doc_id}_chunk{index}",
            "text": build_searchable_text(chunk, metadata),
            "content_text": chunk,
            "metadata_text": metadata_search_text(metadata),
            "source_doc": doc_id,
            "chunk_index": index,
            **(metadata or {}),
        }
        for index, chunk in enumerate(chunks)
    ]
    if hasattr(bm25_store, "replace_documents"):
        bm25_store.replace_documents(doc_id, bm25_documents)
    else:
        bm25_store.add_documents(bm25_documents)


def print_results(results):
    if not results:
        print("No evidence above relevance threshold.")
        return

    for rank, result in enumerate(results, start=1):
        print(rank, result.id, result.score, result.text[:100])


def find_rank(results, expected_id):
    for rank, result in enumerate(results, start=1):
        if result.id == expected_id:
            return rank

    return None
