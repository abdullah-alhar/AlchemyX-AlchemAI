import hashlib
from time import perf_counter
from pathlib import Path

import chromadb

INDEX_SCHEMA_VERSION = 6

try:
    from .Retrieval_Result import RetrievalResult
    from .embeddings import VoyageEmbeddingFunction
    from .chunking import chunk_text
    from .document_metadata import build_searchable_text
    from ..config import (
        DEFAULT_CHROMA_DIRECTORY,
        DEFAULT_COLLECTION_NAME,
        EMBEDDING_BATCH_SIZE,
    )
    from ..telemetry import log
except ImportError:
    from Retrieval_Result import RetrievalResult
    from embeddings import VoyageEmbeddingFunction
    from chunking import chunk_text
    from document_metadata import build_searchable_text
    from src.alchemyx.config import (
        DEFAULT_CHROMA_DIRECTORY,
        DEFAULT_COLLECTION_NAME,
        EMBEDDING_BATCH_SIZE,
    )
    from src.alchemyx.telemetry import log


class RetrievalPipeline:
    def __init__(
        self,
        api_key,
        collection_name=DEFAULT_COLLECTION_NAME,
        persist_directory=DEFAULT_CHROMA_DIRECTORY,
    ):
        # Setting up the voyage embedding function, used for documents being stored
        self.document_ef = VoyageEmbeddingFunction(api_key=api_key, input_type="document")
        self.query_ef = VoyageEmbeddingFunction(api_key=api_key, input_type="query")

        # Setting up persistent chromadb client and collection, using voyage for embeddings
        self.persist_directory = Path(persist_directory)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.document_ef,
            metadata={"hnsw:space": "cosine"}

        )
        self._last_timing = {}

    def add_document(self, doc_id, text, chunk_size=300, overlap=50, metadata=None):

        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
            document_type=(metadata or {}).get("document_type"),
        )
        searchable_chunks = [
            build_searchable_text(chunk, metadata)
            for chunk in chunks
        ]

        chunk_ids = [f"{doc_id}_chunk{i}" for i in range(len(chunks))]

        # Store which document each chunk came from — useful later for citing sources
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        metadatas = [
            {
                **(metadata or {}),
                "source_doc": doc_id,
                "chunk_index": i,
                "content_hash": content_hash,
                "index_schema_version": INDEX_SCHEMA_VERSION,
            }
            for i in range(len(chunks))
        ]

        # Re-indexing a document replaces its old chunks instead of accumulating
        # stale chunk ids/content across runs or repeated ingestion calls.
        self.collection.delete(where={"source_doc": doc_id})

        if not chunks:
            print(f"Removed chunks for empty document '{doc_id}'")
            return

        for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            end = start + EMBEDDING_BATCH_SIZE
            self.collection.add(
                documents=searchable_chunks[start:end],
                ids=chunk_ids[start:end],
                metadatas=metadatas[start:end]
            )

        print(f"Added {len(chunks)} chunks from '{doc_id}'")

    def delete_documents_except(self, source_docs):
        source_docs = set(source_docs)
        records = self.collection.get(include=["metadatas"])
        ids = records.get("ids") or []
        metadatas = records.get("metadatas") or []
        ids_to_delete = []
        removed_source_docs = set()

        for index, result_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            source_doc = metadata.get("source_doc", "")
            if source_doc not in source_docs:
                ids_to_delete.append(result_id)
                removed_source_docs.add(source_doc or result_id)

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)

        return sorted(removed_source_docs)

    def has_document(self, doc_id, text=None):
        """Return whether Chroma already contains the current document version."""
        records = self.collection.get(
            where={"source_doc": doc_id},
            limit=1,
            include=["metadatas"],
        )
        ids = records.get("ids") or []
        if not ids:
            return False

        metadatas = records.get("metadatas") or []
        metadata = metadatas[0] if metadatas else {}
        if metadata.get("index_schema_version") != INDEX_SCHEMA_VERSION:
            return False

        if text is None:
            return True

        stored_hash = metadata.get("content_hash")
        if stored_hash is None:
            # Entries created before content hashes were introduced are still
            # valid persistent index entries; the build step can hydrate BM25
            # from the current extracted text without re-embedding them.
            return True
        current_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return stored_hash == current_hash

    def query(self, question, n_results=3) -> list[RetrievalResult]:
        started = perf_counter()
        query_embeddings = self.query_ef([question])
        embedding_elapsed = perf_counter() - started
        chroma_started = perf_counter()
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results
        )
        self._last_timing = {
            "Embedding": embedding_elapsed,
            "Chroma retrieval": perf_counter() - chroma_started,
        }
        log(f"Embedding: {embedding_elapsed:.2f}s")
        log(f"Chroma retrieval: {self._last_timing['Chroma retrieval']:.2f}s")
        return self._to_retrieval_results(results)

    def search_many(self, queries, n_results=5) -> list[RetrievalResult]:
        queries = list(queries)
        if not queries:
            return []

        query_embeddings = self.query_ef(queries)
        raw_results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results
        )

        results_by_id = {}

        for result in self._to_retrieval_results(raw_results):
            existing_result = results_by_id.get(result.id)
            if existing_result is not None and existing_result.score >= result.score:
                continue

            results_by_id[result.id] = result

        return sorted(
            results_by_id.values(),
            key=lambda result: result.score,
            reverse=True,
        )

    @staticmethod
    def _to_retrieval_results(raw_results) -> list[RetrievalResult]:
        retrieval_results = []

        ids_by_query = raw_results.get("ids") or []
        documents_by_query = raw_results.get("documents") or []
        metadatas_by_query = raw_results.get("metadatas") or []
        distances_by_query = raw_results.get("distances") or []

        for query_index, ids in enumerate(ids_by_query):
            documents = (
                documents_by_query[query_index]
                if query_index < len(documents_by_query)
                else []
            )
            metadatas = (
                metadatas_by_query[query_index]
                if query_index < len(metadatas_by_query)
                else []
            )
            distances = (
                distances_by_query[query_index]
                if query_index < len(distances_by_query)
                else []
            )

            for result_index, result_id in enumerate(ids):
                metadata = (
                    metadatas[result_index]
                    if result_index < len(metadatas) and metadatas[result_index]
                    else {}
                )
                distance = (
                    distances[result_index]
                    if result_index < len(distances)
                    else 0.0
                )
                score = 1.0 - distance if distance is not None else 0.0

                retrieval_results.append(
                    RetrievalResult(
                        id=result_id,
                        text=(
                            documents[result_index]
                            if result_index < len(documents)
                            else ""
                        ),
                        source_doc=metadata.get("source_doc", ""),
                        chunk_index=metadata.get("chunk_index", 0),
                        score=score,
                    )
                )
        return retrieval_results
