import json
import re
from contextlib import contextmanager
from pathlib import Path

from rank_bm25 import BM25Okapi

try:
    from .Retrieval_Result import RetrievalResult
except ImportError:
    from Retrieval_Result import RetrievalResult

METADATA_QUERY_STOPWORDS = {
    "a",
    "an",
    "archive",
    "are",
    "available",
    "corpus",
    "doc",
    "docs",
    "document",
    "documents",
    "file",
    "files",
    "folder",
    "in",
    "is",
    "related",
    "the",
    "there",
    "what",
    "which",
}


class BM25Store:
    def __init__(self, persist_path=None):
        self.persist_path = Path(persist_path) if persist_path else None
        self.documents = []
        self.tokenized_documents = []
        self.bm25 = None
        self._deferred = False
        self._load()

    @contextmanager
    def deferred_writes(self):
        """Rebuild and save once at the end instead of after every document."""
        if self._deferred:
            yield self
            return
        self._deferred = True
        try:
            yield self
        finally:
            self._deferred = False
            self._commit()

    def add_documents(self, documents):
        self.documents.extend(documents)
        self._commit()

    def replace_documents(self, source_doc, documents):
        self.documents = [
            document
            for document in self.documents
            if document.get("source_doc") != source_doc
        ]
        self.documents.extend(documents)
        self._commit()

    def delete_documents_except(self, source_docs):
        source_docs = set(source_docs)
        removed = {
            document.get("source_doc", "")
            for document in self.documents
            if document.get("source_doc", "") not in source_docs
        }
        self.documents = [
            document
            for document in self.documents
            if document.get("source_doc", "") in source_docs
        ]
        self._commit()
        return sorted(source_doc for source_doc in removed if source_doc)

    def _commit(self):
        if self._deferred:
            return
        self._rebuild()
        self._save()

    def _rebuild(self):
        self.tokenized_documents = [
            self._tokenize(self._search_text(document))
            for document in self.documents
        ]
        self.bm25 = (
            BM25Okapi(self.tokenized_documents)
            if self.tokenized_documents
            else None
        )

    def _load(self):
        if self.persist_path is None or not self.persist_path.exists():
            return

        try:
            data = json.loads(self.persist_path.read_text(encoding="utf-8"))
            documents = data.get("documents", [])
            if isinstance(documents, list):
                self.documents = documents
                self._rebuild()
        except (OSError, json.JSONDecodeError, TypeError, AttributeError):
            self.documents = []
            self._rebuild()

    def _save(self):
        if self.persist_path is None:
            return

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"documents": self.documents}
        self.persist_path.write_text(
            json.dumps(payload, ensure_ascii=True),
            encoding="utf-8",
        )

    def search(self, query, top_k=5):
        if self.bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_results = sorted(
            (
                RetrievalResult(
                    id=document.get("id", ""),
                    text=document.get("text", ""),
                    source_doc=document.get("source_doc", ""),
                    chunk_index=document.get("chunk_index", 0),
                    score=float(score),
                )
                for document, score in zip(self.documents, scores)
            ),
            key=lambda result: result.score,
            reverse=True,
        )

        return ranked_results[:top_k]

    def search_metadata(self, query, top_k=5):
        if not self.documents:
            return []

        tokenized_query = [
            token
            for token in self._tokenize(query)
            if token not in METADATA_QUERY_STOPWORDS
        ]
        if not tokenized_query:
            return self._metadata_listing(top_k)

        scored_by_source_doc = {}

        for document in self.documents:
            metadata_text = document.get("metadata_text", "")
            tokens = self._tokenize(metadata_text)
            score = sum(tokens.count(token) for token in tokenized_query)
            if score <= 0:
                continue

            result = (
                RetrievalResult(
                    id=document.get("id", ""),
                    text=document.get("text", ""),
                    source_doc=document.get("source_doc", ""),
                    chunk_index=document.get("chunk_index", 0),
                    score=float(score),
                )
            )
            existing = scored_by_source_doc.get(result.source_doc)
            if existing is None or existing.score < result.score:
                scored_by_source_doc[result.source_doc] = result

        return sorted(
            scored_by_source_doc.values(),
            key=lambda result: result.score,
            reverse=True,
        )[:top_k]

    def _metadata_listing(self, top_k):
        results = []
        seen_source_docs = set()

        for document in self.documents:
            source_doc = document.get("source_doc", "")
            if source_doc in seen_source_docs:
                continue

            seen_source_docs.add(source_doc)
            results.append(
                RetrievalResult(
                    id=document.get("id", ""),
                    text=document.get("text", ""),
                    source_doc=source_doc,
                    chunk_index=document.get("chunk_index", 0),
                    score=1.0,
                )
            )
            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def _search_text(document):
        return document.get("search_text") or document.get("text", "")

    @staticmethod
    def _tokenize(text):
        return re.findall(r"\w+", text.lower())
