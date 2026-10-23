from alchemyx.retrieval.bm25_store import BM25Store


def _document(source_doc, text):
    return {"id": f"{source_doc}_chunk0", "text": text, "source_doc": source_doc, "chunk_index": 0}


def test_deferred_writes_save_once_at_the_end(tmp_path):
    path = tmp_path / "bm25.json"
    store = BM25Store(persist_path=path)

    with store.deferred_writes():
        store.replace_documents("a.txt", [_document("a.txt", "ash plains convoy")])
        store.replace_documents("b.txt", [_document("b.txt", "northern gate orders")])
        store.replace_documents("c.txt", [_document("c.txt", "storm roads shift")])
        assert not path.exists()

    assert path.exists()
    reloaded = BM25Store(persist_path=path)
    assert reloaded.search("orders", top_k=1)[0].source_doc == "b.txt"


def test_replace_documents_drops_old_chunks(tmp_path):
    store = BM25Store(persist_path=tmp_path / "bm25.json")
    store.replace_documents("a.txt", [_document("a.txt", "old text")])
    store.replace_documents("a.txt", [_document("a.txt", "new text")])

    assert [document["text"] for document in store.documents] == ["new text"]
