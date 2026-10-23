from alchemyx.retrieval.chunking import chunk_text


def test_heading_paragraph_keeps_sentence_chunking():
    sentences = " ".join(f"Sentence number {i} is here." for i in range(40))
    text = f"Chapter One\n\n{sentences}"

    chunks = chunk_text(text, chunk_size=50, overlap=10)

    assert chunks[0].startswith("Chapter One Sentence number 0")
    # Sentence-based chunks end on a sentence boundary, word windows do not.
    assert all(chunk.endswith(".") for chunk in chunks)


def test_unpunctuated_single_paragraph_uses_word_windows():
    text = "\n".join(f"name: row{i}; value: {i}" for i in range(60))

    chunks = chunk_text(text, chunk_size=40, overlap=10)

    assert len(chunks) > 1
    assert all(len(chunk.split()) <= 40 for chunk in chunks)
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]


def test_empty_text_has_no_chunks():
    assert chunk_text("   \n\n  ") == []
