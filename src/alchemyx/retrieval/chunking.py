import re


def chunk_text(
    text,
    chunk_size=300,
    overlap=50,
    min_chunk_size=20,
    document_type=None,
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be 0 or greater")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if min_chunk_size < 0:
        raise ValueError("min_chunk_size must be 0 or greater")

    if document_type == "md":
        return _chunk_markdown(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
        )

    return _chunk_prose(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_size=min_chunk_size,
    )


def _chunk_words(words, chunk_size, overlap, min_chunk_size):
    if not words:
        return []

    chunks = []
    ranges = []
    start = 0
    step = chunk_size - overlap

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]

        if len(chunk_words) >= min_chunk_size or start == 0:
            chunks.append(" ".join(chunk_words))
            ranges.append((start, end))
        else:
            previous_start, previous_end = ranges[-1]
            merge_start = max(previous_end, start)
            if merge_start < end:
                chunks[-1] = " ".join(words[previous_start:end])
                ranges[-1] = (previous_start, end)
            break

        if end == len(words):
            break

        start += step

    return chunks


def _chunk_prose(text, chunk_size, overlap, min_chunk_size):
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]
    if not paragraphs:
        return []

    units = []
    for paragraph in paragraphs:
        # A paragraph without sentence punctuation (a heading, a list, CSV
        # rows) is kept as one unit; oversized units are word-split below.
        if not _has_sentence_boundary(paragraph):
            units.append(paragraph)
            continue
        units.extend(_sentences(paragraph))

    chunks = []
    current = []

    for unit in units:
        unit = " ".join(unit.split())
        if not unit:
            continue

        unit_words = unit.split()
        if len(unit_words) > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current = []
            chunks.extend(
                _chunk_words(
                    unit_words,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    min_chunk_size=min_chunk_size,
                )
            )
            continue

        candidate = [*current, unit]
        if len(" ".join(candidate).split()) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(" ".join(current))
        current = _overlap_tail(current, overlap)
        current.append(unit)

    if current:
        tail = " ".join(current)
        if chunks and len(tail.split()) < min_chunk_size:
            chunks[-1] = f"{chunks[-1]} {tail}"
        else:
            chunks.append(tail)

    return chunks


def _sentences(paragraph):
    collapsed = " ".join(paragraph.split())
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", collapsed)
        if sentence.strip()
    ]


def _has_sentence_boundary(text):
    return bool(re.search(r"[.!?](?:\s|$)", text))


def _overlap_tail(units, overlap):
    if overlap <= 0 or not units:
        return []

    tail = []
    word_count = 0
    for unit in reversed(units):
        unit_words = unit.split()
        if tail and word_count + len(unit_words) > overlap:
            break
        tail.insert(0, unit)
        word_count += len(unit_words)
        if word_count >= overlap:
            break
    return tail


def _chunk_markdown(text, chunk_size, overlap, min_chunk_size):
    chunks = []

    for section in _markdown_sections(text):
        section_words = section.split()
        if not section_words:
            continue
        if len(section_words) <= chunk_size:
            chunks.append(" ".join(section_words))
            continue

        chunks.extend(
            _chunk_markdown_section(
                section,
                chunk_size=chunk_size,
                overlap=overlap,
                min_chunk_size=min_chunk_size,
            )
        )

    return chunks


def _markdown_sections(text):
    sections = []
    current = []

    for line in text.splitlines():
        if re.match(r"^#{1,6}\s+\S", line) and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current).strip())

    return sections


def _chunk_markdown_section(section, chunk_size, overlap, min_chunk_size):
    lines = section.splitlines()
    heading = lines[0].strip() if lines and re.match(r"^#{1,6}\s+\S", lines[0]) else ""
    body = "\n".join(lines[1:] if heading else lines).strip()
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", body)
        if paragraph.strip()
    ]

    chunks = []
    current = [heading] if heading else []

    for paragraph in paragraphs:
        candidate = [*current, paragraph]
        if len(" ".join(candidate).split()) <= chunk_size:
            current = candidate
            continue

        if current and (not heading or current != [heading]):
            chunks.append(" ".join(" ".join(current).split()))

        paragraph_words = paragraph.split()
        heading_words = heading.split()
        available_size = chunk_size - len(heading_words) if heading_words else chunk_size

        if len(paragraph_words) > available_size:
            split_chunks = _chunk_words(
                paragraph_words,
                chunk_size=max(1, available_size),
                overlap=min(overlap, max(0, available_size - 1)),
                min_chunk_size=min_chunk_size,
            )
            chunks.extend(
                _with_heading(heading, chunk)
                for chunk in split_chunks
            )
            current = [heading] if heading else []
        else:
            current = [heading, paragraph] if heading else [paragraph]

    if current and (not heading or current != [heading]):
        chunks.append(" ".join(" ".join(current).split()))

    return chunks


def _with_heading(heading, chunk):
    if not heading:
        return chunk
    return f"{heading} {chunk}"
