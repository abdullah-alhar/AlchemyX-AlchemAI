import csv
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

try:
    from ..config import TESSERACT_CMD
    from ..retrieval.document_metadata import file_metadata
except ImportError:
    from src.alchemyx.config import TESSERACT_CMD
    from src.alchemyx.retrieval.document_metadata import file_metadata


SUPPORTED_EXTENSIONS = ('.txt', '.md', '.docx', '.pdf', '.csv')


@dataclass
class IngestionResult:
    ingested: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    reused: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


def read_txt_md(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def read_docx(file_path):
    import docx

    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)


def read_csv(file_path):
    lines = []

    with open(file_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            for row in reader:
                fields = [
                    f"{field}: {row.get(field, '')}"
                    for field in reader.fieldnames
                    if row.get(field, '') != ''
                ]
                if fields:
                    lines.append("; ".join(fields))
            return "\n".join(lines)

        f.seek(0)
        for row in csv.reader(f):
            if row:
                lines.append("; ".join(cell for cell in row if cell))

    return "\n".join(lines)


def read_pdf(file_path):
    import pytesseract
    from pdf2image import convert_from_path

    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf

    if os.name == "nt" and TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    text = ""

    with pymupdf.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
        
    if len(text.strip()) < 50:
        print(f"[OCR] Scanned file detected...Processing: {file_path}")
        images = convert_from_path(file_path)
        ocr_text = ""
        for img in images:
            ocr_text += pytesseract.image_to_string(img)
        return ocr_text
        
    return text


def extract_text(file_path):
    normalized_path = file_path.lower()

    if normalized_path.endswith(('.txt', '.md')):
        return read_txt_md(file_path)
    if normalized_path.endswith('.docx'):
        return read_docx(file_path)
    if normalized_path.endswith('.csv'):
        return read_csv(file_path)
    if normalized_path.endswith('.pdf'):
        return read_pdf(file_path)

    return ""


def make_doc_id(corpus_folder_path, file_path):
    relative_path = os.path.relpath(file_path, corpus_folder_path)
    return relative_path.replace(os.sep, "/")


def resolve_corpus_path(corpus_folder_path, base_path=None):
    corpus_path = Path(corpus_folder_path).expanduser()
    if corpus_path.is_absolute():
        return corpus_path

    base = Path(base_path) if base_path else Path.cwd()
    return base / corpus_path


def ingest_corpus(
    corpus_folder_path,
    pipeline,
    bm25_store,
    base_path=None,
    document_registry=None,
):
    try:
        from alchemyx.retrieval.main import add_document_to_indexes
    except ImportError:
        from src.alchemyx.retrieval.main import add_document_to_indexes

    corpus_path = resolve_corpus_path(corpus_folder_path, base_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus folder does not exist: {corpus_path}")
    if not corpus_path.is_dir():
        raise NotADirectoryError(f"Corpus path is not a folder: {corpus_path}")

    result = IngestionResult()
    indexed_source_docs = set()

    deferred = getattr(bm25_store, "deferred_writes", None)
    with deferred() if deferred else nullcontext():
        for root, _dirs, files in os.walk(corpus_path):
            for file in files:
                file_path = os.path.join(root, file)
                extracted_text = ""

                try:
                    if not file.lower().endswith(SUPPORTED_EXTENSIONS):
                        result.skipped.append(make_doc_id(corpus_path, file_path))
                        print(f"Skipped unsupported file: {file}")
                        continue

                    extracted_text = extract_text(file_path)

                    if extracted_text:
                        doc_id = make_doc_id(corpus_path, file_path)
                        metadata = file_metadata(corpus_path, file_path, doc_id)
                        already_indexed = (
                            hasattr(pipeline, "has_document")
                            and pipeline.has_document(doc_id, extracted_text)
                        )
                        add_document_to_indexes(
                            pipeline=pipeline,
                            bm25_store=bm25_store,
                            doc_id=doc_id,
                            text=extracted_text,
                            metadata=metadata,
                        )
                        if document_registry is not None:
                            document_registry.replace_document(doc_id, metadata)
                        indexed_source_docs.add(doc_id)
                        if already_indexed:
                            result.reused.append(doc_id)
                            print(f"↻ Reused existing index: {file}.")
                        else:
                            result.ingested.append(doc_id)
                            print(f"✅ Success: {file} ingested.")
                    else:
                        result.skipped.append(make_doc_id(corpus_path, file_path))
                        print(f"Skipped empty file: {file}")

                except Exception as e:
                    result.failed[make_doc_id(corpus_path, file_path)] = str(e)
                    print(f"❌ Error: {file} can't read file. Reason: {e}")

    if not result.failed:
        result.removed = prune_indexes(
            pipeline=pipeline,
            bm25_store=bm25_store,
            document_registry=document_registry,
            source_docs=indexed_source_docs,
        )

    return result


def prune_indexes(pipeline, bm25_store, source_docs, document_registry=None):
    removed = []

    if hasattr(pipeline, "delete_documents_except"):
        removed.extend(pipeline.delete_documents_except(source_docs))
    if hasattr(bm25_store, "delete_documents_except"):
        removed.extend(bm25_store.delete_documents_except(source_docs))
    if document_registry is not None and hasattr(document_registry, "delete_documents_except"):
        removed.extend(document_registry.delete_documents_except(source_docs))

    removed = sorted(set(removed))
    for source_doc in removed:
        print(f"Removed stale index: {source_doc}")

    return removed


def print_ingestion_summary(result):
    print("\nIngestion summary:")
    print(f"Ingested: {len(result.ingested)}")
    print(f"Reused: {len(result.reused)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Failed: {len(result.failed)}")
    print(f"Removed stale: {len(result.removed)}")

    if result.failed:
        print("\nFailed files:")
        for doc_id, reason in result.failed.items():
            print(f"- {doc_id}: {reason}")


if __name__ == "__main__":
    import sys

    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root / "src"))

    try:
        from dotenv import find_dotenv, load_dotenv
        from alchemyx.config import DEFAULT_CORPUS_PATH
        from alchemyx.retrieval.main import create_retrieval_stack
    except ImportError:
        from dotenv import find_dotenv, load_dotenv
        from src.alchemyx.config import DEFAULT_CORPUS_PATH
        from src.alchemyx.retrieval.main import create_retrieval_stack

    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    project_root = Path(dotenv_path).parent if dotenv_path else project_root
    my_corpus_folder = DEFAULT_CORPUS_PATH

    my_corpus_folder = resolve_corpus_path(my_corpus_folder, project_root)
    pipeline, bm25_store, _hybrid_search, _reranker = create_retrieval_stack()
    
    print(f"Corpus starting reading: {my_corpus_folder}")
    result = ingest_corpus(my_corpus_folder, pipeline, bm25_store)
    print_ingestion_summary(result)
    
    print("\nDone! Corpus ingestion completed.")
