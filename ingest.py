"""Milestone 1: read a PDF and split it into page-tagged chunks."""
import re
import sys

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1200      # characters per chunk (~300 words)
CHUNK_OVERLAP = 200    # repeat a little text between chunks so sentences aren't cut in half


def make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],   # prefer paragraphs, then lines, then sentences
    )


def page_chunks(page, page_no: int, splitter) -> list[Document]:
    """All chunks for ONE page: narrative text pieces + whole tables as markdown."""
    chunks: list[Document] = []

    # 1. Narrative text -> several chunks per page
    for i, piece in enumerate(splitter.split_text(page.get_text("text"))):
        if len(piece.strip()) < 50:              # skip page numbers, headers, tiny fragments
            continue
        chunks.append(Document(
            page_content=piece,
            metadata={"chunk_id": f"p{page_no}-t{i}", "page": page_no, "type": "text"},
        ))

    # 2. Tables -> one chunk each, kept whole as markdown
    for j, table in enumerate(page.find_tables().tables):
        if table.row_count < 3 or table.col_count < 2:   # skip charts / false detections
            continue
        md = re.sub(r"\|Col\d+", "| ", table.to_markdown())
        if len(md) > 80:
            chunks.append(Document(
                page_content=f"[Table on page {page_no}]\n{md}",
                metadata={"chunk_id": f"p{page_no}-tbl{j}", "page": page_no, "type": "table"},
            ))
    return chunks


def load_chunks(pdf_path: str) -> list[Document]:
    splitter = make_splitter()
    chunks: list[Document] = []
    with pymupdf.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            page_no = page_index + 1             # humans count pages from 1
            if page_no % 25 == 0:
                print(f"  ...processed {page_no} of {len(pdf)} pages", flush=True)
            chunks.extend(page_chunks(page, page_no, splitter))
    return chunks


if __name__ == "__main__":
    chunks = load_chunks(sys.argv[1])
    tables = [c for c in chunks if c.metadata["type"] == "table"]
    print(f"Total chunks: {len(chunks)}  (text: {len(chunks) - len(tables)}, tables: {len(tables)})")
    print(f"Pages covered: {chunks[0].metadata['page']} to {chunks[-1].metadata['page']}")