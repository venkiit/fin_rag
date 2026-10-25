"""Milestone 1: read a PDF and split it into page-tagged chunks."""
import sys
import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1200      # characters per chunk (~300 words)
CHUNK_OVERLAP = 200    # repeat a little text between chunks so sentences aren't cut in half


def load_chunks(pdf_path: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],   # prefer splitting at paragraphs, then lines, then sentences
    )
    chunks: list[Document] = []

    with pymupdf.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            page_no = page_index + 1          # humans count pages from 1

            # 1. Narrative text -> several chunks per page
            text = page.get_text("text")
            for i, piece in enumerate(splitter.split_text(text)):
                if len(piece.strip()) < 50:    # skip page numbers, headers, tiny fragments
                    continue
                chunks.append(Document(
                    page_content=piece,
                    metadata={"chunk_id": f"p{page_no}-t{i}", "page": page_no, "type": "text"},
                ))

            # 2. Tables -> one chunk each, kept whole as markdown
            # A real Table also needs a header row plus at least 2 data rows, otherwise they are skipped as there are a lot of charts in the reports
            for j, table in enumerate(page.find_tables().tables):
                  if table.row_count < 3 or table.col_count < 2:
                    continue
                  md = table.to_markdown().replace("|Col", "| Col")   # keep placeholder headers readable
                  if len(md) > 80:
                    chunks.append(Document(
                        page_content=f"[Table on page {page_no}]\n{md}",
                        metadata={"chunk_id": f"p{page_no}-tbl{j}", "page": page_no, "type": "table"},
                    ))
    return chunks


if __name__ == "__main__":
    path = sys.argv[1]
    chunks = load_chunks(path)
    tables = [c for c in chunks if c.metadata["type"] == "table"]
    print(f"Total chunks: {len(chunks)}  (text: {len(chunks) - len(tables)}, tables: {len(tables)})")
    print(f"Pages covered: {chunks[0].metadata['page']} to {chunks[-1].metadata['page']}")
    print("\n--- Sample text chunk ---")
    print(chunks[len(chunks) // 2].metadata)
    print(chunks[len(chunks) // 2].page_content[:500])
    if tables:
        print("\n--- Sample table chunk ---")
        print(tables[0].metadata)
        print(tables[0].page_content[:600])