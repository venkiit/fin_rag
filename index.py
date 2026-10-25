"""Milestone 2: embed chunks locally (FastEmbed) and search them with FAISS."""
import sys
import time
from pathlib import Path

from fastembed import TextEmbedding
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from ingest import load_chunks

EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # small, strong English embedding model (~70 MB)
INDEX_DIR = Path("data/index")           # where the FAISS index is saved


class LocalEmbeddings(Embeddings):
    """Runs the embedding model on your machine. LangChain calls these two methods."""

    def __init__(self, model_name: str = EMBED_MODEL):
        self.model = TextEmbedding(model_name=model_name, cache_dir="data/models")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Used once, when building the index: one vector per chunk
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # Used for every question. query_embed adds the special prefix bge models expect
        return next(iter(self.model.query_embed(text))).tolist()


def build_or_load_index(pdf_path: str, embeddings: Embeddings) -> FAISS:
    if INDEX_DIR.exists():
        print("Loading saved index from disk...")
        return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)

    print("Building index (one-time, takes a few minutes)...")
    start = time.time()
    chunks = load_chunks(pdf_path)
    print(f"  {len(chunks)} chunks extracted. Embedding them...")
    store = FAISS.from_documents(chunks, embeddings)   # embeds every chunk and stores the vectors
    store.save_local(str(INDEX_DIR))
    print(f"  Index built and saved in {time.time() - start:.0f}s")
    return store


if __name__ == "__main__":
    embeddings = LocalEmbeddings()
    store = build_or_load_index(sys.argv[1], embeddings)

    while True:
        question = input("\nAsk a question (or press Enter to quit): ").strip()
        if not question:
            break
        results = store.similarity_search_with_score(question, k=5)
        for rank, (doc, distance) in enumerate(results, start=1):
            meta = doc.metadata
            preview = doc.page_content[:200].replace("\n", " ")
            print(f"\n#{rank}  page {meta['page']}  [{meta['type']}]  distance={distance:.3f}")
            print(f"    {preview}...")