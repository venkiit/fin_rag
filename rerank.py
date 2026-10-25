import sys


from fastembed.rerank.cross_encoder import TextCrossEncoder

from index import LocalEmbeddings, build_or_load_index

#RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"   # small cross-encoder, ~80 MB, runs locally
RERANK_MODEL = "jinaai/jina-reranker-v1-turbo-en"
CANDIDATES = 20                                  # how many FAISS results to re-score
KEEP = 8                                          # how many to send to Gemini


class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        #self.model = TextCrossEncoder(model_name=model_name, cache_dir="data/models")
        self.model = TextCrossEncoder(model_name=model_name, cache_dir="data/models", threads=1)

    def rerank(self, question: str, docs: list) -> list[tuple]:
        """Score each (question, chunk) pair together; return (doc, score) pairs, best first."""
        scores = list(self.model.rerank(question, [d.page_content for d in docs]))
        return sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)


def retrieve_and_rerank(store, reranker: Reranker, question: str,
                        candidates: int = CANDIDATES, keep: int = KEEP) -> list:
    docs = store.similarity_search(question, k=candidates)   # stage 1: FAISS, broad
    ranked = reranker.rerank(question, docs)                  # stage 2: cross-encoder, precise
    return [doc for doc, _ in ranked[:keep]]


if __name__ == "__main__":
    # Before/after comparison: how did re-ranking change the order?
    store = build_or_load_index(sys.argv[1], LocalEmbeddings())
    reranker = Reranker()

    while True:
        question = input("\nAsk a question (or press Enter to quit): ").strip()
        if not question:
            break
        docs = store.similarity_search(question, k=CANDIDATES)
        faiss_rank = {d.metadata["chunk_id"]: i for i, d in enumerate(docs, start=1)}
        ranked = reranker.rerank(question, docs)

        print(f"\n{'NEW':>4} {'OLD':>4} {'MOVE':>5}  {'PAGE':>4}  {'SCORE':>6}  PREVIEW")
        for new, (doc, score) in enumerate(ranked[:8], start=1):
            old = faiss_rank[doc.metadata["chunk_id"]]
            move = f"+{old - new}" if old > new else (str(old - new) if old < new else "=")
            preview = doc.page_content[:70].replace("\n", " ")
            print(f"{new:>4} {old:>4} {move:>5}  {doc.metadata['page']:>4}  {score:>6.2f}  {preview}")
