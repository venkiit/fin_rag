"""Golden-set evaluation: does re-ranking put the correct page higher than FAISS alone?"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
from pathlib import Path

from index import LocalEmbeddings, build_or_load_index
from rerank import CANDIDATES, Reranker

KS = (1, 3, 5, 8)


def first_hit(pages: list[int], expected: set[int]) -> int | None:
    """1-based rank of the first chunk from a correct page, or None if not found."""
    for rank, page in enumerate(pages, start=1):
        if page in expected:
            return rank
    return None


def scores(ranks: list) -> dict:
    n = len(ranks)
    out = {f"Hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in KS}
    out["MRR"] = sum(1 / r for r in ranks if r) / n      # Mean Reciprocal Rank
    return out


if __name__ == "__main__":
    store = build_or_load_index(sys.argv[1], LocalEmbeddings())
    reranker = Reranker()
    golden = json.loads(Path("golden_set.json").read_text())

    faiss_ranks, rerank_ranks, rows = [], [], []
    for item in golden:
        expected = set(item["expected_pages"])
        docs = store.similarity_search(item["question"], k=CANDIDATES)
        faiss_pages = [d.metadata["page"] for d in docs]
        reranked_pages = [d.metadata["page"] for d, _ in reranker.rerank(item["question"], docs)]

        f, r = first_hit(faiss_pages, expected), first_hit(reranked_pages, expected)
        faiss_ranks.append(f)
        rerank_ranks.append(r)
        rows.append((item["question"], sorted(expected), f, r))
        print(f"{item['id']}: FAISS rank {f}  ->  re-ranked {r}")

    fs, rs = scores(faiss_ranks), scores(rerank_ranks)
    lines = ["## Re-ranking: before vs after", "",
             f"Golden set: {len(golden)} questions. FAISS top-{CANDIDATES} candidates, "
             "re-ranked with jina-reranker-v1-turbo-en.", "",
             "| Metric | FAISS only | FAISS + re-ranker | Change |", "|---|---|---|---|"]
    for k in fs:
        lines.append(f"| {k} | {fs[k]:.2f} | {rs[k]:.2f} | {rs[k] - fs[k]:+.2f} |")
    lines += ["", "| Question | Correct page(s) | FAISS rank | Re-ranked rank |", "|---|---|---|---|"]
    for q, pages, f, r in rows:
        lines.append(f"| {q} | {pages} | {f or f'not in top {CANDIDATES}'} | {r or f'not in top {CANDIDATES}'} |")

        #lines.append(f"| {q} | {pages} | {f or 'not in top 20'} | {r or 'not in top 20'} |")

    Path("eval").mkdir(exist_ok=True)
    Path("eval/rerank_comparison.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
