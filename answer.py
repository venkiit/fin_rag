"""Milestones 3-6: route -> retrieve + re-rank -> answer with citations -> RAGAS score."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # macOS: FAISS + ONNX both ship OpenMP
os.environ.setdefault("OMP_NUM_THREADS", "1")           # avoid OpenMP deadlocks

import re
import sys

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from evaluate import RagasScorer, log_result
from index import LocalEmbeddings, build_or_load_index
from llm import get_llm
from rerank import Reranker, retrieve_and_rerank
from router import plan_query

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a financial analyst answering questions about a company's annual report.\n"
     "Use ONLY the numbered sources below.\n"
     "Rules:\n"
     "- Cite every factual sentence with its source tag, e.g. 'Revenue grew 6% [S2].'\n"
     "- Quote figures exactly as written, with units and currency.\n"
     "- Reports contain standalone and consolidated statements. Prefer consolidated (group) "
     "figures and say which one you are quoting.\n"
     "- For multi-part questions, answer each part under a short bold label.\n"
     "- If the sources don't contain the answer, say: \"The retrieved sections of the report don't cover this.\"\n"
     "- Never use outside knowledge.\n\n"
     "Sources:\n{context}"),
    ("human", "{question}"),
])

OUT_OF_SCOPE = ("I can only answer questions about the uploaded annual report, such as revenue, "
                "profit, headcount, risks, segments or outlook. I can't help with that question.")


def format_context(docs) -> str:
    """Number the chunks [S1]..[S8] so the LLM can cite them."""
    return "\n\n---\n\n".join(
        f"[S{i}] (page {d.metadata['page']}, {d.metadata['type']})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


def message_text(msg) -> str:
    """Get plain text from the LLM reply (some models return a list of parts)."""
    return msg.text() if callable(getattr(msg, "text", None)) else str(msg.content)


def build_chain(store, llm=None, reranker=None):
    llm = llm or get_llm()
    reranker = reranker or Reranker()

    # Generation sub-chain: build inputs -> prompt -> LLM -> plain text
    generate = (
        RunnableLambda(lambda x: {"context": format_context(x["docs"]), "question": x["question"]})
        | PROMPT
        | llm
        | RunnableLambda(message_text)
    )

    def add_plan(x: dict) -> dict:
        return {**x, "plan": plan_query(llm, x["question"])}

    def add_docs(x: dict) -> dict:
        plan = x["plan"]
        if plan.route == "out_of_scope":
            return {**x, "docs": []}
        if plan.route == "multi_part":
            merged, seen = [], set()
            for sub in plan.sub_questions:                       # retrieve separately for each part
                for doc in retrieve_and_rerank(store, reranker, sub, keep=4):
                    if doc.metadata["chunk_id"] not in seen:     # skip duplicates across parts
                        seen.add(doc.metadata["chunk_id"])
                        merged.append(doc)
            return {**x, "docs": merged[:8]}
        return {**x, "docs": retrieve_and_rerank(store, reranker, x["question"])}

    def add_answer(x: dict) -> dict:
        if x["plan"].route == "out_of_scope":
            return {**x, "answer": OUT_OF_SCOPE}
        return {**x, "answer": generate.invoke(x)}

    # LCEL sequence: {question} -> +plan -> +docs -> +answer (each step in order, main thread)
    return RunnableLambda(add_plan) | RunnableLambda(add_docs) | RunnableLambda(add_answer)


def citations(answer: str, docs) -> list[tuple[str, int]]:
    """Map the [S#] tags the LLM used back to real page numbers. Handles [S1] and [S1, S6]."""
    used = set()
    for bracket in re.findall(r"\[([^\]]+)\]", answer):
        used.update(int(n) for n in re.findall(r"S(\d+)", bracket))
    return [(f"S{i}", d.metadata["page"]) for i, d in enumerate(docs, start=1) if i in used]


if __name__ == "__main__":
    embeddings = LocalEmbeddings()
    store = build_or_load_index(sys.argv[1], embeddings)
    llm = get_llm()
    chain = build_chain(store, llm=llm)
    scorer = RagasScorer(llm, embeddings)

    while True:
        question = input("\nAsk a question (or press Enter to quit): ").strip()
        if not question:
            break
        result = chain.invoke({"question": question})
        cited = citations(result["answer"], result["docs"])

        print(f"\nROUTE: {result['plan'].route}  {result['plan'].sub_questions}")
        print("\nANSWER:\n" + result["answer"])
        print("\nSOURCES:")
        for tag, page in cited:
            print(f"  [{tag}] -> page {page}")

        if result["plan"].route == "out_of_scope" or not result["docs"]:
            continue                                   # nothing retrieved, nothing to score
        print("\nScoring with RAGAS (about 10-20s)...")
        scores = scorer.score(question, result["answer"], [d.page_content for d in result["docs"]])
        print("RAGAS:", scores)
        log_result({
            "question": question,
            "route": result["plan"].route,
            "sub_questions": result["plan"].sub_questions,
            "answer": result["answer"],
            "retrieved_pages": [d.metadata["page"] for d in result["docs"]],
            "cited_pages": [page for _, page in cited],
            "scores": scores,
        })