"""Milestone 3+4: answer questions with Gemini, grounded in re-ranked chunks, with page citations."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # macOS: FAISS + ONNX both ship OpenMP
os.environ.setdefault("OMP_NUM_THREADS", "1")           # avoid OpenMP deadlocks

import re
import sys

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from index import LocalEmbeddings, build_or_load_index
from rerank import Reranker, retrieve_and_rerank

load_dotenv()                                    # loads GOOGLE_API_KEY from .env
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a financial analyst answering questions about a company's annual report.\n"
     "Use ONLY the numbered sources below.\n"
     "Rules:\n"
     "- Cite every factual sentence with its source tag, e.g. 'Revenue grew 6% [S2].'\n"
     "- Quote figures exactly as written, with units and currency.\n"
     "- Reports contain standalone and consolidated statements. Prefer consolidated (group) "
     "figures and say which one you are quoting.\n"
     "- If the sources don't contain the answer, say: \"The retrieved sections of the report don't cover this.\"\n"
     "- Never use outside knowledge.\n\n"
     "Sources:\n{context}"),
    ("human", "{question}"),
])


def format_context(docs) -> str:
    """Number the chunks [S1]..[S8] so Gemini can cite them."""
    return "\n\n---\n\n".join(
        f"[S{i}] (page {d.metadata['page']}, {d.metadata['type']})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


def message_text(msg) -> str:
    """Get plain text from Gemini's reply (newer models can return a list of parts)."""
    return msg.text() if callable(getattr(msg, "text", None)) else str(msg.content)


def build_chain(store):
    llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)
    reranker = Reranker()

    # Generation sub-chain: build inputs -> prompt -> Gemini -> plain text
    generate = (
        RunnableLambda(lambda x: {"context": format_context(x["docs"]), "question": x["question"]})
        | PROMPT
        | llm
        | RunnableLambda(message_text)
    )

    def add_docs(x: dict) -> dict:
        # FAISS top-20 -> cross-encoder -> best 8 (runs in the main thread)
        return {**x, "docs": retrieve_and_rerank(store, reranker, x["question"])}

    def add_answer(x: dict) -> dict:
        return {**x, "answer": generate.invoke(x)}

    # LCEL sequence: {question} -> +docs -> +answer, each step in order, same thread
    return RunnableLambda(add_docs) | RunnableLambda(add_answer)


def citations(answer: str, docs) -> list[tuple[str, int]]:
    """Map the [S#] tags Gemini used back to real page numbers from metadata.
    Handles both [S1] and grouped forms like [S1, S6]."""
    used = set()
    for bracket in re.findall(r"\[([^\]]+)\]", answer):      # text inside every [...]
        used.update(int(n) for n in re.findall(r"S(\d+)", bracket))
    return [(f"S{i}", d.metadata["page"]) for i, d in enumerate(docs, start=1) if i in used]


if __name__ == "__main__":
    store = build_or_load_index(sys.argv[1], LocalEmbeddings())
    chain = build_chain(store)

    while True:
        question = input("\nAsk a question (or press Enter to quit): ").strip()
        if not question:
            break
        result = chain.invoke({"question": question})
        print("\nANSWER:\n" + result["answer"])
        print("\nSOURCES:")
        for tag, page in citations(result["answer"], result["docs"]):
            print(f"  [{tag}] -> page {page}")