"""Milestone 7: FastAPI app - upload a report, ask questions, see citations and RAGAS scores.

Run:  uvicorn api:app --port 8000      then open http://localhost:8000
API docs are generated automatically at http://localhost:8000/docs
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # macOS: FAISS + ONNX both ship OpenMP
os.environ.setdefault("OMP_NUM_THREADS", "1")

import asyncio
import json
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pymupdf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel

from answer import build_chain, citations
from evaluate import RagasScorer, log_result, read_log
from index import INDEX_DIR, LocalEmbeddings
from ingest import make_splitter, page_chunks
from llm import get_llm
from rerank import Reranker

UPLOAD_DIR = Path("data/uploads")
INDEXES_DIR = Path("data/indexes")
DEFAULT_ID = "hcl_2026"
DEFAULT_PDF = Path("reports/hcl_2026.pdf")

models: dict = {}       # embeddings, llm, reranker, scorer - loaded once at startup
docs: dict = {}         # doc_id -> {name, status, stage, done, total, pdf, chain, ...}
answers: dict = {}      # query_id -> what RAGAS needs to score that answer later


def pdf_url(doc_id: str, pdf_path: Path | None) -> str | None:
    if pdf_path and pdf_path.exists():
        return f"/documents/{doc_id}/pdf"
    return os.getenv("DEFAULT_PDF_URL") if doc_id == DEFAULT_ID else None


def register(doc_id: str, name: str, store, pdf_path: Path | None) -> None:
    docs[doc_id] = {
        "name": name, "status": "ready", "stage": "ready", "done": 0, "total": 0,
        "pdf": pdf_path, "chain": build_chain(store, llm=models["llm"], reranker=models["reranker"]),
    }


def public(doc_id: str) -> dict:
    d = docs[doc_id]
    return {"doc_id": doc_id, "name": d["name"], "status": d["status"], "stage": d["stage"],
            "done": d["done"], "total": d["total"], "error": d.get("error"),
            "pdf_url": pdf_url(doc_id, d.get("pdf"))}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    models["embeddings"] = LocalEmbeddings()
    models["llm"] = get_llm()
    models["reranker"] = Reranker()
    models["scorer"] = RagasScorer(models["llm"], models["embeddings"])

    load = lambda path: FAISS.load_local(str(path), models["embeddings"], allow_dangerous_deserialization=True)
    if INDEX_DIR.exists():                                   # the pre-built HCLTech index
        register(DEFAULT_ID, "HCLTech Annual Report 2025-26", load(INDEX_DIR), DEFAULT_PDF)
    for meta_file in INDEXES_DIR.glob("*/meta.json"):        # reports uploaded earlier
        meta = json.loads(meta_file.read_text())
        register(meta["doc_id"], meta["name"], load(meta_file.parent), Path(meta["pdf"]))
    yield


app = FastAPI(title="Filings Analyst API", version="1.0", lifespan=lifespan)


# ---------- indexing an uploaded PDF, with progress ----------
async def index_pdf(doc_id: str, pdf_path: Path) -> None:
    """Runs on the server's main event loop, yielding after every page and batch so
    progress requests are answered while it works (no background threads needed)."""
    d = docs[doc_id]
    try:
        splitter, chunks = make_splitter(), []
        with pymupdf.open(pdf_path) as pdf:
            d.update(stage="reading", done=0, total=len(pdf))
            for i, page in enumerate(pdf):
                chunks.extend(page_chunks(page, i + 1, splitter))
                d["done"] = i + 1
                await asyncio.sleep(0)                       # let status requests through
        if not chunks:
            raise ValueError("No extractable text found. Is this a scanned PDF?")

        d.update(stage="embedding", done=0, total=len(chunks))
        vectors = []
        for start in range(0, len(chunks), 32):
            batch = chunks[start:start + 32]
            vectors.extend(models["embeddings"].embed_documents([c.page_content for c in batch]))
            d["done"] = start + len(batch)
            await asyncio.sleep(0)

        store = FAISS.from_embeddings(
            list(zip([c.page_content for c in chunks], vectors)),
            models["embeddings"], metadatas=[c.metadata for c in chunks])
        out = INDEXES_DIR / doc_id
        store.save_local(str(out))
        (out / "meta.json").write_text(json.dumps({"doc_id": doc_id, "name": d["name"], "pdf": str(pdf_path)}))
        register(doc_id, d["name"], store, pdf_path)
    except Exception as e:
        d.update(status="error", stage="error", error=str(e))


# ---------- routes ----------
@app.get("/", include_in_schema=False)
async def home():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "documents": len(docs), "llm_model": os.getenv("LLM_MODEL")}


@app.get("/documents")
async def list_documents():
    return [public(doc_id) for doc_id in docs]


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")
    doc_id = uuid.uuid4().hex[:8]
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = UPLOAD_DIR / f"{doc_id}.pdf"
    with pdf_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    docs[doc_id] = {"name": file.filename, "status": "processing", "stage": "queued",
                    "done": 0, "total": 0, "pdf": pdf_path}
    asyncio.create_task(index_pdf(doc_id, pdf_path))         # indexing continues after we reply
    return public(doc_id)


@app.get("/documents/{doc_id}/status")
async def document_status(doc_id: str):
    if doc_id not in docs:
        raise HTTPException(404, "Unknown document")
    return public(doc_id)


@app.get("/documents/{doc_id}/pdf", include_in_schema=False)
async def document_pdf(doc_id: str):
    d = docs.get(doc_id)
    if not d or not d.get("pdf") or not Path(d["pdf"]).exists():
        raise HTTPException(404, "PDF not available")
    return FileResponse(d["pdf"], media_type="application/pdf", headers={"Content-Disposition": "inline"})


class QueryRequest(BaseModel):
    doc_id: str = DEFAULT_ID
    question: str


@app.post("/query")
async def query(req: QueryRequest):
    d = docs.get(req.doc_id)
    if not d:
        raise HTTPException(404, "Unknown document")
    if d["status"] != "ready":
        raise HTTPException(409, "This document is still being indexed.")

    result = d["chain"].invoke({"question": req.question})   # route -> retrieve -> re-rank -> answer
    cited = citations(result["answer"], result["docs"])
    query_id = uuid.uuid4().hex[:10]
    answers[query_id] = {"doc_id": req.doc_id, "question": req.question, "result": result, "cited": cited}

    by_tag = {f"S{i}": doc for i, doc in enumerate(result["docs"], start=1)}
    return {
        "query_id": query_id,
        "route": result["plan"].route,
        "sub_questions": result["plan"].sub_questions,
        "answer": result["answer"],
        "citations": [{"tag": tag, "page": page, "type": by_tag[tag].metadata["type"],
                       "snippet": by_tag[tag].page_content[:300]} for tag, page in cited],
        "retrieved": [{"rank": i, "page": doc.metadata["page"], "chunk_id": doc.metadata["chunk_id"]}
                      for i, doc in enumerate(result["docs"], start=1)],
        "pdf_url": pdf_url(req.doc_id, d.get("pdf")),
        "can_evaluate": bool(result["docs"]),
    }


@app.post("/evaluate/{query_id}")
async def evaluate_answer(query_id: str):
    item = answers.get(query_id)
    if not item:
        raise HTTPException(404, "Unknown query")
    result = item["result"]
    if not result["docs"]:
        return {"skipped": "out of scope - nothing retrieved to score"}
    scores = await models["scorer"].ascore(
        item["question"], result["answer"], [doc.page_content for doc in result["docs"]])
    log_result({
        "doc_id": item["doc_id"], "question": item["question"], "route": result["plan"].route,
        "sub_questions": result["plan"].sub_questions, "answer": result["answer"],
        "retrieved_pages": [doc.metadata["page"] for doc in result["docs"]],
        "cited_pages": [page for _, page in item["cited"]], "scores": scores,
    })
    return scores


@app.get("/evaluations")
async def evaluations(limit: int = 50):
    return read_log(limit)