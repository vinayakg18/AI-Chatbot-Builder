from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple
from pathlib import Path
import json
import io
import requests


# parsing
import pandas as pd
import docx2txt
from PyPDF2 import PdfReader

# embeddings + faiss
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ----------------- App & CORS -----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten later to http://localhost:3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Models & Storage -----------------
class ChatRequest(BaseModel):
    message: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

STORAGE_DIR = Path("./storage")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = STORAGE_DIR / "faiss.index"
TEXTS_PATH = STORAGE_DIR / "texts.json"

# sentence-transformers model (small, fast, good)
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
emb_model = SentenceTransformer(EMB_MODEL_NAME)

# in-memory text store mirrors the JSON file
# each item: {"id": int, "text": str, "meta": {"filename": str, "chunk": int}}
texts: List[dict] = []

# FAISS index (cosine similarity via inner product on normalized vectors)
dimension = 384  # all-MiniLM-L6-v2 output size
index = faiss.IndexFlatIP(dimension)

def load_persisted():
    global texts, index
    if TEXTS_PATH.exists():
        texts = json.loads(TEXTS_PATH.read_text(encoding="utf-8"))
    if INDEX_PATH.exists():
        # must match the same index type used above (IndexFlatIP)
        loaded = faiss.read_index(str(INDEX_PATH))
        # If empty file was written, keep fresh index
        if loaded.ntotal > 0:
            # Replace the fresh index with the loaded one
            # Make sure metric type is inner product for cosine
            # If you previously saved a different type, delete the index file.
            global index
            index = loaded

def persist():
    # save texts
    TEXTS_PATH.write_text(json.dumps(texts, ensure_ascii=False, indent=2), encoding="utf-8")
    # save faiss index
    faiss.write_index(index, str(INDEX_PATH))

load_persisted()

# ----------------- Utilities -----------------
def chunk_text(s: str, max_chars: int = 800, overlap: int = 120) -> List[str]:
    """Simple char-based chunker with overlap. Good enough for MVP."""
    s = s.strip().replace("\r", "")
    chunks = []
    start = 0
    n = len(s)
    while start < n:
        end = min(start + max_chars, n)
        chunk = s[start:end]
        # try to end on a sentence boundary
        if end < n:
            last_dot = chunk.rfind(".")
            if last_dot > 200:  # don't cut too early
                end = start + last_dot + 1
                chunk = s[start:end]
        chunks.append(chunk.strip())
        start = max(end - overlap, end)  # ensure progress
    # filter tiny pieces
    return [c for c in chunks if len(c) > 20]

def embed_texts(text_list: List[str]) -> np.ndarray:
    vecs = emb_model.encode(text_list, normalize_embeddings=True)  # normalize for cosine
    return np.array(vecs, dtype="float32")

def parse_file(upload: UploadFile) -> str:
    name = upload.filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(upload.file)
        out = []
        for p in reader.pages:
            t = p.extract_text() or ""
            out.append(t)
        return "\n".join(out)
    elif name.endswith(".docx"):
        # docx2txt needs a path or a file-like object; wrap bytes
        data = upload.file.read()
        upload.file.seek(0)
        bio = io.BytesIO(data)
        return docx2txt.process(bio)
    elif name.endswith(".csv"):
        df = pd.read_csv(upload.file)
        return df.to_string()
    elif name.endswith(".txt"):
        return upload.file.read().decode("utf-8", errors="ignore")
    else:
        raise ValueError("Unsupported file type. Use .pdf, .docx, .csv, or .txt")

# ----------------- Routes -----------------
@app.get("/")
def root():
    return {"status": "Backend is running 🚀", "chunks": len(texts), "indexed": index.ntotal}

@app.post("/chat")
def chat(req: ChatRequest):
    # Placeholder echo; real QA uses /search + LLM in the next step
    return {"reply": f"Hello, you said: {req.message}"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        raw = parse_file(file)
    except Exception as e:
        return {"error": str(e)}

    # chunk and embed
    chunks = chunk_text(raw)
    if not chunks:
        return {"filename": file.filename, "message": "No extractable text."}

    vecs = embed_texts(chunks)

    # add to FAISS and text store
    start_id = len(texts)
    for i, (c, v) in enumerate(zip(chunks, vecs)):
        texts.append({"id": start_id + i, "text": c, "meta": {"filename": file.filename, "chunk": i}})
    index.add(vecs)

    # persist to disk
    persist()

    return {
        "filename": file.filename,
        "num_chunks": len(chunks),
        "indexed_total": int(index.ntotal),
        "preview": chunks[0][:400] if chunks else ""
    }

@app.post("/search")
def search(req: SearchRequest):
    if index.ntotal == 0 or len(texts) == 0:
        return {"results": [], "message": "No data indexed yet. Upload a document first."}

    q_vec = embed_texts([req.query])  # shape (1, dim)
    scores, idxs = index.search(q_vec, req.top_k)
    results = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
        if idx == -1:
            continue
        item = texts[idx]
        results.append({
            "score": float(score),           # cosine similarity in [ -1, 1 ]
            "text": item["text"],
            "filename": item["meta"]["filename"],
            "chunk": item["meta"]["chunk"]
        })
    return {"results": results}

@app.post("/reset")
def reset_index():
    global texts, index
    texts = []
    index = faiss.IndexFlatIP(dimension)
    persist()  # writes empty structures
    return {"message": "Index cleared."}

@app.post("/ask")
def ask(req: SearchRequest):
    """
    Query the FAISS index, then use Ollama (Mistral) to generate a natural answer.
    """
    if index.ntotal == 0 or len(texts) == 0:
        return {"answer": "No data indexed yet. Upload a document first."}

    # Step 1. Retrieve top chunks
    q_vec = embed_texts([req.query])
    scores, idxs = index.search(q_vec, req.top_k)
    context_chunks = []
    for idx in idxs[0]:
        if idx == -1:
            continue
        context_chunks.append(texts[idx]["text"])
    context = "\n\n".join(context_chunks[:5])

    # Step 2. Build prompt for the model
    prompt = f"""
You are an AI assistant answering questions based only on the following context.
Keep answers short, factual, and quote exact numbers or terms when possible.

Context:
{context}

Question:
{req.query}

Answer:
"""

    # Step 3. Call Ollama locally
    try:
        response = requests.post(
            "http://host.docker.internal:11434/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False},
            timeout=180,
        )
        data = response.json()
        answer_text = data.get("response", "").strip()
    except Exception as e:
        answer_text = f"Ollama error: {str(e)}"

    return {"answer": answer_text, "context_used": context[:500]}
