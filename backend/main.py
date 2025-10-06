from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path
import json
import io
import requests
import pandas as pd
import docx2txt
from PyPDF2 import PdfReader
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ----------------- App Setup -----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Models -----------------
class ChatRequest(BaseModel):
    message: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

# ----------------- Paths -----------------
STORAGE_DIR = Path("./storage")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = STORAGE_DIR / "faiss.index"
TEXTS_PATH = STORAGE_DIR / "texts.json"

# ----------------- Embedding Setup -----------------
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
emb_model = SentenceTransformer(EMB_MODEL_NAME)
dimension = 384  # embedding size for all-MiniLM-L6-v2

texts: List[dict] = []
index = faiss.IndexFlatIP(dimension)

# ----------------- Persistence -----------------
def load_persisted():
    global texts, index
    if TEXTS_PATH.exists():
        texts = json.loads(TEXTS_PATH.read_text(encoding="utf-8"))
    if INDEX_PATH.exists():
        loaded = faiss.read_index(str(INDEX_PATH))
        if loaded.ntotal > 0:
            index = loaded

def persist():
    TEXTS_PATH.write_text(json.dumps(texts, ensure_ascii=False, indent=2), encoding="utf-8")
    faiss.write_index(index, str(INDEX_PATH))

load_persisted()

# ----------------- Utilities -----------------
def split_into_chunks(text: str, max_words: int = 300) -> List[str]:
    """Split text into word-based chunks for FAISS indexing."""
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def embed_texts(text_list: List[str]) -> np.ndarray:
    """Get embeddings for a list of text chunks."""
    vecs = emb_model.encode(text_list, normalize_embeddings=True)
    return np.array(vecs, dtype="float32")

def add_to_index(chunks: List[str], filename: str = "unknown"):
    """Embed and add chunks to FAISS index and memory store."""
    global index, texts
    embeddings = embed_texts(chunks)
    index.add(embeddings)
    start_id = len(texts)
    for i, chunk in enumerate(chunks):
        texts.append({
            "id": start_id + i,
            "text": chunk,
            "meta": {"filename": filename, "chunk": i}
        })
    persist()

def parse_file(upload: UploadFile) -> str:
    """Read and extract text from PDF, DOCX, TXT, or CSV files."""
    name = upload.filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(upload.file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif name.endswith(".docx"):
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
        raise ValueError("Unsupported file type: use .pdf, .docx, .csv, or .txt")

# ----------------- Routes -----------------
@app.get("/")
def root():
    return {"status": "Backend running 🚀", "chunks": len(texts), "indexed": index.ntotal}

@app.post("/chat")
def chat(req: ChatRequest):
    return {"reply": f"Hello, you said: {req.message}"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handles file upload, chunking, embedding, and auto-summary."""
    try:
        contents = await file.read()
        ext = Path(file.filename).suffix.lower()

        # Extract text
        if ext == ".pdf":
            reader = PdfReader(io.BytesIO(contents))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif ext in [".docx", ".txt"]:
            text = contents.decode("utf-8", errors="ignore")
        elif ext == ".csv":
            text = pd.read_csv(io.BytesIO(contents)).to_string()
        else:
            return {"error": "Unsupported file format"}

        # Split + embed + index
        chunks = split_into_chunks(text)
        add_to_index(chunks, filename=file.filename)

        # Auto summary
        summary_prompt = f"Summarize this document in 5 short, clear sentences:\n\n{text[:3000]}"
        try:
            res = requests.post(
                "http://host.docker.internal:11434/api/generate",
                json={"model": "mistral", "prompt": summary_prompt, "stream": False},
                timeout=60
            )
            data = res.json()
            summary = data.get("response", "").strip()
        except Exception as e:
            summary = f"(Summary failed: {e})"

        return {
            "filename": file.filename,
            "chunks_added": len(chunks),
            "summary": summary,
            "preview": text[:500]
        }

    except Exception as e:
        return {"error": str(e)}

@app.post("/search")
def search(req: SearchRequest):
    if index.ntotal == 0 or not texts:
        return {"results": [], "message": "No data indexed yet. Upload a document first."}

    q_vec = embed_texts([req.query])
    scores, idxs = index.search(q_vec, req.top_k)

    results = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
        if idx == -1:
            continue
        item = texts[idx]
        results.append({
            "score": float(score),
            "text": item["text"],
            "filename": item["meta"]["filename"],
            "chunk": item["meta"]["chunk"]
        })

    return {"results": results}

@app.post("/ask")
def ask(req: SearchRequest):
    if index.ntotal == 0 or not texts:
        return {"answer": "No data indexed yet. Upload a document first."}

    q_vec = embed_texts([req.query])
    scores, idxs = index.search(q_vec, req.top_k)
    context_chunks = [texts[idx]["text"] for idx in idxs[0] if idx != -1]
    context = "\n\n".join(context_chunks[:5])

    prompt = f"""
You are a helpful assistant. Use the context below to answer accurately.

Context:
{context}

Question:
{req.query}

Answer:
"""

    try:
        res = requests.post(
            "http://host.docker.internal:11434/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False},
            timeout=120
        )
        data = res.json()
        answer = data.get("response", "").strip()
    except Exception as e:
        answer = f"Error contacting Ollama: {str(e)}"

    return {"answer": answer, "context_used": context[:500]}

@app.post("/reset")
def reset_index():
    global texts, index
    texts = []
    index = faiss.IndexFlatIP(dimension)
    persist()
    return {"message": "Index cleared."}
