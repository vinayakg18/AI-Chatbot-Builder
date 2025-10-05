# 🤖 AI Chatbot Builder

**AI Chatbot Builder** is a full-stack, locally runnable chatbot that can understand and answer questions about uploaded PDF documents.  
It’s powered by **FastAPI**, **FAISS**, and **Ollama’s Mistral model**, with a modern **Next.js frontend** for an interactive chat experience.

This project was built to explore how local LLMs and retrieval-based architectures can create document-aware AI assistants without relying on external APIs or cloud costs.

---

## 🚀 Key Features

- 📄 Chat with your documents — upload any PDF and ask natural questions about its contents  
- 🧠 Local AI inference using **Mistral** through **Ollama**, keeping everything on your machine  
- ⚡ Fast retrieval powered by **FAISS** for accurate context search  
- 🌐 Clean dark-themed **Next.js + TailwindCSS** frontend  
- 🐳 Fully containerized with **Docker Compose**  
- 🔒 100% privacy — no external API calls, no data leaves your system  

---

## 🧩 Architecture Overview

```
PDF → Text Extraction → FAISS Indexing → Query Embedding
        ↓                         ↑
     User Query ─────► FastAPI Backend ─────► Ollama (Mistral)
                                       ↓
                                AI Response
                                       ↓
                                Next.js Frontend
```

**Tech Stack**
- **Frontend:** Next.js 15, React, TailwindCSS  
- **Backend:** FastAPI, FAISS, Sentence Transformers, PyPDF2  
- **AI Model:** Mistral via Ollama  
- **Containerization:** Docker + Docker Compose  

---

## 🧠 How It Works

1. Upload a PDF or document.  
2. The backend extracts text, splits it into chunks, and creates embeddings using Sentence Transformers.  
3. Chunks are stored in a FAISS index for fast semantic retrieval.  
4. When you ask a question, the backend finds the most relevant chunks and builds a context prompt.  
5. The context and question are passed to **Mistral** through **Ollama**, which generates a natural-language answer.  

---

## ⚙️ Local Setup

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed  
- [Ollama](https://ollama.ai) installed with Mistral model pulled  

```bash
ollama pull mistral
ollama serve
```

---

### 1️⃣ Clone this repository
```bash
git clone https://github.com/vinayakg18/AI-Chatbot-Builder.git
cd AI-Chatbot-Builder
```

### 2️⃣ Start backend + frontend
```bash
docker-compose up --build
```

- Backend → http://localhost:8000  
- Frontend → http://localhost:3000  

### 3️⃣ Verify Ollama
```bash
curl http://localhost:11434/api/tags
```
Expected output lists `mistral:latest`

---

## 🧪 Usage Demo

1. Go to http://localhost:3000  
2. Upload any PDF (contract, article, etc.)  
3. Ask questions like  
   - “What is the salary mentioned?”  
   - “Who is the employer?”  
   - “What is the employee’s address?”  
4. The chatbot will read from your uploaded file and answer contextually.

---

## 🐳 Docker Architecture

```yaml
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend/storage:/app/storage
    environment:
      - PORT=8000
    extra_hosts:
      - "host.docker.internal:host-gateway"

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

This ensures the backend connects to **Ollama on your host machine**, and all vector data persists under `backend/storage`.

---

## 🧰 Folder Structure

```
AI-Chatbot-Builder/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── storage/
│
├── frontend/
│   ├── src/app/
│   ├── package.json
│   ├── Dockerfile
│
├── docker-compose.yml
├── LICENSE
└── README.md
```

---

## 📦 Dependencies

**Backend**
- fastapi  
- uvicorn  
- faiss-cpu  
- sentence-transformers  
- PyPDF2  
- docx2txt  
- pandas  
- numpy  
- requests  

**Frontend**
- next  
- react  
- tailwindcss  

---

## 🧠 Example Outputs: Based on an offer letter

| Query | Response |
|-------|-----------|
| “What is my salary?” | “Your salary is $80.00 USD per hour, and $70.00 USD during training.” |
| “Who is the employer?” | “XYZ, located at ### St, San Francisco, CA 94107.” |
| “Who is the employee?” | “Vinayakraddi Giriyammanavar.” |

---

## 🛠️ Future Enhancements

- Support for multiple PDFs  
- Optional Hugging Face / Groq API fallback  
- User authentication  
- Persistent conversation memory  
- Deployment via Render, Fly.io, Railway, or VPS  

---

## 👨‍💻 Author

**Vinayakraddi Giriyammanavar**  
Graduate Student | Software Engineer | AI Enthusiast  

- GitHub → [vinayakg18](https://github.com/vinayakg18)  
- LinkedIn → [Vinayakraddi Giriyammanavar](https://linkedin.com/in/vinayakraddi-giriyammanavar)

---

## 🧾 License

This project is released under the **MIT License**.  
Feel free to fork, modify, and experiment — attribution is appreciated!

---

> _Built for learning and exploration. This project shows how modern LLMs like Mistral can run locally to enable private, document-aware AI assistants._
