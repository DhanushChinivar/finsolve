# FinSolve AI Assistant

An enterprise RAG chatbot for internal company data. Employees query finance, engineering, marketing, and HR knowledge bases and only get answers they're actually allowed to see.

## What makes it interesting

Most RAG demos let everyone query everything. FinSolve adds two layers on top: **semantic routing** (classify the query's intent before hitting the vector DB) and **RBAC filtering** (intersect the query's intent with the user's role before retrieval). A finance analyst asking about engineering SLAs gets a 403, not a leaked answer.

## Architecture

```
User query
    │
    ▼
Input Guardrails ──► block PII / injection / off-topic
    │
    ▼
Semantic Router (HuggingFace encoder + SemanticRouter)
    │  classifies intent → finance / engineering / marketing / hr_general / general
    ▼
RBAC Intersection
    │  route_collections ∩ role_collections = allowed_collections
    │  empty intersection → 403
    ▼
Qdrant vector search
    │  filter: metadata.collection IN allowed_collections
    │          AND metadata.access_roles CONTAINS user_role
    ▼
Groq LLaMA-3.1-8b-instant (answer generation)
    │
    ▼
Output Guardrails ──► warn if no source docs cited
    │
    ▼
Response + sources
```

## RBAC Matrix

| Role        | Collections accessible              | Demo user |
|-------------|-------------------------------------|-----------|
| employee    | general                             | alice     |
| finance     | finance, general                    | bob       |
| engineering | engineering, general                | carol     |
| marketing   | marketing, general                  | dave      |
| hr          | hr, general                         | frank     |
| c_level     | all                                 | erin      |

## Stack

- **LLM:** Groq (`llama-3.1-8b-instant`)
- **Embeddings:** `all-MiniLM-L6-v2` (SentenceTransformers)
- **Vector DB:** Qdrant (single collection, RBAC via payload metadata)
- **Routing:** `semantic-router` with HuggingFace encoder
- **API:** FastAPI
- **UI:** Next.js 16 + React 19 + Tailwind CSS v4

## Setup

```bash
# 1. Clone
git clone <repo-url> && cd finsolve
```

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Env
cp .env.example .env
# Fill in: GROQ_API_KEY, QDRANT_URL, QDRANT_COLLECTION_NAME
```

```bash
# Qdrant (Docker)
docker run -p 6333:6333 qdrant/qdrant

# Ingest documents
python ingestion/ingest.py

# Start API
uvicorn api.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
# Set: NEXT_PUBLIC_API_BASE=http://localhost:8000

npm run dev
```

Open http://localhost:3000

## Environment Variables

| Variable                  | Default                   | Description                        |
|---------------------------|---------------------------|------------------------------------|
| `GROQ_API_KEY`            | —                         | Groq API key (required)            |
| `QDRANT_URL`              | `http://localhost:6333`   | Qdrant instance URL                |
| `QDRANT_COLLECTION_NAME`  | `finbot`                  | Collection name                    |
| `ALLOWED_ORIGINS`         | `http://localhost:3000`   | Comma-separated frontend origins   |
| `API_PORT`                | `8000`                    | Backend port                       |
| `NEXT_PUBLIC_API_BASE`    | `http://localhost:8000`   | Frontend → backend URL             |

## Guardrails

**Input (pre-retrieval)**
- Rate limit: 20 queries per session per user
- Prompt injection detection (regex patterns)
- Off-topic rejection (non-business keywords)
- PII detection (Aadhaar, phone, email, bank account patterns)

**Output (post-generation)**
- Warns if no source documents were retrieved

## Project layout

```
finsolve/
├── backend/
│   ├── api/
│   │   └── main.py              FastAPI app, login + query endpoints
│   ├── routing/
│   │   └── router.py            Semantic router + RBAC intersection
│   ├── guardrails/
│   │   └── guardrails.py        Input + output guardrail checks
│   ├── ingestion/
│   │   └── ingest.py            Document → chunks → Qdrant
│   ├── evaluation/
│   │   └── evaluation.py        RAGAS evaluation pipeline
│   ├── data/
│   │   ├── finance/             Quarterly reports, budgets, vendor payments
│   │   ├── engineering/         SLA reports, incident logs, sprint metrics
│   │   ├── marketing/           Campaign reports, acquisition data
│   │   ├── hr/                  HR data (CSV)
│   │   └── general/             Employee handbook (PDF)
│   └── requirements.txt
└── frontend/
    └── src/app/
        ├── page.tsx             Login panel + chat UI
        ├── layout.tsx           Root layout
        └── globals.css          Tailwind base styles
```
