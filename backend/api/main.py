import os
import logging
from typing import Optional, List, Dict, Tuple
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from sentence_transformers import SentenceTransformer

# Import custom modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from routing.router import get_collections_for_query
    from guardrails.guardrails import check_input_guardrails, check_output_guardrails
except ModuleNotFoundError:
    from backend.routing.router import get_collections_for_query
    from backend.guardrails.guardrails import (
        check_input_guardrails,
        check_output_guardrails,
    )

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── FastAPI App Setup ──────────────────────────
app = FastAPI(
    title="FinSolve AI Assistant API",
    description="Query financial, HR, and engineering data with AI",
    version="1.0.0"
)

# CORS Configuration
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize Services ──────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
)

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333")
)
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "finbot")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Demo users for /login
DEMO_USERS: Dict[str, Dict[str, str]] = {
    "alice": {"password": "alice123", "role": "employee",   "user_id": "usr_emp_001"},
    "bob":   {"password": "bob123",   "role": "finance",    "user_id": "usr_fin_001"},
    "carol": {"password": "carol123", "role": "engineering","user_id": "usr_eng_001"},
    "dave":  {"password": "dave123",  "role": "marketing",  "user_id": "usr_mkt_001"},
    "erin":  {"password": "erin123",  "role": "c_level",    "user_id": "usr_cx_001"},
    "frank": {"password": "frank123", "role": "hr",         "user_id": "usr_hr_001"},
}

# ── Pydantic Models ──────────────────────────
class QueryRequest(BaseModel):
    query: str
    user_role: str
    top_k: int = 5  # Number of context chunks to retrieve

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    role: str

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    user_id: str
    role: str

class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    llm_available: bool

# ── Helper Functions ──────────────────────────
def build_rbac_filter(allowed_collections: List[str], user_role: str) -> Filter:
    """Build Qdrant filter based on RBAC metadata"""
    collection_condition: FieldCondition
    if len(allowed_collections) == 1:
        collection_condition = FieldCondition(
            key="metadata.collection",
            match=MatchValue(value=allowed_collections[0])
        )
    else:
        collection_condition = FieldCondition(
            key="metadata.collection",
            match=MatchAny(any=allowed_collections)
        )

    role_condition = FieldCondition(
        key="metadata.access_roles",
        match=MatchAny(any=[user_role])
    )

    return Filter(must=[collection_condition, role_condition])

def retrieve_context(
    allowed_collections: List[str],
    query: str,
    user_role: str,
    top_k: int = 5
) -> Tuple[List[Dict[str, str]], List[str]]:
    """Retrieve context from Qdrant using RBAC metadata filter"""
    try:
        query_vector = embedder.encode(query).tolist()

        query_filter = build_rbac_filter(allowed_collections, user_role)

        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter
        )

        chunks: List[Dict[str, str]] = []
        sources: List[str] = []

        for result in results or []:
            payload = result.payload or {}
            content = payload.get("content", "")
            metadata = payload.get("metadata", {})
            source_doc = metadata.get("source_document", "unknown")
            if content:
                chunks.append({"content": content, "source": source_doc})
            if source_doc and source_doc not in sources:
                sources.append(source_doc)

        return chunks, sources

    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        return [], []

def generate_answer(query: str, context: List[Dict[str, str]]) -> str:
    """Generate answer using LLM with retrieved context"""
    try:
        context_text = "\n\n".join(
            [f"[Source: {item['source']}]\n{item['content']}" for item in context]
        )

        prompt = f"""Based on the following context, answer the question:

Context:
{context_text}

Question: {query}

Answer: Be concise and use information from the context. If the context doesn't contain relevant information, say so."""

        response = llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return "Sorry, I couldn't generate an answer. Please try again."

# ── API Endpoints ──────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    qdrant_ok = False
    llm_ok = False
    
    try:
        qdrant_client.get_collections()
        qdrant_ok = True
    except Exception as e:
        logger.error(f"Qdrant check failed: {str(e)}")
    
    try:
        llm.invoke("ping")
        llm_ok = True
    except Exception as e:
        logger.error(f"LLM check failed: {str(e)}")
    
    return HealthResponse(
        status="ok" if (qdrant_ok and llm_ok) else "degraded",
        qdrant_connected=qdrant_ok,
        llm_available=llm_ok
    )

@app.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Demo login endpoint (static users)"""
    user = DEMO_USERS.get(request.username)
    if not user or user["password"] != request.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return LoginResponse(user_id=user["user_id"], role=user["role"])

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, x_user_id: Optional[str] = Header(None)):
    """Query the FinSolve assistant"""

    user_id = x_user_id or request.user_role

    # ── Step 1: Input Guardrails ──────────────────────────
    is_safe, guardrail_message = check_input_guardrails(request.query, user_id)
    if not is_safe:
        raise HTTPException(
            status_code=400,
            detail=f"Query blocked: {guardrail_message}"
        )

    # ── Step 2: Semantic Routing + Role Intersection ──────────────────────────
    allowed_collections = get_collections_for_query(request.query, request.user_role)
    if not allowed_collections:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: Your role '{request.user_role}' has no access to this query"
        )

    # ── Step 3: Retrieve Context with RBAC Filter ──────────────────────────
    context_chunks, sources = retrieve_context(
        allowed_collections, request.query, request.user_role, request.top_k
    )

    if not context_chunks:
        logger.warning("No context found for query with RBAC filter")
        context_chunks = [{"content": "No specific information found in knowledge base", "source": "none"}]

    # ── Step 4: Generate Answer ──────────────────────────
    answer = generate_answer(request.query, context_chunks)

    # ── Step 5: Output Guardrails ──────────────────────────
    answer, has_warning = check_output_guardrails(answer, sources)
    if has_warning:
        logger.warning("Output guardrails warning: response lacks sources")

    return QueryResponse(
        answer=answer,
        sources=sources,
        role=request.user_role
    )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "FinSolve AI Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "login": "/login (POST)",
            "query": "/query (POST)",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", 8000)),
        log_level="info"
    )
