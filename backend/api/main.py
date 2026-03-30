import os
import logging
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Import custom modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from routing.router import SemanticRouter, Route, HuggingFaceEncoder
from guardrails.guardrails import check_input_guardrails, check_output_guardrails

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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

# Initialize semantic router
encoder = HuggingFaceEncoder()
routes = [
    Route(
        name="finance",
        utterances=[
            "What are the quarterly financial results?",
            "Show me the revenue and expenses",
            "What's our budget allocation?",
            "Tell me about investor relations",
            "What are Q3 earnings?",
            "Show financial metrics",
            "What's the annual financial report?",
        ]
    ),
    Route(
        name="engineering",
        utterances=[
            "What's our system architecture?",
            "Tell me about the API design",
            "How do we handle incidents?",
            "What's the SLA for our system?",
            "Explain the microservices setup",
            "What's our deployment process?",
            "Tell me about infrastructure setup",
        ]
    ),
    Route(
        name="marketing",
        utterances=[
            "What are our campaign performance metrics?",
            "Tell me about brand guidelines",
            "How much budget for marketing?",
            "What's our market share?",
            "Tell me about our campaigns",
        ]
    ),
    Route(
        name="hr",
        utterances=[
            "What's the leave policy?",
            "Tell me about work hours",
            "What are HR benefits?",
            "What's the dress code?",
            "Tell me about maternity leave",
        ]
    ),
    Route(
        name="general",
        utterances=[
            "Tell me about the company",
            "What's the office location?",
            "General company information",
        ]
    ),
]

router = SemanticRouter(
    routes=routes,
    encoder=encoder
)

# Role-based access control
COLLECTION_ACCESS = {
    "general":     ["employee", "finance", "engineering", "marketing", "c_level"],
    "finance":     ["finance", "c_level"],
    "engineering": ["engineering", "c_level"],
    "marketing":   ["marketing", "c_level"],
    "hr":          ["c_level"],
}

# ── Pydantic Models ──────────────────────────
class QueryRequest(BaseModel):
    query: str
    user_id: str
    role: str = "employee"  # Default role
    top_k: int = 5  # Number of context chunks to retrieve

class QueryResponse(BaseModel):
    query: str
    answer: str
    source_collection: str
    context_chunks: List[str]
    confidence: float
    safe: bool

class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    llm_available: bool

# ── Helper Functions ──────────────────────────
def has_access(role: str, collection: str) -> bool:
    """Check if user role has access to collection"""
    allowed_roles = COLLECTION_ACCESS.get(collection, [])
    return role in allowed_roles

def retrieve_context(collection: str, query: str, top_k: int = 5) -> tuple[List[str], float]:
    """Retrieve context from Qdrant based on query"""
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        query_vector = embedder.encode(query).tolist()
        
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            query_filter={
                "must": [
                    {
                        "key": "department",
                        "match": {"value": collection}
                    }
                ]
            }
        )
        
        chunks = []
        confidence = 0.0
        
        if results:
            for result in results:
                if result.payload:
                    chunks.append(result.payload.get("chunk", ""))
                    confidence = max(confidence, result.score)
        
        return chunks, confidence
        
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        return [], 0.0

def generate_answer(query: str, context: List[str]) -> str:
    """Generate answer using LLM with retrieved context"""
    try:
        context_text = "\n\n".join([f"[Source {i+1}]\n{chunk}" for i, chunk in enumerate(context)])
        
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

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, x_user_id: Optional[str] = Header(None)):
    """Query the FinSolve assistant"""
    
    # Use header user_id if provided, otherwise use request user_id
    user_id = x_user_id or request.user_id
    
    # ── Step 1: Input Guardrails ──────────────────────────
    is_safe, guardrail_message = check_input_guardrails(request.query, user_id)
    if not is_safe:
        raise HTTPException(
            status_code=400,
            detail=f"Query blocked: {guardrail_message}"
        )
    
    # ── Step 2: Semantic Routing ──────────────────────────
    route_result = router(request.query)
    collection = route_result.name if route_result else "general"
    
    # ── Step 3: Access Control ──────────────────────────
    if not has_access(request.role, collection):
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: Your role '{request.role}' cannot access '{collection}' collection"
        )
    
    # ── Step 4: Retrieve Context ──────────────────────────
    context_chunks, confidence = retrieve_context(collection, request.query, request.top_k)
    
    if not context_chunks:
        logger.warning(f"No context found for query in collection '{collection}'")
        context_chunks = ["No specific information found in knowledge base"]
    
    # ── Step 5: Generate Answer ──────────────────────────
    answer = generate_answer(request.query, context_chunks)
    
    # ── Step 6: Output Guardrails ──────────────────────────
    is_safe_output, safety_message = check_output_guardrails(answer, user_id)
    if not is_safe_output:
        logger.warning(f"Output blocked by guardrails: {safety_message}")
        answer = "I cannot provide this information due to safety guidelines."
    
    return QueryResponse(
        query=request.query,
        answer=answer,
        source_collection=collection,
        context_chunks=context_chunks[:3],  # Return top 3 for response
        confidence=confidence,
        safe=is_safe_output
    )

@app.get("/collections")
async def list_collections():
    """List available collections and roles"""
    return {
        "collections": list(COLLECTION_ACCESS.keys()),
        "access_control": COLLECTION_ACCESS
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "FinSolve AI Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "collections": "/collections",
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
