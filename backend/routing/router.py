import logging
from typing import Optional, List
from semantic_router import Route
from semantic_router.routers import SemanticRouter
from dotenv import load_dotenv
from semantic_router.encoders import HuggingFaceEncoder

load_dotenv()
logger = logging.getLogger(__name__)

# ── Routes (utterances written by you) ──────────────────────────
finance_route = Route(
    name="finance",
    utterances=[
        "What are the quarterly financial results?",
        "Show me the revenue and expenses",
        "What's our budget allocation?",
        "Tell me about investor relations",
        "What are Q3 earnings?",
        "Show financial metrics",
        "What's the annual financial report?",
        "Budget forecast for next year?",
        "Tell me about financial performance",
        "Profit margins and ROI analysis",
    ]
)

engineering_route = Route(
    name="engineering",
    utterances=[
        "What's our system architecture?",
        "Tell me about the API design",
        "How do we handle incidents?",
        "What's the SLA for our system?",
        "Explain the microservices setup",
        "How to onboard to the platform?",
        "What's our deployment process?",
        "Tell me about infrastructure setup",
        "What are the system requirements?",
        "How does our CI/CD pipeline work?",
    ]
)

marketing_route = Route(
    name="marketing",
    utterances=[
        "What are our campaign performance metrics?",
        "Tell me about brand guidelines",
        "How much budget for marketing?",
        "What's our market share?",
        "Tell me about competitor analysis",
        "What are our customer acquisition costs?",
        "Show campaign performance reports",
        "How many leads did we generate?",
        "What's our marketing strategy?",
        "Tell me about brand positioning",
    ]
)

hr_general_route = Route(
    name="hr_general",
    utterances=[
        "What's the leave policy?",
        "Tell me about health benefits",
        "How many vacation days do I get?",
        "What's the company handbook?",
        "Tell me about employee benefits",
        "What are company policies?",
        "How to request time off?",
        "What's the code of conduct?",
        "Tell me about HR policies",
        "How do I access company resources?",
    ]
)

cross_department_route = Route(
    name="general",
    utterances=[
        "General company information",
        "Tell me about the company",
        "What does FinSolve do?",
        "Company overview",
        "General FAQs",
        "Broad company questions",
        "Everything you need to know",
        "General guidance",
        "Multiple department info",
        "Cross-department information",
    ]
)

# ── Collection mappings ──────────────────────────────────────────
ROUTE_COLLECTIONS = {
    "finance":     ["finance", "general"],
    "engineering": ["engineering", "general"],
    "marketing":   ["marketing", "general"],
    "hr_general":  ["general", "hr"],
    "general":     ["general", "finance", "engineering", "marketing", "hr"],
}

ROLE_COLLECTIONS = {
    "employee":    ["general"],
    "finance":     ["finance", "general"],
    "engineering": ["engineering", "general"],
    "marketing":   ["marketing", "general"],
    "hr":          ["hr", "general"],
    "c_level":     ["general", "finance", "engineering", "marketing", "hr"],
}

# ── Initialize router ────────────────────────────────────────────
encoder = HuggingFaceEncoder()
routes = [finance_route, engineering_route, marketing_route, hr_general_route, cross_department_route]
route_layer = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")

# ── Main function ────────────────────────────────────────────────
def get_collections_for_query(query: str, user_role: str) -> Optional[List[str]]:
    # Step 1: classify query intent
    route = route_layer(query)
    route_name = route.name if route else "general"
    logger.info(f"Route: {route_name} | Role: {user_role}")

    # Step 2: get collections for this route
    route_cols = ROUTE_COLLECTIONS.get(route_name, ["general"])

    # Step 3: get collections user is allowed
    user_cols = ROLE_COLLECTIONS.get(user_role, ["general"])

    # Step 4: intersect
    allowed = [c for c in route_cols if c in user_cols]

    # Step 5: empty = access denied
    if not allowed:
        return None

    return allowed