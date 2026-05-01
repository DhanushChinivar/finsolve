import logging
from typing import Optional, List
from semantic_router import Route, RouteLayer
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
        "What is our net revenue this year?",
        "How much did we spend on operations?",
        "What is the EBITDA for this quarter?",
        "Show me the cash flow statement",
        "What are the vendor payment details?",
        "How much is the operating expenditure?",
        "What is our gross margin percentage?",
        "Tell me about department budget breakdown",
        "What were Q1 Q2 Q3 Q4 financial numbers?",
        "What is our current burn rate?",
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
        "What is the uptime percentage for our services?",
        "Show me the incident report logs",
        "What happened during the last outage?",
        "What are the sprint velocity metrics?",
        "How many bugs were fixed this sprint?",
        "What is the API rate limit?",
        "Tell me about the database schema",
        "What services are in our tech stack?",
        "What is the mean time to recovery?",
        "Show me engineering performance metrics",
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
        "What was the total addressable reach in Q1?",
        "What was LinkedIn follower growth this quarter?",
        "How many impressions did our campaign get?",
        "What is our social media engagement rate?",
        "Show me Q1 Q2 Q3 Q4 marketing performance",
        "What was the conversion rate for our campaigns?",
        "Tell me about our digital marketing metrics",
        "What is our email open rate?",
        "How did our paid ads perform?",
        "What is the ROI on our marketing spend?",
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
        "What is the maternity leave policy?",
        "How do I apply for sick leave?",
        "What are the working hours?",
        "Tell me about the employee referral program",
        "What is the salary review process?",
        "How many employees are in the company?",
        "What is the onboarding process for new hires?",
        "Tell me about performance appraisal",
        "What is the dress code policy?",
        "How do I raise a grievance?",
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
        "What is FinSolve's mission?",
        "Tell me something about the organization",
        "How is the company structured?",
        "What are the company values?",
        "Give me a summary of the business",
        "What does the company do across all departments?",
        "Overview of all teams",
        "Who are the key stakeholders?",
        "What is the company culture like?",
        "Tell me everything about FinSolve",
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
route_layer = RouteLayer(encoder=encoder, routes=routes)

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