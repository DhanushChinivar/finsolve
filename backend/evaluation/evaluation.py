import os
import json
import logging
from typing import List, Dict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Sample evaluation dataset — 40 QA pairs
EVAL_DATASET = [
    # General
    {"question": "How many days of annual leave do employees get?", "ground_truth": "Employees get 15-21 days of privilege/annual leave per year."},
    {"question": "What is the notice period for junior staff?", "ground_truth": "The notice period for junior/mid staff is typically 30 days."},
    {"question": "What is the hybrid work policy?", "ground_truth": "FinSolve operates a hybrid model: 3 days office, 2 days remote per week."},
    {"question": "What are core working hours?", "ground_truth": "Core hours are 10:00 AM - 4:00 PM IST, mandatory for all employees."},
    {"question": "How much is the employee referral bonus?", "ground_truth": "Referral bonus ranges from ₹10,000 to ₹50,000 depending on the role."},
    {"question": "What is the maternity leave policy?", "ground_truth": "26 weeks paid leave for first two children, 12 weeks for subsequent children."},
    {"question": "How do I submit an IT support ticket?", "ground_truth": "Submit via IT Service Portal at serviceportal.finsolvtech.com or call +91-80-6789-0123."},
    {"question": "What is the gym membership subsidy?", "ground_truth": "FinSolve subsidizes gym membership up to ₹2,000 per month."},
    {"question": "What is the dress code on Fridays?", "ground_truth": "Smart casual or ethnic wear is permitted on Fridays."},
    {"question": "How many sick days are allowed per year?", "ground_truth": "Employees get 12 sick days per year, which are non-cumulative."},

    # Finance
    {"question": "What was the total marketing spend in Q1 2024?", "ground_truth": "Total marketing spend in Q1 2024 was ₹25 Crore."},
    {"question": "How many customers were acquired in Q1 2024?", "ground_truth": "148,500 new customers were acquired in Q1 2024."},
    {"question": "What was the ROI of the InstantPay campaign?", "ground_truth": "The InstantPay campaign achieved a 3.36x ROI."},
    {"question": "What was the customer acquisition cost in Q1?", "ground_truth": "The CAC was ₹1,681 per customer in Q1 2024."},
    {"question": "Which campaign had the highest ROI in Q1?", "ground_truth": "The InstantPay Launch Campaign had the highest ROI at 3.36x."},
    {"question": "What was the blended ROI for Q1?", "ground_truth": "The blended ROI for Q1 2024 was 2.84x."},
    {"question": "What was the marketing attributed revenue in Q1?", "ground_truth": "Marketing attributed revenue in Q1 was ₹68.5 Lakh."},
    {"question": "What is the budget for the European market entry?", "ground_truth": "UK: ₹45L, Germany: ₹38L, France: ₹35L for European market entry campaigns."},
    {"question": "What was the conversion rate target vs actual in Q1?", "ground_truth": "Target was 12%, actual was 10.8%, below target by 1.2 percentage points."},
    {"question": "How many impressions did the InstantPay campaign get?", "ground_truth": "The InstantPay campaign received 2,100,000 impressions."},

    # Engineering
    {"question": "What was the average sprint velocity in Q4 2024?", "ground_truth": "Average velocity in Q4 2024 was 219 story points per sprint."},
    {"question": "How many features were shipped in 2024?", "ground_truth": "156 features were shipped in 2024."},
    {"question": "What was the deployment success rate in 2024?", "ground_truth": "The deployment success rate was 99.3% in 2024."},
    {"question": "How many engineers were on the team at end of 2024?", "ground_truth": "The team had 63 engineers at the end of 2024."},
    {"question": "What was the test coverage target?", "ground_truth": "The test coverage target was ≥85%."},
    {"question": "How many bugs were fixed in 2024?", "ground_truth": "287 bugs were fixed in 2024 across all sprints."},
    {"question": "What was the completion rate in Q4?", "ground_truth": "Q4 2024 completion rate was 99.1%."},
    {"question": "How many production deployments happened in 2024?", "ground_truth": "There were 52 production deployments in 2024."},
    {"question": "What major feature was launched in Q1 2024?", "ground_truth": "InstantPay (same-day payments) was launched in Q1 2024."},
    {"question": "What was the average code review time in Q4?", "ground_truth": "Average code review time in Q4 was 4.6 hours."},

    # Marketing
    {"question": "What platforms were used for social media marketing in Q1?", "ground_truth": "Instagram, LinkedIn, Twitter, Facebook, and YouTube were used."},
    {"question": "Which social media platform had highest engagement in Q1?", "ground_truth": "YouTube had the highest engagement rate at 8.9%."},
    {"question": "What was LinkedIn follower growth in Q1 2024?", "ground_truth": "LinkedIn grew by 40,200 followers (+27.7%) in Q1 2024."},
    {"question": "What was the email open rate for InstantPay campaign?", "ground_truth": "The email open rate was 30%, compared to industry average of 18%."},
    {"question": "What markets did FinSolve enter in Q1 2024?", "ground_truth": "FinSolve entered UK, Germany, and France markets in Q1 2024."},
    {"question": "What was the winning A/B test variant for landing page?", "ground_truth": "Variant B with minimal form (3 fields) and social proof won with 5.8% conversion rate."},
    {"question": "How many influencers were used in InstantPay campaign?", "ground_truth": "15 fintech micro-influencers were used in the InstantPay campaign."},
    {"question": "What was the recommendation for Q2 regarding France?", "ground_truth": "Invest additional ₹10 Lakh and increase influencer partnerships from 3 to 8."},
    {"question": "What was the total addressable reach in Q1?", "ground_truth": "Total addressable reach was 9.23 million impressions across all channels."},
    {"question": "What was the interest to consideration conversion rate?", "ground_truth": "Strong 77.4% interest-to-consideration conversion rate was achieved."},
]


def run_evaluation(rag_pipeline_fn) -> Dict:
    """
    Run RAGAs evaluation on the pipeline.
    rag_pipeline_fn: function that takes (question, role) and returns (answer, contexts)
    """
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in EVAL_DATASET:
        question = item["question"]
        ground_truth = item["ground_truth"]

        try:
            answer, retrieved_contexts = rag_pipeline_fn(question, "c_level")
            questions.append(question)
            answers.append(answer)
            contexts.append(retrieved_contexts)
            ground_truths.append(ground_truth)
        except Exception as e:
            logger.error(f"Failed on question: {question} — {e}")

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ],
        llm=llm,
    )

    logger.info(f"RAGAs Results: {results}")
    return results


if __name__ == "__main__":
    print("Evaluation dataset ready with", len(EVAL_DATASET), "QA pairs.")
    print("Run this after building the RAG pipeline in the API.")