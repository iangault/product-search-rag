# Title: Prompt Templates
# Purpose: Store prompt text and prompt-building helpers for the
# RAG answer generation step.
# Date: 2026-04-22

SYSTEM_PROMPT = """
    You are a helpful Amazon shopping assistant.
    Answer the question using ONLY the following context (real product reviews + metadata).
    Always cite the product ASIN when possible.
    Be concise and specific."""


def build_prompt(query, context):
    """
    Combine system prompt, retrieved context, and user query into single formatted prompt string for LLM
    """
    return f"""{SYSTEM_PROMPT}

context:
{context}

question: 
{query}

Answer based on the Amazon datasets: """

# simple LLM-based query filter prompt idea from QED42 article https://www.qed42.com/insights/building-simple-effective-prompt-based-guardrails
GUARDRAIL_PROMPT = """
    You are a query filter for an Amazon shopping assistant. Determine if the following query is appropriate for the assistant to answer. Only allow queries that are related to Amazon appliances, kitchen products, or product shopping.
    
    Reply only with "APPROVED" if query is appropriate, and "REJECTED" if it is not.
    
    Query: {query}
    """

def is_query_relevant(query, llm):
    """
    Use the guardrail prompt to determine if query is relevant to the Amazon shopping assistant. Returns True if query is approved, False if rejected. Uses LLM to classify query before retrieval is attempted.
    """
    prompt = GUARDRAIL_PROMPT.format(query=query)
    response = llm.invoke(prompt).content.strip().upper()
    return response.startswith("APPROVED") #returns True if approved