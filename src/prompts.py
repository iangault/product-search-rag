# Script for prompts

SYSTEM_PROMPT = """
    You are a helpful Amazon shopping assistant.
    Answer the question using ONLY the following context (real product reviews + metadata).
    Always cite the product ASIN when possible.
    Be concise and specific."""


def build_prompt(query, context):
    """
    Build final prompt that is passed to LLM
    """
    return f"""{SYSTEM_PROMPT}

context:
{context}

question: 
{query}

Answer based on the Amazon datasets: """
