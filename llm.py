"""
Single place that constructs the model used by every agent.
Using Groq (free tier, no card required) instead of a paid API - the
default open-weight 120B model is strong enough for this pipeline and
costs nothing. Keeping this centralized makes it trivial to swap
models/providers per-agent later, which is itself a nice thing to mention
in an interview.
"""

import os
from langchain_groq import ChatGroq

DEFAULT_MODEL = "openai/gpt-oss-120b"


def get_llm(model: str = DEFAULT_MODEL, temperature: float = 0.4):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
            "and add it to your .env file."
        )
    return ChatGroq(model=model, temperature=temperature, api_key=api_key)
