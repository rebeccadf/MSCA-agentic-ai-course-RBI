"""OpenAI client and models shared by 02_run_agents.py and 03_evaluate_all_results.py.

Settings in .env: OPENAI_API_KEY, OPENAI_MODEL (fast), OPENAI_STRONG_MODEL (strong).
"""
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel, set_tracing_disabled

load_dotenv()

# Disable tracing to prevent non-fatal error messages about the OpenAI API key
set_tracing_disabled(True)

FAST_MODEL = os.getenv("OPENAI_MODEL") or "gpt-5-nano"
STRONG_MODEL = os.getenv("OPENAI_STRONG_MODEL") or "gpt-5-mini"

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    # On 429 (rate limit) the client waits for the server's Retry-After and retries
    max_retries=int(os.getenv("LLM_MAX_RETRIES", "10")),
    # Ask for gzip, not Brotli: with brotli < 1.2 installed, decoding Brotli
    # responses fails with "process() takes no keyword arguments"
    default_headers={"Accept-Encoding": "gzip, deflate"},
)

# Cheap model for routing, arithmetic and formatting
fast_model = OpenAIChatCompletionsModel(model=FAST_MODEL, openai_client=client)
# Stronger model for reading filings and judging
strong_model = OpenAIChatCompletionsModel(model=STRONG_MODEL, openai_client=client)
