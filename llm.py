"""One place to create the LLM. Switch provider/model in .env without touching code."""
import os

from dotenv import load_dotenv

load_dotenv()


def get_llm(temperature: float = 0):
    provider = os.getenv("LLM_PROVIDER", "openrouter")

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("LLM_MODEL", "google/gemini-2.5-flash"),
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",   # OpenRouter speaks the OpenAI API format
            temperature=temperature,
            max_tokens=1500,                           # cap reply length (answers are short)
            max_retries=2,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        temperature=temperature,
        max_retries=2,
    )