from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for the LLM."""

    model_name: str = "gpt-4o-mini"
    api_key: str = os.getenv("OPENAI_API_KEY")
    temperature: float = 0


@dataclass
class RAGConfig:
    """Configuration for RAG processing."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    retriever_k: int = 1
