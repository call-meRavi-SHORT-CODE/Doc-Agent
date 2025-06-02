from pydantic import BaseModel
from typing import Literal

FrameworkChoices = Literal["langgraph", "autogen"]         # extend when needed
LLMChoices       = Literal["openai", "groq", "gemini"]     # extend when needed
VectorChoices    = Literal["faiss", "chroma", "annoy"]     # extend when needed

class RAGRequest(BaseModel):
    framework: FrameworkChoices
    llm_model: LLMChoices
    vector_store: VectorChoices
    query: str

class RAGResponse(BaseModel):
    answer: str
