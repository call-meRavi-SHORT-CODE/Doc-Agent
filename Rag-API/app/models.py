from pydantic import BaseModel
from typing import Literal

FrameworkChoices = Literal["langgraph", "autogen","llamaindex","dspy"]         # extend when needed
LLMChoices       = Literal['gpt-4o','gpt-4o-mini', "gpt-4.1", "gpt-4.1-mini", "gpt-3.5-turbo", 'llama3-8b-8192','gemma2-9b-it',"llama-3.3-70b-versatile","gemini-2.0-flash"]    # extend when needed
VectorChoices    = Literal["faiss", "chroma", "annoy"]     # extend when needed

class RAGRequest(BaseModel):
    framework: FrameworkChoices
    llm_model: LLMChoices
    vector_store: VectorChoices
    query: str

class RAGResponse(BaseModel):
    answer: str