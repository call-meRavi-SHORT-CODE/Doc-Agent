from pydantic import BaseModel
from typing import Literal

FrameworkChoices = Literal["langgraph", "autogen"]         # extend when needed
LLMChoices       = Literal['gpt-4o','gpt-4o-mini', "gpt-4.1", "gpt-4.1-mini", "gpt-3.5-turbo", 'llama3-8b-8192','llama3-70b-8192',"llama-3.3-70b-versatile"]    # extend when needed
VectorChoices    = Literal["faiss", "chroma", "annoy"]     # extend when needed

class RAGRequest(BaseModel):
    framework: FrameworkChoices
    llm_model: LLMChoices
    vector_store: VectorChoices
    query: str

class RAGResponse(BaseModel):
    answer: str