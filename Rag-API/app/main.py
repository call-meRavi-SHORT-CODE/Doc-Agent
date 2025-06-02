from fastapi import FastAPI
from app.routers.ask import router as ask_router

app = FastAPI(
    title="Configurable RAG Agent API",
    version="1.0.0"
)

# Mount the /ask router
app.include_router(ask_router, prefix="", tags=["RAG"])

@app.get("/")
def root():
    return {"message": "Welcome to the RAG Agent API. Visit /docs for usage."}

@app.get("/health")
def health_check():
    return {"status": "ok"}
