# app/routers/ask.py

from fastapi import APIRouter, HTTPException
from app.models import RAGRequest, RAGResponse
from app.services.vector_store import get_vector_store
from app.services.llm import get_llm
from app.services.rag_chain import build_rag_retrieval_chain
from app.services.frameworks import get_agent
import logging
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from config import OPENAI_API_KEY

router = APIRouter()



@router.post("/ask", response_model=RAGResponse)
def ask(request: RAGRequest):
    try:
        # 1) Initialize™
        vector_store = get_vector_store(request.vector_store)
        llm = get_llm(request.llm_model)
        rag_chain = build_rag_retrieval_chain(llm, vector_store)

        # 2) Create agent
        agent = get_agent(request.framework, llm, rag_chain)

        # 3) Run the agent in “stream” mode
        inputs = {"messages": [("user", request.query)]}
        response_text = ""
        for step in agent.stream(inputs, stream_mode="values"):
            msg = step["messages"][-1]
            response_text += msg.content

       
        

        #print(response_text)

        return RAGResponse(answer=response_text)

    except Exception as e:
        # Log the full traceback so you see exactly where it blew up
        logging.exception("Error inside /ask:")
        raise HTTPException(status_code=500, detail="Internal server error. Check logs for details.")