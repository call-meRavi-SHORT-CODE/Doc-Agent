# app/routers/ask.py

from fastapi import APIRouter, HTTPException
from app.models import RAGRequest, RAGResponse
from app.services.vector_store import get_vector_store
from app.services.llm import get_llm,get_llama_index_llm
from app.services.rag_chain import build_rag_retrieval_chain
from app.services.frameworks import get_agent
import logging
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from config import OPENAI_API_KEY
import asyncio

router = APIRouter()


@router.post("/ask", response_model=RAGResponse)
def ask(request: RAGRequest):
    try:
        # 1) Initialize™
        vector_store = get_vector_store(request.vector_store)

        llm = get_llm(request.llm_model)
        
        rag_chain = build_rag_retrieval_chain(llm, vector_store)

        # 2) Create agent

        if request.framework == "langgraph":
            llm = get_llm(request.llm_model)
        elif request.framework =="llamaindex":
            llm = get_llama_index_llm(request.llm_model)

        else:
            return True
        
        
        agent = get_agent(request.framework, llm, rag_chain)

        if request.framework == "langgraph":
            # 3) Run the agent in “stream” mode
            inputs = {"messages": [("user", request.query)]}
            response_text = ""
            for step in agent.stream(inputs, stream_mode="values"):
                msg = step["messages"][-1]
                response_text += msg.content

            return RAGResponse(answer=response_text)

        elif request.framework == "llamaindex":

            async def main():
                response_text = await agent.run(user_msg=request.query)
                return response_text
            
            response_text = asyncio.run(main())

            return RAGResponse(answer=str(response_text))

        elif request.framework == "dspy":

            pred = agent(question=request.query)
            response_text = pred.answer


            return RAGResponse(answer=str(response_text))
        
        else:
            return True



       
        

        #print(response_text)
        #return RAGResponse(answer=response_text)

        

    except Exception as e:
        # Log the full traceback so you see exactly where it blew up
        logging.exception("Error inside /ask:")
        raise HTTPException(status_code=500, detail="Internal server error. Check logs for details.")