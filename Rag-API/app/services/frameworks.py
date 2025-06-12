# app/services/framework_factory.py

from typing import Any
from app.tools.run_command import run_command_tool
from Prompt.prompts import system_prompt
from langgraph.prebuilt import create_react_agent



def get_agent(framework_name: str, llm, rag_chain):
    """
    Returns a ReACT‐style agent configured for the chosen framework.
    Currently supports "langgraph". AutoGen is a placeholder.
    """

    from typing import Dict
    from langchain.tools import tool
    from app.tools.doc_qa import doc_qa_tool as _raw_doc_qa_tool

    @tool
    def _doc_qa(query: str) -> str:
        """
        Retrieve an answer from the indexed documentation using RAG.
        """
        # Here, `rag_chain` is closed over from the outer scope.
        result: Dict[str, Any] = rag_chain.invoke({"input": query})
        return result.get("answer", "No answer found.")

    tools = [
        _doc_qa,            # Now a named function with its own docstring
        run_command_tool    # Already has a docstring in app/tools/run_command.py
    ]

    if framework_name == "langgraph":
        # Pass your actual system_prompt into create_react_agent
        return create_react_agent(model=llm, tools=tools, prompt=system_prompt)

    elif framework_name == "autogen":
        # Placeholder for an AutoGen‐based agent
        raise       ("AutoGen framework not implemented yet")

    else:
        raise ValueError(f"Unsupported framework: {framework_name}")
