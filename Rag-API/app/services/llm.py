from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq


def get_llm(model_name: str):
    """
    Returns an initialized chat LLM. 
    Supported: "openai", "groq", "gemini", etc.
    """

    if "gpt" in model_name:
        return init_chat_model(model_name, model_provider="openai")

    elif model_name == "llama3-8b-8192":
        return ChatGroq(model="llama3-8b-8192",temperature=0.5)
    
    elif model_name == "llama3-70b-8192":
        return ChatGroq(model="llama3-70b-8192",temperature=0.5)
    
    
    
    
    elif model_name == "llama-3.3-70b-versatile":
        # Replace "groq-llm-name" with actual Groq model identifier
        return ChatGroq(model="llama-3.3-70b-versatile",temperature=0.5)


    elif model_name == "gemini":
        # Replace "gemini-llm-name" with actual Gemini identifier
        return init_chat_model("gemini-llm-name", model_provider="google")
    else:
        raise ValueError(f"Unsupported LLM model: {model_name}")
