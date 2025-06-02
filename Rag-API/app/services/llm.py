from langchain.chat_models import init_chat_model

def get_llm(model_name: str):
    """
    Returns an initialized chat LLM. 
    Supported: "openai", "groq", "gemini", etc.
    """
    if model_name == "openai":
        return init_chat_model("gpt-4o-mini", model_provider="openai")
    elif model_name == "groq":
        # Replace "groq-llm-name" with actual Groq model identifier
        return init_chat_model("groq-llm-name", model_provider="groq")
    elif model_name == "gemini":
        # Replace "gemini-llm-name" with actual Gemini identifier
        return init_chat_model("gemini-llm-name", model_provider="google")
    else:
        raise ValueError(f"Unsupported LLM model: {model_name}")
