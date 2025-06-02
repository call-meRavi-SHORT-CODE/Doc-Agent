import os
from dotenv import load_dotenv

load_dotenv()


DATA_DIR = 'Rag-API\data'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FAISS_INDEX_DIR = "vector_data/faiss_index"
CHROMA_INDEX_DIR = "vector_data/chroma_index"

# If you have specific embedding objects, import or configure them here:
# e.g. embeddings = OpenAIEmbeddings(...)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()  # adjust parameters as needed
