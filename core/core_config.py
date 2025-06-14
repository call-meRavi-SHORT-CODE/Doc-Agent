import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

class LLMProvider(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    GROQ = "groq"

class Framework(Enum):
    LANGCHAIN = "langchain"
    LLAMAINDEX = "llamaindex"
    AUTOGEN = "autogen"
    CREWAI = "crewai"
    NEO4J = "neo4j"
    GRAPHLIT = "graphlit"
    AWS_BEDROCK = "aws_bedrock"

class VectorStore(Enum):
    FAISS = "faiss"
    CHROMADB = "chromadb"
    MILVUS = "milvus"
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"

@dataclass
class ObservabilityConfig:
    """Configuration for observability tools"""
    traceloop_enabled: bool = True
    grafana_tempo_endpoint: str = "http://localhost:3200"
    service_name: str = "rag-observability-tool"
    environment: str = "development"
    
@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    llm_provider: LLMProvider
    framework: Framework
    vector_store: VectorStore
    model_name: str
    embedding_model: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.observability = ObservabilityConfig()
        self.data_dir = os.getenv("DATA_DIR", "data")
        self.vector_data_dir = os.getenv("VECTOR_DATA_DIR", "vector_data")
        
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment"""
        keys = {}
        
        # OpenAI
        if openai_key := os.getenv("OPENAI_API_KEY"):
            keys["openai"] = openai_key
            
        # Gemini
        if gemini_key := os.getenv("GEMINI_API_KEY"):
            keys["gemini"] = gemini_key
            
        # Groq
        if groq_key := os.getenv("GROQ_API_KEY"):
            keys["groq"] = groq_key
            
        # Vector DB keys
        if qdrant_key := os.getenv("QDRANT_API_KEY"):
            keys["qdrant"] = qdrant_key
            
        if weaviate_key := os.getenv("WEAVIATE_API_KEY"):
            keys["weaviate"] = weaviate_key
            
        # Neo4j
        if neo4j_uri := os.getenv("NEO4J_URI"):
            keys["neo4j_uri"] = neo4j_uri
        if neo4j_user := os.getenv("NEO4J_USER"):
            keys["neo4j_user"] = neo4j_user
        if neo4j_password := os.getenv("NEO4J_PASSWORD"):
            keys["neo4j_password"] = neo4j_password
            
        # AWS
        if aws_access_key := os.getenv("AWS_ACCESS_KEY_ID"):
            keys["aws_access_key"] = aws_access_key
        if aws_secret_key := os.getenv("AWS_SECRET_ACCESS_KEY"):
            keys["aws_secret_key"] = aws_secret_key
        if aws_region := os.getenv("AWS_REGION"):
            keys["aws_region"] = aws_region
            
        return keys
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for specific provider"""
        return self.api_keys.get(provider.lower())
    
    def create_rag_config(
        self,
        llm_provider: str,
        framework: str,
        vector_store: str,
        model_name: str,
        embedding_model: str = None
    ) -> RAGConfig:
        """Create RAG configuration from string inputs"""
        
        # Default embedding models
        if not embedding_model:
            if llm_provider == "openai":
                embedding_model = "text-embedding-3-large"
            elif llm_provider == "gemini":
                embedding_model = "models/embedding-001"
            else:
                embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        
        return RAGConfig(
            llm_provider=LLMProvider(llm_provider),
            framework=Framework(framework),
            vector_store=VectorStore(vector_store),
            model_name=model_name,
            embedding_model=embedding_model
        )

# Global configuration instance
config_manager = ConfigManager()
