from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import logging
import os
from config.core_config import VectorStore, config_manager

logger = logging.getLogger(__name__)

class BaseVectorStore(ABC):
    """Abstract base class for vector stores"""
    
    def __init__(self, embedding_model, collection_name: str = "default", **kwargs):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.kwargs = kwargs
        self.client = None
        self.index_path = kwargs.get('index_path')
    
    @abstractmethod
    def initialize(self):
        """Initialize the vector store client"""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        """Add documents to the vector store"""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    def delete_collection(self):
        """Delete the collection"""
        pass

class FAISSVectorStore(BaseVectorStore):
    """FAISS vector store implementation"""
    
    def initialize(self):
        try:
            from langchain_community.vectorstores import FAISS
            self.store_class = FAISS
            
            if self.index_path and os.path.exists(self.index_path):
                self.client = FAISS.load_local(
                    self.index_path, 
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"FAISS index loaded from {self.index_path}")
            else:
                self.client = None
                logger.info("FAISS store initialized (empty)")
        except ImportError:
            raise ImportError("langchain-community package is required for FAISS")
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        from langchain.schema import Document
        
        docs = [Document(page_content=doc, metadata=meta or {}) 
                for doc, meta in zip(documents, metadatas or [{}] * len(documents))]
        
        if self.client is None:
            self.client = self.store_class.from_documents(docs, self.embedding_model)
        else:
            self.client.add_documents(docs)
        
        if self.index_path:
            self.client.save_local(self.index_path)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        if self.client is None:
            return []
        
        results = self.client.similarity_search_with_score(query, k=k)
        return [(doc.page_content, score) for doc, score in results]
    
    def delete_collection(self):
        if self.index_path and os.path.exists(self.index_path):
            import shutil
            shutil.rmtree(self.index_path)
        self.client = None

class ChromaDBVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation"""
    
    def initialize(self):
        try:
            import chromadb
            from chromadb.config import Settings
            
            persist_directory = self.kwargs.get('persist_directory', './chroma_db')
            
            self.chroma_client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
            except:
                self.collection = self.chroma_client.create_collection(name=self.collection_name)
            
            logger.info(f"ChromaDB initialized with collection: {self.collection_name}")
        except ImportError:
            raise ImportError("chromadb package is required for ChromaDB")
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        if metadatas is None:
            metadatas = [{}] * len(documents)
        
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        documents = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        return [(doc, 1.0 - dist) for doc, dist in zip(documents, distances)]
    
    def delete_collection(self):
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except:
            pass

class MilvusVectorStore(BaseVectorStore):
    """Milvus vector store implementation"""
    
    def initialize(self):
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
            
            host = self.kwargs.get('host', 'localhost')
            port = self.kwargs.get('port', 19530)
            
            connections.connect("default", host=host, port=port)
            
            # Define schema if collection doesn't exist
            if not utility.has_collection(self.collection_name):
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.kwargs.get('dimension', 1536))
                ]
                schema = CollectionSchema(fields, description="Document collection")
                self.collection = Collection(self.collection_name, schema)
                
                # Create index
                index_params = {
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                self.collection.create_index("vector", index_params)
            else:
                self.collection = Collection(self.collection_name)
            
            self.collection.load()
            logger.info(f"Milvus initialized with collection: {self.collection_name}")
        except ImportError:
            raise ImportError("pymilvus package is required for Milvus")
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        # Generate embeddings
        embeddings = self.embedding_model.embed_documents(documents)
        
        entities = [
            documents,
            embeddings
        ]
        
        self.collection.insert(entities)
        self.collection.flush()
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        query_embedding = self.embedding_model.embed_query(query)
        
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            [query_embedding], 
            "vector", 
            search_params, 
            limit=k,
            output_fields=["content"]
        )
        
        return [(hit.entity.get("content"), 1.0 / (1.0 + hit.distance)) 
                for hit in results[0]]
    
    def delete_collection(self):
        from pymilvus import utility
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector store implementation"""
    
    def initialize(self):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams
            
            host = self.kwargs.get('host', 'localhost')
            port = self.kwargs.get('port', 6333)
            api_key = config_manager.get_api_key('qdrant')
            
            self.client = QdrantClient(host=host, port=port, api_key=api_key)
            
            # Create collection if it doesn't exist
            try:
                self.client.get_collection(self.collection_name)
            except:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.kwargs.get('dimension', 1536),
                        distance=Distance.COSINE
                    )
                )
            
            logger.info(f"Qdrant initialized with collection: {self.collection_name}")
        except ImportError:
            raise ImportError("qdrant-client package is required for Qdrant")
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        from qdrant_client.http.models import PointStruct
        
        if ids is None:
            ids = list(range(len(documents)))
        
        if metadatas is None:
            metadatas = [{}] * len(documents)
        
        # Generate embeddings
        embeddings = self.embedding_model.embed_documents(documents)
        
        points = [
            PointStruct(
                id=idx,
                vector=embedding,
                payload={"content": doc, **metadata}
            )
            for idx, doc, embedding, metadata in zip(ids, documents, embeddings, metadatas)
        ]
        
        self.client.upsert(collection_name=self.collection_name, points=points)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        query_embedding = self.embedding_model.embed_query(query)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k
        )
        
        return [(hit.payload.get("content", ""), hit.score) for hit in results]
    
    def delete_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass

class WeaviateVectorStore(BaseVectorStore):
    """Weaviate vector store implementation"""
    
    def initialize(self):
        try:
            import weaviate
            
            url = self.kwargs.get('url', 'http://localhost:8080')
            api_key = config_manager.get_api_key('weaviate')
            
            if api_key:
                auth_config = weaviate.AuthApiKey(api_key=api_key)
                self.client = weaviate.Client(url=url, auth_client_secret=auth_config)
            else:
                self.client = weaviate.Client(url=url)
            
            # Create class/schema if it doesn't exist
            class_name = self.collection_name.capitalize()
            
            if not self.client.schema.exists(class_name):
                class_schema = {
                    "class": class_name,
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"]
                        }
                    ]
                }
                self.client.schema.create_class(class_schema)
            
            self.class_name = class_name
            logger.info(f"Weaviate initialized with class: {self.class_name}")
        except ImportError:
            raise ImportError("weaviate-client package is required for Weaviate")
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        if metadatas is None:
            metadatas = [{}] * len(documents)
        
        with self.client.batch as batch:
            for doc, metadata in zip(documents, metadatas):
                data_object = {
                    "content": doc,
                    **metadata
                }
                batch.add_data_object(data_object, self.class_name)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        result = (
            self.client.query
            .get(self.class_name, ["content"])
            .with_near_text({"concepts": [query]})
            .with_limit(k)
            .with_additional(["certainty"])
            .do()
        )
        
        objects = result["data"]["Get"][self.class_name]
        return [(obj["content"], obj["_additional"]["certainty"]) for obj in objects]
    
    def delete_collection(self):
        try:
            self.client.schema.delete_class(self.class_name)
        except:
            pass

class VectorStoreFactory:
    """Factory class for creating vector store instances"""
    
    _providers = {
        VectorStore.FAISS: FAISSVectorStore,
        VectorStore.CHROMADB: ChromaDBVectorStore,
        VectorStore.MILVUS: MilvusVectorStore,
        VectorStore.QDRANT: QdrantVectorStore,
        VectorStore.WEAVIATE: WeaviateVectorStore,
    }
    
    @classmethod
    def create_vector_store(
        cls,
        provider: VectorStore,
        embedding_model,
        collection_name: str = "default",
        **kwargs
    ) -> BaseVectorStore:
        """Create vector store instance based on provider"""
        
        if provider not in cls._providers:
            raise ValueError(f"Unsupported vector store provider: {provider}")
        
        # Create and initialize vector store
        store_class = cls._providers[provider]
        store = store_class(
            embedding_model=embedding_model,
            collection_name=collection_name,
            **kwargs
        )
        store.initialize()
        
        return store
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available providers"""
        return [provider.value for provider in cls._providers.keys()]
                