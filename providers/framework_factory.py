# providers/framework_factory.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
from config.core_config import Framework, config_manager
from providers.llm_factory import BaseLLM
from providers.vector_store_factory import BaseVectorStore
import json

logger = logging.getLogger(__name__)

class BaseFramework(ABC):
    """Abstract base class for AI frameworks"""
    
    def __init__(self, llm: BaseLLM, vector_store: BaseVectorStore, **kwargs):
        self.llm = llm
        self.vector_store = vector_store
        self.kwargs = kwargs
        self.client = None
    
    @abstractmethod
    def initialize(self):
        """Initialize the framework"""
        pass
    
    @abstractmethod
    def create_rag_chain(self, system_prompt: str = None) -> Any:
        """Create RAG chain for question answering"""
        pass
    
    @abstractmethod
    def query(self, question: str, **kwargs) -> str:
        """Query the RAG system"""
        pass

class LangChainFramework(BaseFramework):
    """LangChain framework implementation"""
    
    def initialize(self):
        try:
            from langchain.chains import RetrievalQA
            from langchain.prompts import PromptTemplate
            from langchain_core.prompts import ChatPromptTemplate
            from langchain.chains.combine_documents import create_stuff_documents_chain
            from langchain.chains import create_retrieval_chain
            
            self.RetrievalQA = RetrievalQA
            self.PromptTemplate = PromptTemplate
            self.ChatPromptTemplate = ChatPromptTemplate
            self.create_stuff_documents_chain = create_stuff_documents_chain
            self.create_retrieval_chain = create_retrieval_chain
            
            logger.info("LangChain framework initialized")
        except ImportError:
            raise ImportError("langchain packages are required for LangChain framework")
    
    def create_rag_chain(self, system_prompt: str = None) -> Any:
        if not system_prompt:
            system_prompt = """
            You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know.
            
            Context: {context}
            Question: {input}
            Answer:
            """
        
        # Create prompt template
        prompt = self.ChatPromptTemplate.from_template(system_prompt)
        
        # Create document chain
        document_chain = self.create_stuff_documents_chain(self.llm, prompt)
        
        # Create retriever
        retriever = self.vector_store.client.as_retriever() if hasattr(self.vector_store.client, 'as_retriever') else None
        
        if retriever:
            # Create retrieval chain
            self.chain = self.create_retrieval_chain(retriever, document_chain)
        else:
            # Fallback for vector stores without retriever interface
            self.chain = document_chain
        
        return self.chain
    
    def query(self, question: str, **kwargs) -> str:
        if not hasattr(self, 'chain'):
            self.create_rag_chain()
        
        try:
            if hasattr(self.chain, 'invoke'):
                result = self.chain.invoke({"input": question})
                return result.get('answer', result)
            else:
                # Fallback query
                docs = self.vector_store.similarity_search(question, k=5)
                context = "\n".join([doc[0] for doc in docs])
                
                prompt = f"""
                Context: {context}
                Question: {question}
                Answer:
                """
                
                return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"LangChain query error: {e}")
            raise

class LlamaIndexFramework(BaseFramework):
    """LlamaIndex framework implementation"""
    
    def initialize(self):
        try:
            from llama_index.core import VectorStoreIndex, ServiceContext
            from llama_index.core.query_engine import RetrieverQueryEngine
            from llama_index.core.retrievers import VectorIndexRetriever
            
            self.VectorStoreIndex = VectorStoreIndex
            self.ServiceContext = ServiceContext
            self.RetrieverQueryEngine = RetrieverQueryEngine
            self.VectorIndexRetriever = VectorIndexRetriever
            
            logger.info("LlamaIndex framework initialized")
        except ImportError:
            raise ImportError("llama-index packages are required for LlamaIndex framework")
    
    def create_rag_chain(self, system_prompt: str = None) -> Any:
        # Create service context
        service_context = self.ServiceContext.from_defaults(llm=self.llm)
        
        # Create index from vector store
        self.index = self.VectorStoreIndex.from_vector_store(
            self.vector_store,
            service_context=service_context
        )
        
        # Create query engine
        self.query_engine = self.index.as_query_engine()
        
        return self.query_engine
    
    def query(self, question: str, **kwargs) -> str:
        if not hasattr(self, 'query_engine'):
            self.create_rag_chain()
        
        try:
            response = self.query_engine.query(question)
            return str(response)
        except Exception as e:
            logger.error(f"LlamaIndex query error: {e}")
            raise

class AutoGenFramework(BaseFramework):
    """AutoGen framework implementation"""
    
    def initialize(self):
        try:
            import autogen
            
            self.autogen = autogen
            logger.info("AutoGen framework initialized")
        except ImportError:
            raise ImportError("pyautogen package is required for AutoGen framework")
    
    def create_rag_chain(self, system_prompt: str = None) -> Any:
        if not system_prompt:
            system_prompt = """
            You are an assistant for question-answering tasks. 
            Use the retrieved context to answer questions accurately.
            """
        
        # Create AutoGen agents
        self.assistant = self.autogen.AssistantAgent(
            name="assistant",
            system_message=system_prompt,
            llm_config={"model": self.llm.model_name}
        )
        
        self.user_proxy = self.autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            code_execution_config=False
        )
        
        return self.assistant
    
    def query(self, question: str, **kwargs) -> str:
        if not hasattr(self, 'assistant'):
            self.create_rag_chain()
        
        try:
            # Get relevant context
            docs = self.vector_store.similarity_search(question, k=5)
            context = "\n".join([doc[0] for doc in docs])
            
            # Format question with context
            formatted_question = f"""
            Context: {context}
            
            Question: {question}
            
            Please provide a comprehensive answer based on the context.
            """
            
            # Start conversation
            self.user_proxy.initiate_chat(
                self.assistant,
                message=formatted_question
            )
            
            # Get the last message from assistant
            chat_history = self.user_proxy.chat_messages[self.assistant]
            return chat_history[-1]['content']
        except Exception as e:
            logger.error(f"AutoGen query error: {e}")
            raise

class CrewAIFramework(BaseFramework):
    """CrewAI framework implementation"""
    
    def initialize(self):
        try:
            from crewai import Agent, Task, Crew
            
            self.Agent = Agent
            self.Task = Task
            self.Crew = Crew
            
            logger.info("CrewAI framework initialized")
        except ImportError:
            raise ImportError("crewai package is required for CrewAI framework")
    
    def create_rag_chain(self, system_prompt: str = None) -> Any:
        if not system_prompt:
            system_prompt = """
            You are a research assistant specialized in answering questions 
            based on retrieved documents. Provide accurate and comprehensive answers.
            """
        
        # Create research agent
        self.researcher = self.Agent(
            role='Research Assistant',
            goal='Answer questions based on retrieved context',
            backstory=system_prompt,
            verbose=True,
            allow_delegation=False
        )
        
        return self.researcher
    
    def query(self, question: str, **kwargs) -> str:
        if not hasattr(self, 'researcher'):
            self.create_rag_chain()
        
        try:
            # Get relevant context
            docs = self.vector_store.similarity_search(question, k=5)
            context = "\n".join([doc[0] for doc in docs])
            
            # Create task
            task = self.Task(
                description=f"""
                Based on the following context, answer the question: {question}
                
                Context: {context}
                
                Provide a comprehensive and accurate answer.
                """,
                agent=self.researcher
            )
            
            # Create crew and execute
            crew = self.Crew(
                agents=[self.researcher],
                tasks=[task],
                verbose=True
            )
            
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            logger.error(f"CrewAI query error: {e}")
            raise

class Neo4jFramework(BaseFramework):
    """Neo4j framework implementation for graph-based RAG"""
    
    def initialize(self):
        try:
            from neo4j import GraphDatabase
            
            uri = config_manager.get_api_key('neo4j_uri') or 'bolt://localhost:7687'
            user = config_manager.get_api_key('neo4j_user') or 'neo4j'
            password = config_manager.get_api_key('neo4j_password') or 'password'
            
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info("Neo4j framework initialized")
        except ImportError:
            raise ImportError("neo4j package is required for Neo4j framework")
    
    def create_rag_chain(self, system_prompt: str = None) -> Any:
        # Initialize knowledge graph structure
        with self.driver.session() as session:
            # Create constraints and indexes
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
            session.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.content)")
        
        return self.driver
    
    def query(self, question: str, **kwargs) -> str:
        try:
            # Get relevant documents from vector store
            docs = self.vector_store.similarity_search(question, k=5)
            
            # Query knowledge graph for relationships
            with self.driver.session() as session:
                # Simple graph query - can be enhanced based on your graph structure
                result = session.run("""
                    MATCH (d:Document)
                    WHERE d.content CONTAINS $keyword
                    RETURN d.content as content
                    LIMIT 5
                """, keyword=question.split()[0] if question.split() else "")
                
                graph_context = [record["content"] for record in result]
            
            # Combine vector and graph context
            vector_context = "\n".join([doc[0] for doc in docs])
            combined_context = vector_context + "\n" + "\n".join(graph_context)
            
            # Generate answer using LLM
            prompt = f"""
            Based on the following context from both vector search and knowledge graph:
            
            Context: {combined_context}
            
            Question: {question}
            
            Please provide a comprehensive answer.
            """
            
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"Neo4j query error: {e}")
            raise

class AWSBedrockFramework(BaseFramework):
    """AWS Bedrock framework implementation"""
    
    def initialize(self):
        try:
            import boto3
            
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                aws_access_key_id=config_manager.get_api_key('aws_access_key'),
                aws_secret_access_key=config_manager.get_api_key('aws_secret_key'),
                region_name=config_manager.get_api_key('aws_region') or 'us-east-1'
            )
            
            logger.info("AWS Bedrock framework initialized")
        except ImportError:
            raise ImportError("boto3 package is required for AWS Bedrock framework")
    
    def create_rag_chain(self, system_prompt: str = None) -> Any:
        # AWS Bedrock setup
        return self.bedrock_client
    
    def query(self, question: str, **kwargs) -> str:
        try:
            # Get relevant context
            docs = self.vector_store.similarity_search(question, k=5)
            context = "\n".join([doc[0] for doc in docs])
            
            # Format prompt for Bedrock
            prompt = f"""
            Context: {context}
            
            Question: {question}
            
            Answer:
            """
            
            # Use Bedrock model (example with Claude)
            response = self.bedrock_client.invoke_model(
                modelId='anthropic.claude-v2',
                body=json.dumps({
                    'prompt': f"\n\nHuman: {prompt}\n\nAssistant:",
                    'max_tokens_to_sample': 1000,
                    'temperature': 0.7
                })
            )
            
            result = json.loads(response['body'].read())
            return result['completion']
        except Exception as e:
            logger.error(f"AWS Bedrock query error: {e}")
            raise

class GraphlitFramework(BaseFramework):
    """Graphlit framework implementation"""
    
    def initialize(self):
        # Graphlit would require specific setup
        logger.info("Graphlit framework initialized (placeholder)")
    
    def create_rag_chain(self, system_prompt: str = None) -> Any:
        # Placeholder implementation
        return None
    
    def query(self, question: str, **kwargs) -> str:
        # Placeholder - would implement Graphlit-specific logic
        docs = self.vector_store.similarity_search(question, k=5)
        context = "\n".join([doc[0] for doc in docs])
        
        prompt = f"""
        Context: {context}
        Question: {question}
        Answer:
        """
        
        return self.llm.generate(prompt)

class FrameworkFactory:
    """Factory class for creating framework instances"""
    
    _providers = {
        Framework.LANGCHAIN: LangChainFramework,
        Framework.LLAMAINDEX: LlamaIndexFramework,
        Framework.AUTOGEN: AutoGenFramework,
        Framework.CREWAI: CrewAIFramework,
        Framework.NEO4J: Neo4jFramework,
        Framework.AWS_BEDROCK: AWSBedrockFramework,
        Framework.GRAPHLIT: GraphlitFramework,
    }
    
    @classmethod
    def create_framework(
        cls,
        provider: Framework,
        llm: BaseLLM,
        vector_store: BaseVectorStore,
        **kwargs
    ) -> BaseFramework:
        """Create framework instance based on provider"""
        
        if provider not in cls._providers:
            raise ValueError(f"Unsupported framework provider: {provider}")
        
        # Create and initialize framework
        framework_class = cls._providers[provider]
        framework = framework_class(llm=llm, vector_store=vector_store, **kwargs)
        framework.initialize()
        
        return framework
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available providers"""
        return [provider.value for provider in cls._providers.keys()]
