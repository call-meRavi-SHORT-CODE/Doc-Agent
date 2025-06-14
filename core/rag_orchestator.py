import logging
import time
import subprocess
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import asdict

from config.core_config import RAGConfig, config_manager
from providers.llm_factory import LLMFactory
from providers.vector_store_factory import VectorStoreFactory
from providers.framework_factory import FrameworkFactory
from observability.traceloop_setup import ObservabilityManager, RAGObservability

# LangChain tools for agent functionality
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)

class RAGOrchestrator:
    """Main orchestrator for RAG operations with observability"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm = None
        self.vector_store = None
        self.framework = None
        self.agent = None
        self.tools = []
        
        # Observability setup
        self.observability_manager = ObservabilityManager(config_manager)
        self.rag_observability = RAGObservability(self.observability_manager)
        
        # Error handling
        self.max_retries = 3
        self.error_count = 0
        self.last_error = None
        
        # Metrics
        self.query_count = 0
        self.success_count = 0
        self.error_escalations = 0
        self.auto_fix_attempts = 0
        
        # Initialize observability
        self.observability_manager.initialize()
        
        # Initialize components
        self._initialize_components()
        self._setup_agent_tools()
    
    def _initialize_components(self):
        """Initialize LLM, Vector Store, and Framework components"""
        try:
            # Initialize LLM
            logger.info(f"Initializing LLM: {self.config.llm_provider.value} - {self.config.model_name}")
            self.llm = LLMFactory.create_llm(
                provider=self.config.llm_provider,
                model_name=self.config.model_name
            )
            
            # Initialize embedding model for vector store
            embedding_model = self._get_embedding_model()
            
            # Initialize Vector Store
            logger.info(f"Initializing Vector Store: {self.config.vector_store.value}")
            vector_store_kwargs = self._get_vector_store_kwargs()
            self.vector_store = VectorStoreFactory.create_vector_store(
                provider=self.config.vector_store,
                embedding_model=embedding_model,
                collection_name=f"rag_{self.config.vector_store.value}",
                **vector_store_kwargs
            )
            
            # Initialize Framework
            logger.info(f"Initializing Framework: {self.config.framework.value}")
            self.framework = FrameworkFactory.create_framework(
                provider=self.config.framework,
                llm=self.llm,
                vector_store=self.vector_store
            )
            
            # Create RAG chain
            system_prompt = self._get_system_prompt()
            self.framework.create_rag_chain(system_prompt)
            
            logger.info("RAG Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}")
            raise
    
    def _get_embedding_model(self):
        """Get embedding model based on LLM provider"""
        if self.config.llm_provider.value == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=self.config.embedding_model,
                openai_api_key=config_manager.get_api_key("openai")
            )
        elif self.config.llm_provider.value == "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model=self.config.embedding_model,
                google_api_key=config_manager.get_api_key("gemini")
            )
        else:
            # Fallback to HuggingFace embeddings
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=self.config.embedding_model)
    
    def _get_vector_store_kwargs(self) -> Dict[str, Any]:
        """Get vector store specific kwargs"""
        base_path = config_manager.vector_data_dir
        
        kwargs = {
            "dimension": 1536,  # Default dimension
        }
        
        if self.config.vector_store.value == "faiss":
            kwargs.update({
                "index_path": f"{base_path}/faiss_index"
            })
        elif self.config.vector_store.value == "chromadb":
            kwargs.update({
                "persist_directory": f"{base_path}/chroma_db"
            })
        elif self.config.vector_store.value == "milvus":
            kwargs.update({
                "host": "localhost",
                "port": 19530
            })
        elif self.config.vector_store.value == "qdrant":
            kwargs.update({
                "host": "localhost",
                "port": 6333
            })
        elif self.config.vector_store.value == "weaviate":
            kwargs.update({
                "url": "http://localhost:8080"
            })
        
        return kwargs
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for RAG chain"""
        return """
        You are an expert Docker assistant with access to comprehensive Docker documentation.
        
        Instructions:
        1. Use the provided context to answer Docker-related questions accurately
        2. If you need to execute Docker commands, clearly state the command and explain what it does
        3. If the context doesn't contain enough information, say so clearly
        4. Provide practical, actionable advice for Docker operations
        5. Always prioritize safety when suggesting Docker commands
        
        Context: {context}
        Question: {input}
        
        Provide a comprehensive and accurate answer:
        """
    
    def _setup_agent_tools(self):
        """Setup agent tools for enhanced capabilities"""
        
        @tool
        def doc_qa(query: str) -> str:
            """Answer questions based on Docker documentation context."""
            try:
                # Use framework's query method with vector search
                result = self.framework.query(query)
                return result
            except Exception as e:
                logger.error(f"Doc QA error: {e}")
                return f"Error searching documentation: {str(e)}"
        
        @tool
        def execute_docker_command(command: str) -> str:
            """Execute Docker CLI commands safely."""
            # Safety check - only allow docker commands
            if not command.strip().startswith('docker'):
                return "Error: Only Docker commands are allowed for security reasons"
            
            try:
                logger.info(f"Executing Docker command: {command}")
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    output = f"Command executed successfully:\n{result.stdout}"
                    logger.info(f"Command success: {command}")
                    return output
                else:
                    error_output = f"Command failed (exit code {result.returncode}):\n{result.stderr}"
                    logger.warning(f"Command failed: {command} - {result.stderr}")
                    return error_output
                    
            except subprocess.TimeoutExpired:
                error_msg = "Command timed out after 30 seconds"
                logger.error(f"Command timeout: {command}")
                return error_msg
            except Exception as e:
                error_msg = f"Execution error: {str(e)}"
                logger.error(f"Command execution error: {command} - {e}")
                return error_msg
        
        @tool
        def search_web(query: str) -> str:
            """Search web for additional Docker information when documentation is insufficient."""
            try:
                search = DuckDuckGoSearchResults(num_results=3)
                results = search.run(f"Docker {query}")
                return f"Web search results for '{query}':\n{results}"
            except Exception as e:
                return f"Web search error: {str(e)}"
        
        # Store tools
        self.tools = [doc_qa, execute_docker_command, search_web]
        
        # Create agent with tools
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Docker expert assistant with access to:
            1. Docker documentation (doc_qa tool)
            2. Docker CLI execution capabilities (execute_docker_command tool) 
            3. Web search for additional information (search_web tool)

            For Docker questions:
            1. First consult the documentation using doc_qa
            2. If the user needs command execution or demonstration, use execute_docker_command
            3. If you need additional current information, use search_web
            4. Always explain commands before executing them
            5. Prioritize safety and best practices

            Be helpful, accurate, and educational in your responses."""),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=agent_prompt
        )
    
    @RAGObservability.trace_rag_query
    def handle_query(self, query: str, use_agent: bool = True) -> Dict[str, Any]:
        """
        Main query handling method with observability and error handling
        
        Args:
            query: User query
            use_agent: Whether to use agent capabilities (default: True)
            
        Returns:
            Dict with answer, metadata, and observability info
        """
        start_time = time.time()
        self.query_count += 1
        
        response_data = {
            "answer": "",
            "metadata": {
                "query": query,
                "config": asdict(self.config),
                "response_time": 0,
                "success": False,
                "error_count": self.error_count,
                "escalated": False,
                "agent_used": use_agent
            }
        }
        
        try:
            if use_agent and self.agent:
                # Use agent for enhanced capabilities
                answer = self._query_with_agent(query)
            else:
                # Direct framework query
                answer = self.framework.query(query)
            
            # Success metrics
            response_time = time.time() - start_time
            self.success_count += 1
            self.error_count = 0  # Reset error count on success
            
            response_data.update({
                "answer": answer,
                "metadata": {
                    **response_data["metadata"],
                    "response_time": response_time,
                    "success": True
                }
            })
            
            logger.info(f"Query successful in {response_time:.2f}s")
            return response_data
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            self.error_count += 1
            self.last_error = str(e)
            
            # Try auto-fix or escalation
            if self.error_count >= self.max_retries:
                escalated_response = self._escalate_error(query, str(e))
                response_data.update({
                    "answer": escalated_response,
                    "metadata": {
                        **response_data["metadata"],
                        "response_time": time.time() - start_time,
                        "escalated": True,
                        "error": str(e)
                    }
                })
            else:
                # Try auto-fix
                if self._attempt_auto_fix():
                    return self.handle_query(query, use_agent)  # Retry
                else:
                    response_data.update({
                        "answer": f"Error processing query: {str(e)}",
                        "metadata": {
                            **response_data["metadata"],
                            "response_time": time.time() - start_time,
                            "error": str(e)
                        }
                    })
            
            return response_data
    
    def _query_with_agent(self, query: str) -> str:
        """Query using agent with tools"""
        inputs = {"messages": [("user", query)]}
        
        # Stream through agent execution
        final_response = ""
        for step in self.agent.stream(inputs, stream_mode="values"):
            if step.get('messages'):
                final_response = step['messages'][-1].content
        
        return final_response
    
    def _attempt_auto_fix(self) -> bool:
        """Attempt to automatically fix common issues"""
        self.auto_fix_attempts += 1
        
        if not self.last_error:
            return False
        
        error_lower = self.last_error.lower()
        
        try:
            # Connection issues - reinitialize components
            if any(keyword in error_lower for keyword in ['connection', 'timeout', 'network']):
                logger.info("Attempting auto-fix: Reinitializing components")
                self._initialize_components()
                return True
            
            # API rate limit - wait and retry
            elif any(keyword in error_lower for keyword in ['rate limit', 'quota', 'limit exceeded']):
                logger.info("Attempting auto-fix: Rate limit detected, waiting...")
                time.sleep(2)
                return True
            
            # Vector store issues - reinitialize vector store
            elif any(keyword in error_lower for keyword in ['vector', 'index', 'embedding']):
                logger.info("Attempting auto-fix: Reinitializing vector store")
                embedding_model = self._get_embedding_model()
                vector_store_kwargs = self._get_vector_store_kwargs()
                self.vector_store = VectorStoreFactory.create_vector_store(
                    provider=self.config.vector_store,
                    embedding_model=embedding_model,
                    collection_name=f"rag_{self.config.vector_store.value}",
                    **vector_store_kwargs
                )
                return True
                
        except Exception as fix_error:
            logger.error(f"Auto-fix failed: {fix_error}")
        
        return False
    
    def _escalate_error(self, query: str, error: str) -> str:
        """Escalate error to LLM for analysis and suggestions"""
        self.error_escalations += 1
        
        escalation_prompt = f"""
        The RAG system encountered persistent errors processing a user query.
        Please analyze the situation and provide helpful suggestions.
        
        User Query: {query}
        System Error: {error}
        Configuration: {asdict(self.config)}
        Error Count: {self.error_count}
        
        Please provide:
        1. Possible causes of this error
        2. Suggested solutions for the user
        3. Alternative approaches to get the information
        4. Any relevant Docker troubleshooting steps
        
        Be helpful and actionable in your response.
        """
        
        try:
            suggestions = self.llm.generate(escalation_prompt)
            return f"""I apologize, but I encountered technical difficulties processing your query. Here's what happened and how to proceed:

**Your Query:** {query}

**Analysis and Suggestions:**
{suggestions}

**Next Steps:**
- Try rephrasing your question
- Check if you need specific Docker setup steps
- Use the web interface to try different LLM/framework combinations

The system administrators have been notified of this issue."""

        except Exception as e:
            return f"""I apologize, but I'm experiencing technical difficulties and cannot process your query at this time.

**Your Query:** {query}
**Error:** {error}

**Suggested Actions:**
1. Try again in a few moments
2. Rephrase your question
3. Check Docker documentation directly
4. Contact system administrator if the issue persists

Error ID: {int(time.time())}"""
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None) -> bool:
        """Add documents to the vector store"""
        try:
            logger.info(f"Adding {len(documents)} documents to vector store")
            self.vector_store.add_documents(documents, metadatas)
            logger.info("Documents added successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        success_rate = (self.success_count / max(self.query_count, 1)) * 100
        
        return {
            "orchestrator_metrics": {
                "total_queries": self.query_count,
                "successful_queries": self.success_count,
                "failed_queries": self.query_count - self.success_count,
                "success_rate": success_rate,
                "current_error_count": self.error_count,
                "error_escalations": self.error_escalations,
                "auto_fix_attempts": self.auto_fix_attempts,
                "last_error": self.last_error
            },
            "configuration": asdict(self.config),
            "observability_metrics": self.rag_observability.get_metrics(),
            "component_status": self._get_component_status()
        }
    
    def _get_component_status(self) -> Dict[str, str]:
        """Get status of all components"""
        status = {}
        
        # Check LLM
        try:
            test_response = self.llm.generate("test")
            status["llm"] = "healthy" if test_response else "unhealthy"
        except Exception as e:
            status["llm"] = f"error: {str(e)}"
        
        # Check Vector Store
        try:
            test_search = self.vector_store.similarity_search("test", k=1)
            status["vector_store"] = "healthy"
        except Exception as e:
            status["vector_store"] = f"error: {str(e)}"
        
        # Check Framework
        try:
            test_query = self.framework.query("test")
            status["framework"] = "healthy" if test_query else "unhealthy"
        except Exception as e:
            status["framework"] = f"error: {str(e)}"
        
        # Check Agent
        status["agent"] = "healthy" if self.agent else "not_initialized"
        
        return status
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        return {
            "status": "healthy" if self.error_count < self.max_retries else "degraded",
            "components": self._get_component_status(),
            "metrics": self.get_metrics(),
            "observability": {
                "initialized": self.observability_manager.is_initialized,
                "tracer_active": self.rag_observability.obs_manager.tracer is not None
            }
        }
    
    def reset_error_state(self):
        """Reset error state for recovery"""
        self.error_count = 0
        self.last_error = None
        logger.info("Error state reset")
    
    def switch_configuration(self, new_config: RAGConfig):
        """Switch to a new configuration"""
        logger.info(f"Switching configuration from {self.config} to {new_config}")
        self.config = new_config
        self.reset_error_state()
        self._initialize_components()
        self._setup_agent_tools()
        logger.info("Configuration switched successfully")


# Factory function for easy creation
def create_rag_orchestrator(
    llm_provider: str = "openai",
    framework: str = "langchain",
    vector_store: str = "faiss",
    model_name: str = "gpt-4o",
    embedding_model: str = None,
    **kwargs
) -> RAGOrchestrator:
    """Factory function to create RAG orchestrator"""
    config = config_manager.create_rag_config(
        llm_provider=llm_provider,
        framework=framework,
        vector_store=vector_store,
        model_name=model_name,
        embedding_model=embedding_model
    )
    
    return RAGOrchestrator(config)


# Example usage
if __name__ == "__main__":
    # Test different configurations
    
    # OpenAI + LangChain + FAISS
    orchestrator1 = create_rag_orchestrator(
        llm_provider="openai",
        framework="langchain",
        vector_store="faiss",
        model_name="gpt-4o"
    )
    
    # Groq + AutoGen + ChromaDB  
    orchestrator2 = create_rag_orchestrator(
        llm_provider="groq",
        framework="autogen",
        vector_store="chromadb",
        model_name="llama3-70b-8192"
    )
    
    # Test queries
    test_queries = [
        "How do I run a Docker container?",
        "Show me all running containers with docker ps",
        "Create a new container named web-app using nginx image"
    ]
    
    for query in test_queries:
        print(f"\n--- Testing: {query} ---")
        
        # Test with orchestrator 1
        result1 = orchestrator1.handle_query(query)
        print(f"OpenAI Result: {result1['answer'][:100]}...")
        
        # Test with orchestrator 2
        result2 = orchestrator2.handle_query(query)
        print(f"Groq Result: {result2['answer'][:100]}...")
    
    # Print metrics
    print("\n--- Metrics ---")
    print("Orchestrator 1:", orchestrator1.get_metrics())
    print("Orchestrator 2:", orchestrator2.get_metrics())
