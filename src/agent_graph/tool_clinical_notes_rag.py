from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from agent_graph.load_tools_config import LoadToolsConfig
from qdrant_client import QdrantClient
from pyprojroot import here
import os
from dotenv import load_dotenv
import threading

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("GITHUB_TOKEN")
TOOLS_CFG = LoadToolsConfig()


class ClinicalNotesRAGTool:
    """
    A tool for retrieving relevant clinical notes using a Retrieval-Augmented Generation (RAG) approach with vector embeddings.

    This tool leverages a pre-trained OpenAI embedding model to transform user queries into vector embeddings.
    It then uses these embeddings to query a Qdrant-based vector database to retrieve the top-k most relevant
    clinical notes from a specific collection stored in the database.

    Attributes:
        embedding_model (str): The name of the OpenAI embedding model used for generating vector representations of queries.
        vectordb_dir (str): The directory where the Qdrant database is persisted (for local setup).
        k (int): The number of top-k nearest neighbor documents to retrieve from the vector database.
        qdrant_url (str, optional): URL for a remote Qdrant server, or None for local storage.
        vectordb (QdrantVectorStore): The Qdrant vector database instance connected to the specified collection and embedding model.
    """
    
    # Class-level singleton instances
    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, embedding_model: str, vectordb_dir: str, k: int, collection_name: str, qdrant_url: str = None):
        """
        Implement singleton pattern to reuse the same instance for the same configuration.
        """
        # Create a unique key based on the configuration
        instance_key = (embedding_model, vectordb_dir, collection_name, qdrant_url)
        
        if instance_key not in cls._instances:
            with cls._lock:
                # Double-check after acquiring lock
                if instance_key not in cls._instances:
                    instance = super().__new__(cls)
                    cls._instances[instance_key] = instance
        
        return cls._instances[instance_key]

    def __init__(self, embedding_model: str, vectordb_dir: str, k: int, collection_name: str, qdrant_url: str = None) -> None:
        """
        Initializes the ClinicalNotesRAGTool with the necessary configurations.

        Args:
            embedding_model (str): The name of the embedding model (e.g., "text-embedding-3-small").
            vectordb_dir (str): The directory path where the Qdrant database is stored (for local setup).
            k (int): The number of nearest neighbor documents to retrieve based on query similarity.
            collection_name (str): The name of the collection inside the Qdrant database.
            qdrant_url (str, optional): URL for a remote Qdrant server, or None for local storage.
        """
        # Only initialize if this is the first time
        if hasattr(self, '_initialized'):
            # Update k value which can change between calls
            self.k = k
            return
        
        self._initialized = True
        self.embedding_model = embedding_model
        self.vectordb_dir = vectordb_dir
        self.k = k
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        
        # Check if vectordb_dir exists for local setup
        if not qdrant_url and not os.path.exists(here(vectordb_dir)):
            raise FileNotFoundError(f"Vector database directory not found: {here(vectordb_dir)}. Run prepare_vector_db.py first.")
        
        # Create a single client instance
        self.client = QdrantClient(
            url=qdrant_url,
            path=here(vectordb_dir) if not qdrant_url else None
        )
        
        # Create the vector store using the shared client
        self.vectordb = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=OpenAIEmbeddings(
                model=self.embedding_model,
                base_url="https://models.github.ai/inference",
                api_key=os.getenv("GITHUB_TOKEN")
            )
        )
        
        try:
            points_count = self.client.get_collection(collection_name).points_count
            print(f"Connected to Qdrant collection '{collection_name}' with {points_count} vectors\n")
        except Exception as e:
            print(f"Warning: Could not get collection info: {e}")

    def search(self, query: str) -> str:
        """
        Perform similarity search on the vector database.
        
        Args:
            query (str): The search query.
            
        Returns:
            str: Concatenated content from the top-k most similar documents.
        """
        docs = self.vectordb.similarity_search(query, k=self.k)
        return "\n\n".join([doc.page_content for doc in docs])


# Global singleton instance holder
_rag_tool_instance = None
_rag_tool_lock = threading.Lock()


def get_rag_tool_instance():
    """
    Get or create a singleton instance of ClinicalNotesRAGTool.
    """
    global _rag_tool_instance
    
    if _rag_tool_instance is None:
        with _rag_tool_lock:
            if _rag_tool_instance is None:
                _rag_tool_instance = ClinicalNotesRAGTool(
                    embedding_model=TOOLS_CFG.clinical_notes_rag_embedding_model,
                    vectordb_dir=TOOLS_CFG.clinical_notes_rag_vectordb_directory,
                    k=TOOLS_CFG.clinical_notes_rag_k,
                    collection_name=TOOLS_CFG.clinical_notes_rag_collection_name,
                    qdrant_url=TOOLS_CFG.clinical_notes_rag_qdrant_url
                )
    
    return _rag_tool_instance


@tool
def lookup_clinical_notes(query: str) -> str:
    """Search among clinical notes to find the answer to the query. Input should be the query."""
    try:
        # Use the singleton instance
        rag_tool = get_rag_tool_instance()
        return rag_tool.search(query)
    except Exception as e:
        return f"Error searching clinical notes: {str(e)}. Please ensure the vector database is properly initialized."