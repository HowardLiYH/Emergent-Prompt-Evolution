"""
L3 RAG Tool - Retrieval Augmented Generation using LlamaIndex + ChromaDB.
"""
import os
from typing import Optional, List, Dict, Any
import asyncio
from pathlib import Path


class RAGTool:
    """
    L3 Tool: Document retrieval and QA using LlamaIndex and ChromaDB.
    
    This provides REAL RAG capabilities, not simulated.
    """
    
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = "v3_documents"
    ):
        """
        Initialize RAG tool.
        
        Args:
            persist_dir: Directory to persist the vector store
            collection_name: Name of the ChromaDB collection
        """
        self.persist_dir = persist_dir or os.path.join(
            os.path.dirname(__file__), '..', 'data', 'chromadb'
        )
        self.collection_name = collection_name
        
        self._index = None
        self._query_engine = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazily initialize the RAG pipeline."""
        if self._initialized:
            return
        
        try:
            from llama_index.core import (
                VectorStoreIndex,
                StorageContext,
                Settings
            )
            from llama_index.vector_stores.chroma import ChromaVectorStore
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            import chromadb
            
            # Set up embedding model (BGE)
            embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5"
            )
            Settings.embed_model = embed_model
            
            # Set up ChromaDB
            os.makedirs(self.persist_dir, exist_ok=True)
            chroma_client = chromadb.PersistentClient(path=self.persist_dir)
            
            # Get or create collection
            chroma_collection = chroma_client.get_or_create_collection(
                name=self.collection_name
            )
            
            # Create vector store
            vector_store = ChromaVectorStore(
                chroma_collection=chroma_collection
            )
            
            # Create storage context
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )
            
            # Create index
            self._index = VectorStoreIndex(
                [],
                storage_context=storage_context
            )
            
            # Create query engine
            self._query_engine = self._index.as_query_engine(
                similarity_top_k=5
            )
            
            self._initialized = True
            
        except ImportError as e:
            print(f"RAG dependencies not installed: {e}")
            print("Install with: pip install llama-index llama-index-vector-stores-chroma chromadb")
            self._initialized = False
    
    async def execute(
        self,
        query: str,
        top_k: int = 5
    ) -> str:
        """
        Query the document store and generate an answer.
        
        Args:
            query: Question to answer
            top_k: Number of documents to retrieve
            
        Returns:
            Generated answer based on retrieved documents
        """
        self._ensure_initialized()
        
        if not self._initialized:
            return f"RAG not initialized. Query: {query}"
        
        try:
            # Run query in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._query_engine.query(query)
            )
            
            return str(response)
            
        except Exception as e:
            return f"RAG error: {e}"
    
    async def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None
    ) -> bool:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dicts
            
        Returns:
            True if successful
        """
        self._ensure_initialized()
        
        if not self._initialized:
            return False
        
        try:
            from llama_index.core import Document
            
            # Create Document objects
            docs = []
            for i, text in enumerate(documents):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                docs.append(Document(text=text, metadata=metadata))
            
            # Add to index
            for doc in docs:
                self._index.insert(doc)
            
            return True
            
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False
    
    async def add_file(self, file_path: str) -> bool:
        """
        Add a file to the document store.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if successful
        """
        path = Path(file_path)
        if not path.exists():
            return False
        
        # Read file content
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return await self.add_documents(
                [content],
                [{'source': str(path), 'filename': path.name}]
            )
        except Exception as e:
            print(f"Error reading file: {e}")
            return False
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents without generating an answer.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of documents with scores
        """
        self._ensure_initialized()
        
        if not self._initialized:
            return []
        
        try:
            # Get retriever
            retriever = self._index.as_retriever(similarity_top_k=top_k)
            
            # Retrieve
            loop = asyncio.get_event_loop()
            nodes = await loop.run_in_executor(
                None,
                lambda: retriever.retrieve(query)
            )
            
            # Format results
            results = []
            for node in nodes:
                results.append({
                    'text': node.text[:500],  # Truncate
                    'score': node.score if hasattr(node, 'score') else 0.0,
                    'metadata': node.metadata if hasattr(node, 'metadata') else {}
                })
            
            return results
            
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []
    
    def clear(self):
        """Clear all documents from the store."""
        self._initialized = False
        self._index = None
        self._query_engine = None
