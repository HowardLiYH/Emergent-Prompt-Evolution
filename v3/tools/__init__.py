"""
Real tool implementations for L1-L5 capabilities.

Note: L4 (External) now uses synthetic corpus retrieval, not live web search.
See docs/RETRIEVAL_DISTINCTION.md for explanation.
"""
from .code import CodeExecutionTool
from .vision import VisionTool
from .rigorous_rag import get_rigorous_rag  # L3 RAG (ChromaDB + BGE embeddings)
from .orchestrator import OrchestratorTool

__all__ = [
    'CodeExecutionTool',  # L1: Code execution
    'VisionTool',         # L2: Vision/image analysis
    'get_rigorous_rag',   # L3: RAG (rigorous implementation)
    'OrchestratorTool',   # L5: Agent orchestration
    # L4 (External) uses synthetic corpus - handled in run_training_v2.py
]
