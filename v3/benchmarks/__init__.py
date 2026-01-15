"""
Modern benchmarks (2024-2025) for tool-gated evaluation.
"""
from .livecodebench import LiveCodeBenchLoader
from .mmmu import MMMULoader
from .gpqa import GPQALoader
from .realtimeqa import RealTimeQALoader
from .gaia import GAIALoader
from .mcpmark import MCPMarkLoader

__all__ = [
    'LiveCodeBenchLoader',  # L1 Code
    'MMMULoader',           # L2 Vision
    'GPQALoader',           # L3 RAG
    'RealTimeQALoader',     # L4 Web
    'GAIALoader',           # All levels
    'MCPMarkLoader',        # MCP tool use
]
