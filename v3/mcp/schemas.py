"""
MCP Tool schemas and result types.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ToolLevel(Enum):
    """Tool capability levels."""
    L0 = "L0"  # No tool (base LLM)
    L1 = "L1"  # Code execution
    L2 = "L2"  # Vision/multimodal
    L3 = "L3"  # RAG/retrieval
    L4 = "L4"  # Web access
    L5 = "L5"  # Agent orchestration


@dataclass
class ToolSchema:
    """Schema for an MCP tool."""
    name: str
    level: ToolLevel
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    required_api_keys: List[str] = field(default_factory=list)
    rate_limit: Optional[int] = None  # Requests per minute

    def to_dict(self) -> Dict:
        """Convert to MCP-compatible dict."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class ToolResult:
    """Result from an MCP tool execution."""
    success: bool
    content: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens_used: int = 0
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to serializable dict."""
        return {
            "success": self.success,
            "content": str(self.content) if self.content else None,
            "error": self.error,
            "metadata": self.metadata,
            "tokens_used": self.tokens_used,
            "execution_time_ms": self.execution_time_ms,
        }


# Pre-defined tool schemas
CODE_EXECUTION_SCHEMA = ToolSchema(
    name="code_execution",
    level=ToolLevel.L1,
    description="Execute Python code and return results. Use for calculations, data processing, and algorithmic tasks.",
    input_schema={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute"
            },
            "timeout": {
                "type": "integer",
                "description": "Execution timeout in seconds",
                "default": 30
            }
        },
        "required": ["code"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "stdout": {"type": "string"},
            "stderr": {"type": "string"},
            "return_value": {"type": "any"}
        }
    },
    required_api_keys=["GEMINI_API_KEY"]
)

VISION_SCHEMA = ToolSchema(
    name="vision",
    level=ToolLevel.L2,
    description="Analyze images, charts, and visual content. Use for image understanding and visual QA.",
    input_schema={
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "Base64 encoded image or URL"
            },
            "question": {
                "type": "string",
                "description": "Question about the image"
            }
        },
        "required": ["image", "question"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"}
        }
    },
    required_api_keys=["GEMINI_API_KEY"]
)

RAG_SCHEMA = ToolSchema(
    name="rag",
    level=ToolLevel.L3,
    description="Retrieve relevant documents and answer questions based on retrieved context. Use for document QA.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "top_k": {
                "type": "integer",
                "description": "Number of documents to retrieve",
                "default": 5
            }
        },
        "required": ["query"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "sources": {"type": "array", "items": {"type": "string"}}
        }
    },
    required_api_keys=[]  # Uses local ChromaDB
)

WEB_SEARCH_SCHEMA = ToolSchema(
    name="web_search",
    level=ToolLevel.L4,
    description="Search the web for current information. Use for real-time data and recent events.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return",
                "default": 5
            }
        },
        "required": ["query"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "results": {"type": "array"},
            "answer": {"type": "string"}
        }
    },
    required_api_keys=["TAVILY_API_KEY"],
    rate_limit=60
)

ORCHESTRATOR_SCHEMA = ToolSchema(
    name="orchestrator",
    level=ToolLevel.L5,
    description="Coordinate multiple sub-tasks and agents. Use for complex multi-step workflows.",
    input_schema={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Complex task to decompose and execute"
            },
            "max_steps": {
                "type": "integer",
                "description": "Maximum steps to execute",
                "default": 5
            }
        },
        "required": ["task"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "result": {"type": "string"},
            "steps_taken": {"type": "array"}
        }
    },
    required_api_keys=["GEMINI_API_KEY"]
)

# All schemas
ALL_SCHEMAS = {
    "L1": CODE_EXECUTION_SCHEMA,
    "L2": VISION_SCHEMA,
    "L3": RAG_SCHEMA,
    "L4": WEB_SEARCH_SCHEMA,
    "L5": ORCHESTRATOR_SCHEMA,
}
