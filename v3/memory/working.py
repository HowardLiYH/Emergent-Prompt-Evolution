"""
Working Memory - In-context memory for current task execution.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class WorkingMemoryEntry:
    """An entry in working memory."""
    content: str
    source: str  # 'task', 'retrieved', 'reasoning', 'tool_output'
    relevance: float = 1.0
    timestamp: str = ""


class WorkingMemory:
    """
    Layer 1: Working Memory

    In-context memory that holds:
    - Current task context
    - Retrieved memories from episodic/semantic
    - Reasoning trace
    - Recent tool outputs

    This is the "scratchpad" during task execution.
    """

    MAX_TOKENS = 2000  # Maximum tokens to include in context

    def __init__(self, max_entries: int = 10):
        """
        Initialize working memory.

        Args:
            max_entries: Maximum number of entries to hold
        """
        self.max_entries = max_entries
        self.entries: List[WorkingMemoryEntry] = []
        self.current_task: Optional[Dict] = None

    def set_task(self, task: Dict):
        """Set the current task context."""
        self.current_task = task
        self.clear()

        # Add task to working memory
        self.add(
            content=task.get('question', str(task)),
            source='task',
            relevance=1.0
        )

    def add(
        self,
        content: str,
        source: str,
        relevance: float = 1.0
    ):
        """
        Add an entry to working memory.

        Args:
            content: The content to add
            source: Source type
            relevance: Relevance score (0-1)
        """
        from datetime import datetime

        entry = WorkingMemoryEntry(
            content=content[:500],  # Truncate long content
            source=source,
            relevance=relevance,
            timestamp=datetime.now().isoformat()
        )

        self.entries.append(entry)

        # Evict oldest if over limit
        if len(self.entries) > self.max_entries:
            # Keep most relevant entries
            self.entries.sort(key=lambda e: e.relevance, reverse=True)
            self.entries = self.entries[:self.max_entries]

    def add_retrieved(self, memories: List[str], relevances: Optional[List[float]] = None):
        """Add retrieved memories to working memory."""
        for i, mem in enumerate(memories):
            rel = relevances[i] if relevances and i < len(relevances) else 0.8
            self.add(content=mem, source='retrieved', relevance=rel)

    def add_reasoning(self, reasoning: str):
        """Add reasoning trace."""
        self.add(content=reasoning, source='reasoning', relevance=0.9)

    def add_tool_output(self, output: str):
        """Add tool output."""
        self.add(content=output, source='tool_output', relevance=0.85)

    def get_context(self, max_tokens: Optional[int] = None) -> str:
        """
        Get the current working memory as a context string.

        Args:
            max_tokens: Maximum tokens to include

        Returns:
            Formatted context string
        """
        max_tokens = max_tokens or self.MAX_TOKENS

        parts = []
        current_length = 0

        # Sort by relevance
        sorted_entries = sorted(
            self.entries,
            key=lambda e: e.relevance,
            reverse=True
        )

        for entry in sorted_entries:
            # Rough token estimate
            entry_tokens = len(entry.content.split())

            if current_length + entry_tokens > max_tokens:
                break

            if entry.source == 'task':
                parts.append(f"Task: {entry.content}")
            elif entry.source == 'retrieved':
                parts.append(f"Memory: {entry.content}")
            elif entry.source == 'reasoning':
                parts.append(f"Thought: {entry.content}")
            elif entry.source == 'tool_output':
                parts.append(f"Tool result: {entry.content}")
            else:
                parts.append(entry.content)

            current_length += entry_tokens

        return "\n".join(parts)

    def clear(self):
        """Clear working memory."""
        self.entries = []

    def __len__(self) -> int:
        return len(self.entries)

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            'entries': [
                {
                    'content': e.content,
                    'source': e.source,
                    'relevance': e.relevance,
                    'timestamp': e.timestamp
                }
                for e in self.entries
            ],
            'current_task': self.current_task
        }
