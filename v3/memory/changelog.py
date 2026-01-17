"""
Changelog - Tracks behavioral changes over time with compaction.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import asyncio


@dataclass
class ChangeLogEntry:
    """A single entry in the behavioral changelog."""
    generation: int
    regime: str
    change_type: str  # 'win', 'strategy_lock', 'specialty_change', 'tool_switch'
    description: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ChangeLog:
    """
    Behavioral Changelog with Compaction.

    Tracks agent behavioral changes over time:
    - Competition wins
    - Strategy level-ups
    - Specialty changes
    - Tool preference changes

    When the log gets too large, it's compacted via LLM summarization.
    """

    COMPACTION_THRESHOLD = 100  # Compact when entries exceed this
    KEEP_RECENT = 20  # Keep this many recent entries after compaction

    def __init__(self, agent_id: int):
        """
        Initialize changelog.

        Args:
            agent_id: ID of the agent this changelog belongs to
        """
        self.agent_id = agent_id
        self.entries: List[ChangeLogEntry] = []
        self.summaries: List[str] = []  # Compacted summaries
        self._llm_model = None

    def log(
        self,
        generation: int,
        regime: str,
        change_type: str,
        description: str
    ):
        """
        Log a behavioral change.

        Args:
            generation: Current generation
            regime: Related regime
            change_type: Type of change
            description: Description of the change
        """
        entry = ChangeLogEntry(
            generation=generation,
            regime=regime,
            change_type=change_type,
            description=description
        )
        self.entries.append(entry)

    def log_win(self, generation: int, regime: str, tool: str):
        """Log a competition win."""
        self.log(
            generation=generation,
            regime=regime,
            change_type='win',
            description=f"Won in {regime} using {tool}"
        )

    def log_strategy_lock(
        self,
        generation: int,
        regime: str,
        level: int,
        tool: str
    ):
        """Log a strategy lock."""
        self.log(
            generation=generation,
            regime=regime,
            change_type='strategy_lock',
            description=f"Locked strategy level {level} for {regime} using {tool}"
        )

    def log_specialty_change(
        self,
        generation: int,
        old_specialty: Optional[str],
        new_specialty: str
    ):
        """Log a specialty change."""
        self.log(
            generation=generation,
            regime=new_specialty,
            change_type='specialty_change',
            description=f"Changed specialty from {old_specialty or 'None'} to {new_specialty}"
        )

    def needs_compaction(self) -> bool:
        """Check if changelog needs compaction."""
        return len(self.entries) >= self.COMPACTION_THRESHOLD

    async def compact(self, llm_client: Optional[Any] = None) -> str:
        """
        Compact the changelog via summarization.

        Args:
            llm_client: LLM client for summarization

        Returns:
            The generated summary
        """
        if len(self.entries) <= self.KEEP_RECENT:
            return ""

        # Entries to summarize
        to_summarize = self.entries[:-self.KEEP_RECENT]

        # Group by type for summary
        by_type = {}
        for entry in to_summarize:
            if entry.change_type not in by_type:
                by_type[entry.change_type] = []
            by_type[entry.change_type].append(entry)

        # Build summary
        summary_parts = []

        for change_type, entries in by_type.items():
            regimes = {}
            for e in entries:
                regimes[e.regime] = regimes.get(e.regime, 0) + 1

            regime_str = ", ".join(f"{r}({c})" for r, c in regimes.items())
            summary_parts.append(f"{change_type}: {len(entries)} events in {regime_str}")

        summary = f"[Gen {to_summarize[0].generation}-{to_summarize[-1].generation}] " + "; ".join(summary_parts)

        # Use LLM for more sophisticated summary if available
        if llm_client:
            try:
                summary = await self._llm_summarize(to_summarize, llm_client)
            except Exception:
                pass  # Fall back to simple summary

        # Store summary and keep recent entries
        self.summaries.append(summary)
        self.entries = self.entries[-self.KEEP_RECENT:]

        return summary

    async def _llm_summarize(
        self,
        entries: List[ChangeLogEntry],
        llm_client: Any
    ) -> str:
        """Summarize entries using LLM."""
        entries_text = "\n".join([
            f"- Gen {e.generation}: {e.description}"
            for e in entries[:20]  # Limit for prompt
        ])

        prompt = f"""Summarize these behavioral changelog entries into 2-3 key patterns:

{entries_text}

Focus on:
1. Core behavioral patterns
2. Strategies that consistently work
3. Key failures to avoid

Keep the summary under 100 words."""

        if hasattr(llm_client, 'generate_content'):
            response = llm_client.generate_content(prompt)
            return response.text[:300]
        elif hasattr(llm_client, 'generate'):
            response = llm_client.generate(prompt)
            return response[:300]
        else:
            raise ValueError("Unknown LLM client type")

    def get_recent_summary(self, n_entries: int = 10) -> str:
        """Get a summary of recent changes."""
        recent = self.entries[-n_entries:]

        if not recent:
            return "No recent changes."

        lines = []
        for e in recent:
            lines.append(f"Gen {e.generation}: {e.description}")

        return "\n".join(lines)

    def get_full_summary(self) -> str:
        """Get full changelog including compacted summaries."""
        parts = []

        if self.summaries:
            parts.append("Previous periods:")
            for s in self.summaries:
                parts.append(f"  {s}")

        parts.append(f"\nRecent ({len(self.entries)} entries):")
        parts.append(self.get_recent_summary())

        return "\n".join(parts)

    def export(self) -> Dict:
        """Export to serializable dict."""
        return {
            'agent_id': self.agent_id,
            'entries': [
                {
                    'generation': e.generation,
                    'regime': e.regime,
                    'change_type': e.change_type,
                    'description': e.description,
                    'timestamp': e.timestamp,
                }
                for e in self.entries
            ],
            'summaries': self.summaries,
        }

    def import_from(self, data: Dict):
        """Import from dict."""
        self.agent_id = data.get('agent_id', self.agent_id)
        self.entries = [
            ChangeLogEntry(
                generation=e['generation'],
                regime=e['regime'],
                change_type=e['change_type'],
                description=e['description'],
                timestamp=e.get('timestamp', '')
            )
            for e in data.get('entries', [])
        ]
        self.summaries = data.get('summaries', [])
