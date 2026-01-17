"""
Procedural Memory - Stores learned skills and tool preferences.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class ToolPreference:
    """Learned preference for a tool in a regime."""
    tool: str
    regime: str
    success_rate: float  # Win rate with this tool
    usage_count: int
    last_update: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class StrategyLevel:
    """A locked-in strategy level for a regime."""
    regime: str
    level: int  # 0-3 or similar
    description: str
    locked_at: str = field(default_factory=lambda: datetime.now().isoformat())
    locked_by_wins: int = 0  # How many wins triggered the lock


class ProceduralMemory:
    """
    Layer 4: Procedural Memory

    Stores "how to" knowledge:
    - Tool preferences (which tool works for which regime)
    - Strategy levels (locked via competition wins)
    - Skill progression

    This is the "muscle memory" layer.
    """

    LOCK_THRESHOLD = 5  # Wins needed to lock a strategy

    def __init__(self):
        """Initialize procedural memory."""
        self.tool_preferences: Dict[str, Dict[str, ToolPreference]] = {}
        self.strategy_levels: Dict[str, StrategyLevel] = {}
        self.skill_history: List[Dict] = []

    def update_tool_preference(
        self,
        regime: str,
        tool: str,
        success: bool
    ):
        """
        Update tool preference based on outcome.

        Args:
            regime: Task regime
            tool: Tool that was used
            success: Whether the outcome was successful
        """
        if regime not in self.tool_preferences:
            self.tool_preferences[regime] = {}

        if tool not in self.tool_preferences[regime]:
            self.tool_preferences[regime][tool] = ToolPreference(
                tool=tool,
                regime=regime,
                success_rate=1.0 if success else 0.0,
                usage_count=1
            )
        else:
            pref = self.tool_preferences[regime][tool]
            # Update success rate with exponential moving average
            alpha = 0.3
            pref.success_rate = (
                alpha * (1.0 if success else 0.0) +
                (1 - alpha) * pref.success_rate
            )
            pref.usage_count += 1
            pref.last_update = datetime.now().isoformat()

    def get_best_tool(self, regime: str) -> Optional[str]:
        """
        Get the best tool for a regime based on learned preferences.

        Args:
            regime: Task regime

        Returns:
            Best tool name or None
        """
        if regime not in self.tool_preferences:
            return None

        prefs = self.tool_preferences[regime]
        if not prefs:
            return None

        # Find tool with highest success rate (with usage count tie-breaker)
        best = max(
            prefs.values(),
            key=lambda p: (p.success_rate, p.usage_count)
        )

        return best.tool

    def lock_strategy(
        self,
        regime: str,
        level: int,
        description: str,
        wins: int
    ):
        """
        Lock a strategy level for a regime.

        Called when agent has enough wins in a regime.

        Args:
            regime: Task regime
            level: Strategy level
            description: Description of the strategy
            wins: Number of wins that triggered the lock
        """
        if regime in self.strategy_levels:
            existing = self.strategy_levels[regime]
            if level <= existing.level:
                return  # Don't downgrade

        self.strategy_levels[regime] = StrategyLevel(
            regime=regime,
            level=level,
            description=description,
            locked_by_wins=wins
        )

        # Record in skill history
        self.skill_history.append({
            'regime': regime,
            'level': level,
            'description': description,
            'timestamp': datetime.now().isoformat()
        })

    def get_strategy_level(self, regime: str) -> Optional[StrategyLevel]:
        """Get the current strategy level for a regime."""
        return self.strategy_levels.get(regime)

    def check_for_lock(
        self,
        regime: str,
        wins: int,
        tool: str
    ) -> bool:
        """
        Check if a strategy should be locked based on wins.

        Args:
            regime: Task regime
            wins: Total wins in this regime
            tool: Dominant tool

        Returns:
            True if a new lock was created
        """
        if wins < self.LOCK_THRESHOLD:
            return False

        # Determine level based on wins
        level = min(3, wins // self.LOCK_THRESHOLD)

        current = self.strategy_levels.get(regime)
        if current and current.level >= level:
            return False

        description = f"Locked {tool} for {regime} at level {level}"
        self.lock_strategy(regime, level, description, wins)

        return True

    def get_skill_summary(self) -> Dict:
        """Get a summary of learned skills."""
        return {
            'n_regimes_with_preferences': len(self.tool_preferences),
            'n_locked_strategies': len(self.strategy_levels),
            'locked_regimes': list(self.strategy_levels.keys()),
            'total_skill_events': len(self.skill_history),
        }

    def export(self) -> Dict:
        """Export to serializable dict."""
        return {
            'tool_preferences': {
                regime: {
                    tool: {
                        'success_rate': p.success_rate,
                        'usage_count': p.usage_count,
                        'last_update': p.last_update,
                    }
                    for tool, p in prefs.items()
                }
                for regime, prefs in self.tool_preferences.items()
            },
            'strategy_levels': {
                regime: {
                    'level': sl.level,
                    'description': sl.description,
                    'locked_at': sl.locked_at,
                    'locked_by_wins': sl.locked_by_wins,
                }
                for regime, sl in self.strategy_levels.items()
            },
            'skill_history': self.skill_history,
        }

    def import_from(self, data: Dict):
        """Import from dict."""
        for regime, prefs in data.get('tool_preferences', {}).items():
            self.tool_preferences[regime] = {}
            for tool, p_data in prefs.items():
                self.tool_preferences[regime][tool] = ToolPreference(
                    tool=tool,
                    regime=regime,
                    success_rate=p_data['success_rate'],
                    usage_count=p_data['usage_count'],
                    last_update=p_data.get('last_update', '')
                )

        for regime, sl_data in data.get('strategy_levels', {}).items():
            self.strategy_levels[regime] = StrategyLevel(
                regime=regime,
                level=sl_data['level'],
                description=sl_data['description'],
                locked_at=sl_data.get('locked_at', ''),
                locked_by_wins=sl_data.get('locked_by_wins', 0)
            )

        self.skill_history = data.get('skill_history', [])
