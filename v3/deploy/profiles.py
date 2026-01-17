"""
Specialist Profile Extraction - Creates interpretable profiles from trained agents.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime


@dataclass
class SpecialistProfile:
    """A profile describing a specialist agent."""
    agent_id: int
    specialty: str
    best_tool: str
    win_rate: float
    n_wins: int
    prompt_template: str
    memory_summary: str
    confidence_calibration: float
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            'agent_id': self.agent_id,
            'specialty': self.specialty,
            'best_tool': self.best_tool,
            'win_rate': self.win_rate,
            'n_wins': self.n_wins,
            'prompt_template': self.prompt_template,
            'memory_summary': self.memory_summary,
            'confidence_calibration': self.confidence_calibration,
            'created_at': self.created_at
        }


def extract_profiles(
    agents: List[Any],
    competition_history: Optional[List[Dict]] = None
) -> List[SpecialistProfile]:
    """
    Extract profiles from trained agents.

    Args:
        agents: List of trained agents
        competition_history: Optional competition history for win rates

    Returns:
        List of specialist profiles
    """
    profiles = []

    # Compute win rates from history if provided
    win_rates = {}
    if competition_history:
        wins = {}
        totals = {}
        for round in competition_history:
            winner = round.get('winner_id')
            for p in round.get('participants', []):
                totals[p] = totals.get(p, 0) + 1
                if p == winner:
                    wins[p] = wins.get(p, 0) + 1

        for agent_id in totals:
            win_rates[agent_id] = wins.get(agent_id, 0) / totals[agent_id]

    for agent in agents:
        # Get specialty
        specialty = getattr(agent, 'specialty', None) or 'generalist'

        # Get best tool
        if hasattr(agent, 'beliefs'):
            best_tool = agent.beliefs.get_best_tool(specialty) if specialty != 'generalist' else 'L0'
        else:
            best_tool = 'L0'

        # Get win count
        if hasattr(agent, 'wins'):
            n_wins = sum(agent.wins.values())
        else:
            n_wins = 0

        # Get win rate
        agent_id = getattr(agent, 'id', 0)
        win_rate = win_rates.get(agent_id, 0.5)

        # Build prompt template
        prompt_template = _build_prompt_template(agent, specialty, best_tool)

        # Build memory summary
        memory_summary = _build_memory_summary(agent)

        # Estimate calibration
        confidence_calibration = 0.85  # Placeholder - would need actual prediction data

        profile = SpecialistProfile(
            agent_id=agent_id,
            specialty=specialty,
            best_tool=best_tool,
            win_rate=win_rate,
            n_wins=n_wins,
            prompt_template=prompt_template,
            memory_summary=memory_summary,
            confidence_calibration=confidence_calibration
        )

        profiles.append(profile)

    return profiles


def _build_prompt_template(
    agent: Any,
    specialty: str,
    tool: str
) -> str:
    """Build a prompt template from agent state."""
    template = f"""You are a specialist in {specialty} tasks.
Primary tool: {tool}

When solving tasks:
1. Use {tool} for optimal performance
2. Draw on your experience in {specialty}
3. Be confident in your area of expertise

Task: {{task}}"""

    return template


def _build_memory_summary(agent: Any) -> str:
    """Build a summary of agent's learned memories."""
    if not hasattr(agent, '_episodic_memory'):
        return "No memories recorded."

    memories = agent._episodic_memory
    if not memories:
        return "No memories recorded."

    # Summarize by regime
    by_regime = {}
    for m in memories:
        regime = m.get('regime', 'unknown')
        if regime not in by_regime:
            by_regime[regime] = []
        by_regime[regime].append(m)

    lines = []
    for regime, mems in by_regime.items():
        lines.append(f"- {regime}: {len(mems)} successful experiences")

    return "\n".join(lines) if lines else "No memories recorded."


def explain_specialization(agent: Any) -> str:
    """
    Generate a human-readable explanation of why an agent specialized.

    Args:
        agent: The agent to explain

    Returns:
        Explanation string
    """
    specialty = getattr(agent, 'specialty', None)
    wins = getattr(agent, 'wins', {})

    if not specialty:
        return f"Agent {agent.id} has not specialized yet."

    # Find wins in specialty
    specialty_wins = wins.get(specialty, 0)
    total_wins = sum(wins.values())

    # Get tool preference
    best_tool = 'L0'
    if hasattr(agent, 'beliefs'):
        best_tool = agent.beliefs.get_best_tool(specialty)

    explanation = f"""Agent {agent.id} specialized in {specialty}:

**Why this specialty?**
- Won {specialty_wins}/{total_wins} competitions in {specialty}
- This represents {specialty_wins/max(total_wins,1):.1%} of total wins
- Competitive exclusion pushed towards this niche

**Learned approach:**
- Primary tool: {best_tool}
- Consistent success pattern established through competition

**Key experience:**
"""

    # Add memory insight if available
    if hasattr(agent, '_episodic_memory') and agent._episodic_memory:
        relevant = [m for m in agent._episodic_memory if m.get('regime') == specialty]
        if relevant:
            most_recent = relevant[-1]
            explanation += f'- "{most_recent.get("strategy", "N/A")}"'
        else:
            explanation += "- No specific memories recorded"
    else:
        explanation += "- No memories available"

    return explanation
