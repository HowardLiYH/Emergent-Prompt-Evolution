"""
Specialist Cache - Caches specialist knowledge for O(1) inference.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
import time
from datetime import datetime
import json
import os


@dataclass
class CachedSpecialist:
    """Cached specialist knowledge."""
    agent_id: int
    specialty: str
    best_tool: str
    prompt_template: str
    tool_config: Dict[str, Any]
    created_at: str
    access_count: int = 0
    last_accessed: str = ""


class SpecialistCache:
    """
    Caches specialist knowledge for fast inference.

    After training, specialist knowledge is cached:
    - Specialty assignment
    - Best tool for the specialty
    - Optimized prompt template
    - Tool configuration

    This allows O(1) inference without re-computing.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize specialist cache.

        Args:
            cache_dir: Directory for persistent cache
        """
        self.cache_dir = cache_dir
        self.cache: Dict[int, CachedSpecialist] = {}
        self.regime_to_specialist: Dict[str, int] = {}

    def cache_specialist(
        self,
        agent: Any,
        tool_config: Optional[Dict] = None
    ):
        """
        Cache a specialist's knowledge.

        Args:
            agent: Trained specialist agent
            tool_config: Optional tool configuration
        """
        specialty = getattr(agent, 'specialty', 'generalist')

        # Get best tool
        best_tool = 'L0'
        if hasattr(agent, 'beliefs') and specialty:
            best_tool = agent.beliefs.get_best_tool(specialty)

        # Build prompt template
        prompt_template = self._build_template(agent, specialty, best_tool)

        cached = CachedSpecialist(
            agent_id=agent.id,
            specialty=specialty,
            best_tool=best_tool,
            prompt_template=prompt_template,
            tool_config=tool_config or {},
            created_at=datetime.now().isoformat()
        )

        self.cache[agent.id] = cached

        if specialty and specialty != 'generalist':
            self.regime_to_specialist[specialty] = agent.id

    def cache_population(
        self,
        agents: List[Any],
        tool_configs: Optional[Dict[int, Dict]] = None
    ):
        """
        Cache an entire population.

        Args:
            agents: List of trained agents
            tool_configs: Optional tool configs per agent
        """
        for agent in agents:
            config = tool_configs.get(agent.id) if tool_configs else None
            self.cache_specialist(agent, config)

    def get_specialist(
        self,
        regime: str
    ) -> Optional[CachedSpecialist]:
        """
        Get cached specialist for a regime.

        Args:
            regime: Task regime

        Returns:
            CachedSpecialist or None
        """
        if regime not in self.regime_to_specialist:
            return None

        agent_id = self.regime_to_specialist[regime]
        if agent_id not in self.cache:
            return None

        specialist = self.cache[agent_id]
        specialist.access_count += 1
        specialist.last_accessed = datetime.now().isoformat()

        return specialist

    def get_prompt_for_regime(
        self,
        regime: str,
        task: str
    ) -> Optional[str]:
        """
        Get optimized prompt for a regime.

        Args:
            regime: Task regime
            task: Task description

        Returns:
            Optimized prompt or None
        """
        specialist = self.get_specialist(regime)
        if not specialist:
            return None

        return specialist.prompt_template.format(task=task)

    def _build_template(
        self,
        agent: Any,
        specialty: str,
        tool: str
    ) -> str:
        """Build optimized prompt template."""
        return f"""You are a specialist in {specialty} tasks.
Your primary tool is {tool}.

Based on your training experience:
- Focus on {specialty} problems
- Use {tool} for best results

Task: {{task}}

Provide your answer with confidence."""

    def benchmark_latency(
        self,
        n_lookups: int = 1000
    ) -> Dict[str, float]:
        """
        Benchmark cache lookup latency.

        Args:
            n_lookups: Number of lookups to perform

        Returns:
            Latency statistics
        """
        if not self.cache:
            return {'error': 'Cache is empty'}

        regimes = list(self.regime_to_specialist.keys())
        if not regimes:
            return {'error': 'No regime mappings'}

        latencies = []

        for _ in range(n_lookups):
            regime = regimes[_ % len(regimes)]

            start = time.perf_counter()
            _ = self.get_specialist(regime)
            end = time.perf_counter()

            latencies.append((end - start) * 1000)  # ms

        import numpy as np
        return {
            'mean_ms': np.mean(latencies),
            'median_ms': np.median(latencies),
            'p99_ms': np.percentile(latencies, 99),
            'max_ms': max(latencies),
            'min_ms': min(latencies),
            'n_lookups': n_lookups
        }

    def save(self, path: Optional[str] = None):
        """Save cache to disk."""
        path = path or os.path.join(self.cache_dir or '.', 'specialist_cache.json')

        data = {
            'cache': {
                str(k): {
                    'agent_id': v.agent_id,
                    'specialty': v.specialty,
                    'best_tool': v.best_tool,
                    'prompt_template': v.prompt_template,
                    'tool_config': v.tool_config,
                    'created_at': v.created_at,
                    'access_count': v.access_count,
                }
                for k, v in self.cache.items()
            },
            'regime_to_specialist': self.regime_to_specialist
        }

        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: Optional[str] = None):
        """Load cache from disk."""
        path = path or os.path.join(self.cache_dir or '.', 'specialist_cache.json')

        if not os.path.exists(path):
            return

        with open(path, 'r') as f:
            data = json.load(f)

        self.cache = {}
        for k, v in data.get('cache', {}).items():
            self.cache[int(k)] = CachedSpecialist(
                agent_id=v['agent_id'],
                specialty=v['specialty'],
                best_tool=v['best_tool'],
                prompt_template=v['prompt_template'],
                tool_config=v['tool_config'],
                created_at=v['created_at'],
                access_count=v.get('access_count', 0),
                last_accessed=v.get('last_accessed', '')
            )

        self.regime_to_specialist = data.get('regime_to_specialist', {})
