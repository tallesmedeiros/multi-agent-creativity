from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class Solution:
    """💡 Solução otimizada com arrays NumPy internos"""

    id: int
    description: str
    scores: Dict[str, float] = field(default_factory=dict)
    scores_array: np.ndarray | None = field(default=None)
    llm_evaluations: Dict[str, Dict] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    consensus_history: List[int] = field(default_factory=list)

    def __post_init__(self):
        if self.scores and self.scores_array is None:
            self.scores_array = np.array(list(self.scores.values()))

    def __hash__(self):
        return hash(self.id)

    def get_combined_score(self, agent_id: str, use_llm: bool = False) -> float:
        """Combina scores algorítmicos com avaliações LLM"""
        algo_score = np.mean(list(self.scores.values())) if self.scores else 0.0

        if use_llm and str(agent_id) in self.llm_evaluations:
            llm_score = self.llm_evaluations[str(agent_id)].get("profile_alignment", 0.5)
            return 0.7 * llm_score + 0.3 * algo_score

        return algo_score
