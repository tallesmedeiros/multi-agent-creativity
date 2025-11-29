from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from enums import CreativityMetric, MetricPreference
from visualization import VISUALIZATION_AVAILABLE, plt, sns


@dataclass
class Agent:
    """👤 Agente otimizado com arrays pré-computados"""

    id: int
    name: str
    profile: str
    emoji: str
    metric_preferences: Dict[CreativityMetric, float]
    preferences_array: np.ndarray | None = field(default=None)
    personality_traits: Dict[str, float] = field(default_factory=dict)
    negotiation_history: List[Dict] = field(default_factory=list)
    satisfaction_curve: List[float] = field(default_factory=list)
    original_preferences: Dict[CreativityMetric, MetricPreference] = field(default_factory=dict)

    def __post_init__(self):
        if self.metric_preferences and self.preferences_array is None:
            self.preferences_array = np.array(list(self.metric_preferences.values()))

        if not self.personality_traits:
            self.personality_traits = {
                "flexibility": np.random.uniform(0.3, 0.7),
                "assertiveness": np.random.uniform(0.4, 0.8),
                "openness": np.random.uniform(0.5, 0.9),
            }

    def visualize_preferences(self) -> None:
        """Visualiza preferências do agente com gráficos ricos"""
        if not VISUALIZATION_AVAILABLE:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        metrics = list(self.original_preferences.keys())
        values = list(self.original_preferences.values())
        colors = [v.color for v in values]

        bars = ax1.bar(range(len(metrics)), [v.value for v in values], color=colors)
        ax1.set_xticks(range(len(metrics)))
        ax1.set_xticklabels([m.icon + " " + m.value for m in metrics], rotation=45, ha="right")
        ax1.set_ylim(0, 1)
        ax1.set_title(f"{self.emoji} {self.name} - Preferências de Criatividade")
        ax1.set_ylabel("Peso da Preferência")
        ax1.grid(axis="y", alpha=0.3)

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f"{value.visual}\n{value.value:.1f}",
                    ha="center", va="bottom", fontsize=9)

        traits = list(self.personality_traits.keys())
        trait_values = list(self.personality_traits.values())

        angles = np.linspace(0, 2 * np.pi, len(traits), endpoint=False).tolist()
        trait_values = trait_values + [trait_values[0]]
        angles += angles[:1]

        ax2 = plt.subplot(122, projection="polar")
        ax2.plot(angles, trait_values, "o-", linewidth=2, color="#FF6B6B")
        ax2.fill(angles, trait_values, alpha=0.25, color="#FF6B6B")
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(traits)
        ax2.set_ylim(0, 1)
        ax2.set_title("Traços de Personalidade")
        ax2.grid(True)

        plt.tight_layout()
        if sns is not None:
            sns.despine()
        plt.show()
