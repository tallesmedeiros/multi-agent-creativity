from enum import Enum


class MetricPreference(Enum):
    """📊 Níveis de preferência para métricas"""

    MUITO_BAIXA = (0.1, "▪", "#FF6B6B")
    BAIXA = (0.3, "▪▪", "#FFA07A")
    MEDIA = (0.5, "▪▪▪", "#FFD700")
    ALTA = (0.7, "▪▪▪▪", "#90EE90")
    MUITO_ALTA = (0.9, "▪▪▪▪▪", "#4CAF50")

    @property
    def value(self):  # type: ignore[override]
        return self._value_[0]

    @property
    def visual(self):
        return self._value_[1]

    @property
    def color(self):
        return self._value_[2]


class ConsensusMethod(Enum):
    """🎯 Métodos de consenso disponíveis"""

    BORDA = ("borda", "📊", "Contagem de Borda")
    CONDORCET = ("condorcet", "⚔️", "Vencedor de Condorcet")
    NASH = ("nash", "🤝", "Nash Bargaining")
    SHAPLEY = ("shapley", "💎", "Valor de Shapley")
    LLM_MEDIATED = ("llm", "🧠", "Mediado por LLM")
    HYBRID = ("hybrid", "🔀", "Híbrido Adaptativo")


class CreativityMetric(Enum):
    """✨ Métricas de criatividade"""

    FLUENCIA = ("fluência", "💡", "Quantidade de ideias")
    ORIGINALIDADE = ("originalidade", "🎨", "Unicidade e novidade")
    FLEXIBILIDADE = ("flexibilidade", "🔄", "Diversidade de abordagens")
    ELABORACAO = ("elaboração", "🔬", "Detalhamento e refinamento")
    ADEQUACAO = ("adequação", "✅", "Praticidade e viabilidade")
    IMPACTO = ("impacto", "💥", "Potencial transformador")

    @property
    def value(self):  # type: ignore[override]
        return self._value_[0]

    @property
    def icon(self):
        return self._value_[1]

    @property
    def description(self):
        return self._value_[2]
