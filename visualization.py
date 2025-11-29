from typing import Dict, List

import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from IPython.display import HTML, display, clear_output
    import plotly.graph_objects as go
    import plotly.express as px

    VISUALIZATION_AVAILABLE = True
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")
except ImportError:  # pragma: no cover - optional dependency
    VISUALIZATION_AVAILABLE = False
    plt = None  # type: ignore
    sns = None  # type: ignore

    def display(*_, **__):  # type: ignore
        return None

    def clear_output(*_, **__):  # type: ignore
        return None

    class HTML:  # type: ignore
        def __init__(self, *_args, **_kwargs):
            pass

    class go:  # type: ignore
        class Figure:
            def __getattr__(self, _name):
                return self

        class Scatter:
            def __init__(self, *_, **__):
                pass

    class px:  # type: ignore
        @staticmethod
        def imshow(*_, **__):
            return None


from enums import CreativityMetric


class InteractiveVisualizer:
    """📊 Sistema de visualização rica e interativa"""

    @staticmethod
    def show_consensus_evolution(history: List[Dict]) -> None:
        if not history or not VISUALIZATION_AVAILABLE:
            return

        rounds = [h["round"] for h in history]
        avg_satisfaction = [np.mean(h["satisfactions"]) for h in history]
        min_satisfaction = [np.min(h["satisfactions"]) for h in history]
        max_satisfaction = [np.max(h["satisfactions"]) for h in history]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=rounds,
                y=avg_satisfaction,
                mode="lines+markers",
                name="Satisfação Média",
                line=dict(color="#4CAF50", width=3),
                marker=dict(size=10),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=rounds + rounds[::-1],
                y=max_satisfaction + min_satisfaction[::-1],
                fill="toself",
                fillcolor="rgba(76, 175, 80, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=rounds,
                y=min_satisfaction,
                mode="lines",
                name="Mínimo",
                line=dict(color="#FF6B6B", width=1, dash="dash"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=rounds,
                y=max_satisfaction,
                mode="lines",
                name="Máximo",
                line=dict(color="#4ECDC4", width=1, dash="dash"),
            )
        )

        fig.update_layout(
            title="📈 Evolução da Satisfação do Consenso",
            xaxis_title="Rodada de Negociação",
            yaxis_title="Satisfação",
            yaxis=dict(range=[0, 1]),
            hovermode="x unified",
        )

        fig.show()

    @staticmethod
    def show_solution_comparison(solutions, agents) -> None:
        if not solutions or not VISUALIZATION_AVAILABLE:
            return

        metrics = [m.value for m in CreativityMetric]
        score_matrix = np.array([[s.scores.get(m, 0) for m in metrics] for s in solutions])
        fig = px.imshow(
            score_matrix,
            labels=dict(x="Métrica", y="Solução", color="Score"),
            x=[m.icon + " " + m for m in metrics],
            y=[f"Solução {s.id+1}" for s in solutions],
            color_continuous_scale="Viridis",
        )
        if fig is not None:
            fig.update_layout(title="Comparação de Soluções por Métrica")
            fig.show()

    @staticmethod
    def show_top_solutions_bar_chart(final_consensus, solutions) -> None:
        if not solutions or not final_consensus or not VISUALIZATION_AVAILABLE or plt is None:
            return

        top_solutions = final_consensus[:5]
        labels = [f"Solução {s + 1}" for s in top_solutions]
        scores = [np.mean(list(solutions[s].scores.values())) for s in top_solutions]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, scores, color="#4ECDC4")
        plt.title("Top Soluções do Consenso")
        plt.ylim(0, 1)

        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{score:.2f}",
                     ha="center", va="bottom")

        plt.show()

    @staticmethod
    def show_negotiation_dynamics(agents) -> None:
        if not VISUALIZATION_AVAILABLE:
            return

    @staticmethod
    def show_final_consensus_breakdown(solutions, final_consensus, agents) -> None:
        if not VISUALIZATION_AVAILABLE:
            return

    @staticmethod
    def display_html_report(html: str) -> None:
        if VISUALIZATION_AVAILABLE:
            display(HTML(html))


__all__ = [
    "InteractiveVisualizer",
    "VISUALIZATION_AVAILABLE",
    "plt",
    "sns",
    "display",
    "clear_output",
    "HTML",
]
