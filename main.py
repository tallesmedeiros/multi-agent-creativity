from datetime import datetime
import time

from config import Config
from enums import ConsensusMethod, CreativityMetric, MetricPreference
from framework import CompleteHybridFramework
from visualization import display, HTML


def main():
    """🚀 Função principal com execução completa e interativa"""

    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 8 + "🎯 FRAMEWORK HÍBRIDO DE DECISÃO MULTIAGENTE v5.0 🎯" + " " * 9 + "║")
    print("║" + " " * 78 + "║")
    print("║" + " " * 10 + "Performance Otimizada + Visualizações Completas" + " " * 20 + "║")
    print("╚" + "═" * 78 + "╝\n")

    print("🤔 Escolha o modo de operação:")
    print("  1. Algoritmo Puro (rápido e gratuito)")
    print("  2. Híbrido com LLM Simulado (demonstração completa)")

    choice = input("\nEscolha (1 ou 2): ").strip()
    use_llm = choice == "2"

    if use_llm:
        print("\n✅ Modo híbrido ativado com LLM simulado")
        print("   (Em produção, substitua pelo OpenAI API real)")
    else:
        print("\n⚡ Modo algorítmico puro ativado")

    framework = CompleteHybridFramework(use_llm=use_llm)

    print("\n" + "═" * 80)
    print("📝 ADICIONANDO SOLUÇÕES CRIATIVAS")
    print("═" * 80 + "\n")

    solutions = [
        "Usar uma moeda como chave de fenda improvisada para parafusos pequenos quando não há ferramenta apropriada",
        "Criar arte colocando papel sobre moedas e fazendo decalques para capturar as texturas e desenhos",
        "Usar moedas como pesos precisos para calibrar uma balança digital caseira",
        "Estabilizar uma mesa bamba colocando moedas sob a perna curta",
        "Criar um circuito elétrico simples usando moedas como elementos condutores",
        "Amarrar várias moedas em um fio fino para criar um móbile que produz sons metálicos suaves",
    ]

    framework.add_solutions_batch(solutions)

    print("\n" + "═" * 80)
    print("👥 CONFIGURANDO AGENTES INTELIGENTES")
    print("═" * 80)

    framework.add_agent(
        name="Engenheiro",
        profile="Especialista em soluções práticas e funcionais, valoriza eficiência técnica",
        emoji="⚙️",
        preferences={
            CreativityMetric.FLUENCIA: MetricPreference.MEDIA,
            CreativityMetric.ORIGINALIDADE: MetricPreference.BAIXA,
            CreativityMetric.FLEXIBILIDADE: MetricPreference.MEDIA,
            CreativityMetric.ELABORACAO: MetricPreference.ALTA,
            CreativityMetric.ADEQUACAO: MetricPreference.MUITO_ALTA,
            CreativityMetric.IMPACTO: MetricPreference.ALTA,
        },
    )

    time.sleep(0.5)

    framework.add_agent(
        name="Artista",
        profile="Criativo focado em estética e expressão, busca originalidade e beleza",
        emoji="🎨",
        preferences={
            CreativityMetric.FLUENCIA: MetricPreference.ALTA,
            CreativityMetric.ORIGINALIDADE: MetricPreference.MUITO_ALTA,
            CreativityMetric.FLEXIBILIDADE: MetricPreference.ALTA,
            CreativityMetric.ELABORACAO: MetricPreference.MEDIA,
            CreativityMetric.ADEQUACAO: MetricPreference.BAIXA,
            CreativityMetric.IMPACTO: MetricPreference.MEDIA,
        },
    )

    time.sleep(0.5)

    framework.add_agent(
        name="Cientista",
        profile="Pesquisador metódico, prioriza precisão e princípios científicos",
        emoji="🔬",
        preferences={
            CreativityMetric.FLUENCIA: MetricPreference.MEDIA,
            CreativityMetric.ORIGINALIDADE: MetricPreference.MEDIA,
            CreativityMetric.FLEXIBILIDADE: MetricPreference.BAIXA,
            CreativityMetric.ELABORACAO: MetricPreference.MUITO_ALTA,
            CreativityMetric.ADEQUACAO: MetricPreference.ALTA,
            CreativityMetric.IMPACTO: MetricPreference.ALTA,
        },
    )

    time.sleep(0.5)

    framework.add_agent(
        name="Empreendedor",
        profile="Visionário de negócios, busca inovação com viabilidade comercial",
        emoji="💼",
        preferences={
            CreativityMetric.FLUENCIA: MetricPreference.ALTA,
            CreativityMetric.ORIGINALIDADE: MetricPreference.ALTA,
            CreativityMetric.FLEXIBILIDADE: MetricPreference.MUITO_ALTA,
            CreativityMetric.ELABORACAO: MetricPreference.BAIXA,
            CreativityMetric.ADEQUACAO: MetricPreference.ALTA,
            CreativityMetric.IMPACTO: MetricPreference.MUITO_ALTA,
        },
    )

    print("\n" + "═" * 80)
    input("🎬 Pressione ENTER para iniciar o processo de consenso...")
    print("═" * 80)

    method = ConsensusMethod.HYBRID if use_llm else ConsensusMethod.NASH

    final_consensus = framework.run_hybrid_consensus(method=method, rounds=3)

    print("\n" + "═" * 80)
    print("📄 Deseja gerar relatório HTML detalhado? (s/n)")

    if input().lower() == "s":
        report = framework.generate_detailed_report()
        filename = f"relatorio_multiagente_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"✅ Relatório salvo como: {filename}")
        print("\n📊 Preview do relatório:")
        display(HTML(report[:1000] + "..."))

    print("\n" + "═" * 80)
    print("🎊 SIMULAÇÃO CONCLUÍDA COM SUCESSO! 🎊")
    print("═" * 80)
    print("\n💡 Dicas:")
    print("   • Para máxima performance, instale Numba: pip install numba")
    print("   • Para usar LLM real, configure Config.OPENAI_API_KEY")
    print("   • Visualizações interativas disponíveis no Jupyter/Colab")

    return framework


if __name__ == "__main__":
    main()
