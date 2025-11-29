import time
import re
from datetime import datetime
from multiprocessing import Pool
from typing import Dict, List

import numpy as np

from analysis import kendall_tau_fast
from cache import OptimizedCache
from config import Config, NUMBA_AVAILABLE
from enums import ConsensusMethod, CreativityMetric, MetricPreference
from llm_simulator import OptimizedLLMSimulator
from models.agent import Agent
from models.solution import Solution
from visualization import InteractiveVisualizer, VISUALIZATION_AVAILABLE, clear_output


class CompleteHybridFramework:
    """🚀 Framework Híbrido Completo com Performance e Visualizações"""

    def __init__(self, use_llm: bool = None):
        self.use_llm = use_llm if use_llm is not None else Config.USE_LLM
        self.solutions: List[Solution] = []
        self.agents: List[Agent] = []
        self.consensus_history: List[Dict] = []
        self.llm_simulator = OptimizedLLMSimulator()
        self.visualizer = InteractiveVisualizer()

        # Cache e pré-computação
        self.cache = OptimizedCache(Config.CACHE_SIZE)
        self._precomputed = {}
        self._patterns = None

        # Estatísticas
        self.stats = {
            "llm_calls": 0,
            "cache_hits": 0,
            "total_time": 0,
            "rounds_completed": 0,
            "vectorized_ops": 0,
            "parallel_ops": 0
        }

        # Sincroniza estatísticas com simulador
        self.llm_simulator.stats = self.stats

        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 15 + "🚀 FRAMEWORK HÍBRIDO v5.0 INICIALIZADO 🚀" + " " * 18 + "║")
        print("║" + " " * 78 + "║")
        print(f"║  Modo: {'🧠 LLM Habilitado' if self.use_llm else '⚡ Algoritmo Puro':<40} " + " " * 26 + "║")
        print(f"║  Cache: {'✅ Ativado' if Config.CACHE_LLM_RESPONSES else '❌ Desativado':<40} " + " " * 25 + "║")
        print(f"║  Paralelização: {'✅ ' + str(Config.MAX_WORKERS) + ' workers' if Config.USE_PARALLEL else '❌ Desativada':<40} " + " " * 19 + "║")
        print(f"║  Vetorização: {'✅ NumPy' if Config.USE_VECTORIZATION else '❌ Desativada':<40} " + " " * 19 + "║")
        print(f"║  Numba JIT: {'✅ Disponível' if NUMBA_AVAILABLE else '❌ Não disponível':<40} " + " " * 20 + "║")
        print(f"║  Visualização: {'📊 Interativa' if Config.PLOT_STYLE == 'interactive' else '📈 Estática':<40} " + " " * 17 + "║")
        print("╚" + "═" * 78 + "╝\n")

    def add_solution(self, description: str) -> None:
        """Adiciona solução com análise híbrida e visualização"""
        solution_id = len(self.solutions)
        solution = Solution(id=solution_id, description=description)

        # Análise algorítmica
        solution.scores = self._analyze_solution_algorithmic(description)
        solution.scores_array = np.array(list(solution.scores.values()))

        # Metadados
        solution.metadata = {
            "word_count": len(description.split()),
            "complexity": self._calculate_complexity(description),
            "timestamp": datetime.now().isoformat()
        }

        self.solutions.append(solution)

        # Visualização detalhada
        print(f"{'─' * 80}")
        print(f"✅ Solução {solution_id + 1} adicionada")
        print(f"📝 '{description[:60]}...'")
        print(f"📊 Complexidade: {'█' * int(solution.metadata['complexity'] * 10)}")
        print(f"📈 Scores:")
        for metric, score in solution.scores.items():
            bar = '█' * int(score * 10) + '░' * (10 - int(score * 10))
            print(f"   {metric:<15} {bar} {score:.2f}")

    def add_solutions_batch(self, descriptions: List[str]) -> None:
        """Adiciona múltiplas soluções em batch com paralelização"""
        start_time = time.time()

        print(f"\n{'═' * 80}")
        print(f"📝 ADICIONANDO {len(descriptions)} SOLUÇÕES")
        print(f"{'═' * 80}\n")

        if Config.USE_PARALLEL:
            # Cria pool local com context manager
            with Pool(processes=Config.MAX_WORKERS) as pool:
                scores_list = pool.map(self._analyze_solution_algorithmic, descriptions)
            self.stats["parallel_ops"] += 1
        else:
            scores_list = [self._analyze_solution_algorithmic(desc) for desc in descriptions]

        # Cria soluções
        for i, (desc, scores) in enumerate(zip(descriptions, scores_list)):
            solution = Solution(
                id=len(self.solutions),
                description=desc,
                scores=scores,
                scores_array=np.array(list(scores.values()))
            )

            solution.metadata = {
                "word_count": len(desc.split()),
                "complexity": self._calculate_complexity(desc),
                "timestamp": datetime.now().isoformat()
            }

            self.solutions.append(solution)

            # Animação de progresso
            if Config.SHOW_ANIMATIONS:
                print(f"  [{i+1}/{len(descriptions)}] {desc[:50]}...")
                time.sleep(0.1)

        elapsed = time.time() - start_time
        print(f"\n⚡ {len(descriptions)} soluções processadas em {elapsed:.3f}s")
        print(f"   Velocidade: {len(descriptions)/elapsed:.1f} soluções/s")

    def _compile_patterns(self) -> Dict:
        """Compila padrões regex uma única vez"""
        if self._patterns is None:
            keyword_weights = {
                "fluência": ["usar", "criar", "fazer", "aplicar", "desenvolver"],
                "originalidade": ["improvisada", "arte", "móbile", "único", "novo"],
                "flexibilidade": ["várias", "diferentes", "alternativo", "versátil"],
                "elaboração": ["precisos", "calibrar", "detalhado", "complexo"],
                "adequação": ["simples", "prático", "útil", "eficiente", "fácil"],
                "impacto": ["resolver", "transformar", "melhorar", "revolucionar"]
            }

            self._patterns = {}
            for metric, words in keyword_weights.items():
                self._patterns[metric] = re.compile('|'.join(words), re.IGNORECASE)

        return self._patterns

    def _analyze_solution_algorithmic(self, description: str) -> Dict[str, float]:
        """Análise otimizada com regex pré-compilado"""
        patterns = self._compile_patterns()
        scores = {}

        base_scores = {
            "fluência": 0.5, "originalidade": 0.4, "flexibilidade": 0.45,
            "elaboração": 0.5, "adequação": 0.55, "impacto": 0.4
        }

        boosts = {
            "fluência": 0.1, "originalidade": 0.15, "flexibilidade": 0.12,
            "elaboração": 0.1, "adequação": 0.08, "impacto": 0.13
        }

        for metric, pattern in patterns.items():
            matches = len(pattern.findall(description))
            score = base_scores[metric] + (matches * boosts[metric])
            scores[metric] = min(1.0, score + np.random.uniform(-0.05, 0.05))

        return scores

    def _calculate_complexity(self, description: str) -> float:
        """Calcula complexidade usando vetorização"""
        technical_terms = ["circuito", "calibrar", "precisos", "elétrico"]
        action_verbs = ["usar", "criar", "fazer", "estabilizar"]

        desc_lower = description.lower()
        factors = np.array([
            len(description) / 200,
            sum(1 for term in technical_terms if term in desc_lower) / 4,
            sum(1 for verb in action_verbs if verb in desc_lower) / 4
        ])

        return np.clip(factors.mean(), 0, 1)

    def add_agent(self, name: str, profile: str, emoji: str,
                  preferences: Dict[CreativityMetric, MetricPreference]) -> None:
        """Adiciona agente com visualização completa"""
        agent_id = len(self.agents)

        numeric_preferences = {
            metric: pref.value for metric, pref in preferences.items()
        }

        preferences_array = np.array(list(numeric_preferences.values()))

        agent = Agent(
            id=agent_id,
            name=name,
            profile=profile,
            emoji=emoji,
            metric_preferences=numeric_preferences,
            preferences_array=preferences_array,
            original_preferences=preferences
        )

        self.agents.append(agent)

        # Invalidar cache de matrizes
        self._precomputed.clear()

        # Visualização rica do agente
        print(f"\n{'═' * 80}")
        print(f"{emoji} AGENTE ADICIONADO: {name}")
        print(f"📋 Perfil: {profile}")
        print(f"🎯 Preferências:")

        for metric, pref in preferences.items():
            bar = pref.visual + '░' * (5 - len(pref.visual)//2)
            print(f"   {metric.icon} {metric.value:<15} {bar} ({pref.value:.1f}) - {metric.description}")

        print(f"\n🧠 Traços de Personalidade:")
        for trait, value in agent.personality_traits.items():
            bar = '█' * int(value * 10) + '░' * (10 - int(value * 10))
            print(f"   {trait:<15} {bar} {value:.2f}")

        # Se LLM ativado, pré-calcula avaliações
        if self.use_llm and self.solutions:
            print(f"\n🧠 Processando avaliações com LLM...")
            self._precompute_llm_evaluations(agent)

        # Visualização gráfica do agente
        if Config.SHOW_ANIMATIONS:
            agent.visualize_preferences()

    def _precompute_llm_evaluations(self, agent: Agent) -> None:
        """Pré-calcula avaliações LLM"""
        for solution in self.solutions:
            if str(agent.id) not in solution.llm_evaluations:
                evaluation = self.llm_simulator.evaluate_solution(
                    solution.description,
                    agent.profile
                )
                solution.llm_evaluations[str(agent.id)] = evaluation

    def _get_initial_rankings_vectorized(self) -> np.ndarray:
        """Rankings iniciais usando operações matriciais vetorizadas"""
        n_agents = len(self.agents)
        n_solutions = len(self.solutions)

        pref_matrix = np.array([agent.preferences_array for agent in self.agents])
        sol_matrix = np.array([sol.scores_array for sol in self.solutions])

        score_matrix = pref_matrix @ sol_matrix.T

        pref_sums = pref_matrix.sum(axis=1, keepdims=True)
        pref_sums[pref_sums == 0] = 1
        score_matrix = score_matrix / pref_sums

        rankings = np.argsort(-score_matrix, axis=1)

        self.stats["vectorized_ops"] += 1

        return rankings

    def run_hybrid_consensus(self, method: ConsensusMethod = ConsensusMethod.HYBRID,
                           rounds: int = 3) -> List[int]:
        """Executa consenso híbrido com visualizações completas"""
        start_time = time.time()

        print(f"\n{'╔' + '═' * 78 + '╗'}")
        print(f"║{' ' * 15}🚀 INICIANDO CONSENSO HÍBRIDO 🚀{' ' * 28}║")
        print(f"║{' ' * 78}║")
        print(f"║  Método: {method._value_[2]:<40} {method._value_[1]}{' ' * 24}║")
        print(f"║  Rodadas: {rounds:<40}{' ' * 27}║")
        print(f"║  Agentes: {len(self.agents):<40}{' ' * 27}║")
        print(f"║  Soluções: {len(self.solutions):<40}{' ' * 26}║")
        print(f"╚{'═' * 78}╝\n")

        # Visualização inicial
        if Config.SHOW_ANIMATIONS:
            print("📊 Gerando visualizações iniciais...")
            self.visualizer.show_solution_comparison(self.solutions, self.agents)

        # Rankings iniciais
        agent_rankings = self._get_initial_rankings()

        # Processo de negociação
        for round_num in range(rounds):
            print(f"\n{'━' * 80}")
            print(f"🔄 RODADA {round_num + 1}/{rounds}")
            print(f"{'━' * 80}\n")

            # Consenso da rodada
            if method == ConsensusMethod.HYBRID:
                consensus = self._hybrid_consensus_round(agent_rankings, round_num)
            elif method == ConsensusMethod.LLM_MEDIATED and self.use_llm:
                consensus = self._llm_mediated_consensus(agent_rankings, round_num)
            else:
                consensus = self._algorithmic_consensus(agent_rankings, method)

            # Calcula satisfações
            satisfactions = self._calculate_satisfactions(consensus, agent_rankings)

            # Atualiza histórico
            for i, agent in enumerate(self.agents):
                agent.satisfaction_curve.append(satisfactions[i])

            # Visualização da rodada
            self._display_round_results(round_num + 1, consensus, satisfactions)

            # Negociação adaptativa
            if round_num < rounds - 1:
                agent_rankings = self._adaptive_negotiation(
                    agent_rankings, consensus, satisfactions, round_num
                )

            # Salva histórico
            self.consensus_history.append({
                'round': round_num + 1,
                'consensus': consensus.copy(),
                'satisfactions': satisfactions.copy(),
                'method': method._value_[0]
            })

            self.stats["rounds_completed"] += 1

        # Estatísticas finais
        self.stats["total_time"] = time.time() - start_time

        # Visualizações finais
        self._show_final_results(consensus)

        return consensus

    def _get_initial_rankings(self) -> List[List[int]]:
        """Obtém rankings iniciais com visualização detalhada"""
        print("📊 Calculando rankings iniciais dos agentes...\n")

        if Config.USE_VECTORIZATION:
            rankings_np = self._get_initial_rankings_vectorized()
            agent_rankings = rankings_np.tolist()
        else:
            agent_rankings = []
            for agent in self.agents:
                scores = []
                for solution in self.solutions:
                    if self.use_llm and str(agent.id) in solution.llm_evaluations:
                        score = solution.get_combined_score(str(agent.id), True)
                    else:
                        weights = agent.preferences_array
                        solution_scores = np.array([
                            solution.scores.get(m.value, 0.5)
                            for m in CreativityMetric
                        ])
                        score = np.dot(weights, solution_scores) / (weights.sum() if weights.sum() != 0 else 1)
                    scores.append(score)

                ranking = np.argsort(scores)[::-1].tolist()
                agent_rankings.append(ranking)

        # Exibe rankings
        for agent, ranking in zip(self.agents, agent_rankings):
            print(f"{agent.emoji} {agent.name}:")
            medals = ["🥇", "🥈", "🥉"]
            for pos, sol_idx in enumerate(ranking[:3]):
                medal = medals[pos] if pos < 3 else f"{pos+1}º"
                print(f"  {medal} Solução {sol_idx+1}: {self.solutions[sol_idx].description[:40]}...")
            if len(ranking) > 3:
                print(f"  ... e {len(ranking) - 3} outras soluções")
            print()

        return agent_rankings

    def _hybrid_consensus_round(self, rankings: List[List[int]], round_num: int) -> List[int]:
        """Rodada de consenso híbrido com decisão inteligente"""
        n_solutions = len(self.solutions)

        divergence = self._calculate_divergence(rankings)

        print(f"📊 Análise de Divergência: {divergence:.2%}")

        if divergence > 0.7:
            print("   🔴 Divergência ALTA - conflito significativo")
        elif divergence > 0.4:
            print("   🟡 Divergência MÉDIA - diferenças moderadas")
        else:
            print("   🟢 Divergência BAIXA - boa convergência")

        if divergence > 0.6 and self.use_llm and self.stats.get("llm_calls", 0) < Config.MAX_LLM_CALLS:
            print("🧠 Usando mediação LLM para alta divergência")
            mediation = self.llm_simulator.mediate_negotiation(
                self.agents, self.solutions, round_num
            )
            print(f"💬 Insight LLM: {mediation}")
            return self._nash_bargaining_sa(rankings, n_solutions)
        else:
            print("⚡ Usando consenso algorítmico eficiente")
            return self._borda_count_vectorized(rankings, n_solutions)

    def _llm_mediated_consensus(self, rankings: List[List[int]], round_num: int) -> List[int]:
        """Consenso mediado por LLM"""
        if not self.use_llm:
            return self._algorithmic_consensus(rankings, ConsensusMethod.NASH)

        print("🧠 Iniciando mediação por LLM...")
        mediation = self.llm_simulator.mediate_negotiation(
            self.agents, self.solutions, round_num
        )
        print(f"💬 Mediação: {mediation}")

        return self._nash_bargaining_sa(rankings, len(self.solutions))

    def _algorithmic_consensus(self, rankings: List[List[int]],
                             method: ConsensusMethod) -> List[int]:
        """Consenso algorítmico com métodos otimizados"""
        n = len(self.solutions)

        method_map = {
            ConsensusMethod.BORDA: self._borda_count_vectorized,
            ConsensusMethod.NASH: lambda r, n: self._nash_bargaining_sa(r, n),
            ConsensusMethod.CONDORCET: self._condorcet_winner,
            ConsensusMethod.SHAPLEY: self._shapley_value
        }

        func = method_map.get(method, self._borda_count_vectorized)
        return func(rankings, n)

    def _borda_count_vectorized(self, rankings: List[List[int]], n: int) -> List[int]:
        """Borda Count vetorizado"""
        scores = np.zeros(n)

        for ranking in rankings:
            for pos, sol_idx in enumerate(ranking):
                scores[sol_idx] += (n - pos - 1)

        self.stats["vectorized_ops"] += 1
        return np.argsort(scores)[::-1].tolist()

    def _nash_bargaining_sa(self, rankings: List[List[int]], n: int,
                           temperature: float = 1.0, cooling_rate: float = 0.95) -> List[int]:
        """Nash Bargaining com Simulated Annealing"""
        current = self._borda_count_vectorized(rankings, n)
        best = current.copy()
        best_product = self._calculate_nash_product(current, rankings)

        temp = temperature

        while temp > 0.01:
            i, j = np.random.choice(n, 2, replace=False)
            neighbor = current.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

            neighbor_product = self._calculate_nash_product(neighbor, rankings)
            delta = neighbor_product - best_product

            if delta > 0 or np.random.random() < np.exp(delta / temp):
                current = neighbor
                if neighbor_product > best_product:
                    best = neighbor
                    best_product = neighbor_product

            temp *= cooling_rate

        return best

    def _condorcet_winner(self, rankings: List[List[int]], n: int) -> List[int]:
        """Encontra vencedor de Condorcet"""
        pairwise = np.zeros((n, n))

        for ranking in rankings:
            for i, sol_i in enumerate(ranking):
                for sol_j in ranking[i+1:]:
                    pairwise[sol_i, sol_j] += 1

        copeland = np.sum(pairwise > len(rankings)/2, axis=1) - \
                  np.sum(pairwise < len(rankings)/2, axis=1)

        return np.argsort(copeland)[::-1].tolist()

    def _shapley_value(self, rankings: List[List[int]], n: int) -> List[int]:
        """Calcula valores de Shapley"""
        values = np.zeros(n)

        for sol_idx in range(n):
            for ranking in rankings:
                if sol_idx in ranking:
                    pos = ranking.index(sol_idx)
                    values[sol_idx] += (n - pos) / n

        values /= len(rankings)
        return np.argsort(values)[::-1].tolist()

    def _calculate_nash_product(self, consensus: List[int],
                               agent_rankings: List[List[int]]) -> float:
        """Calcula produto de Nash"""
        satisfactions = self._calculate_satisfactions(consensus, agent_rankings)
        return np.prod(np.maximum(satisfactions, 0.01))

    def _kendall_tau_fast(self, r1: List[int], r2: List[int]) -> float:
        """Kendall tau otimizado"""
        return kendall_tau_fast(r1, r2)

    def _calculate_divergence(self, rankings: List[List[int]]) -> float:
        """Calcula divergência entre rankings"""
        if len(rankings) < 2:
            return 0.0

        correlations = []
        for i in range(len(rankings)):
            for j in range(i+1, len(rankings)):
                tau = self._kendall_tau_fast(rankings[i], rankings[j])
                correlations.append((tau + 1) / 2)

        return 1 - np.mean(correlations)

    def _calculate_satisfactions(self, consensus: List[int],
                                rankings: List[List[int]]) -> List[float]:
        """Calcula satisfação de cada agente"""
        satisfactions = []

        for ranking in rankings:
            tau = self._kendall_tau_fast(consensus, ranking)
            satisfaction = (tau + 1) / 2
            satisfactions.append(satisfaction)

        return satisfactions

    def _adaptive_negotiation(self, current_rankings: List[List[int]],
                            consensus: List[int], satisfactions: List[float],
                            round_num: int) -> List[List[int]]:
        """Negociação adaptativa com personalidade"""
        print("\n💬 Fase de Negociação Adaptativa")
        print("─" * 40)

        new_rankings = []

        for i, (agent, ranking, satisfaction) in enumerate(
            zip(self.agents, current_rankings, satisfactions)
        ):
            flexibility = agent.personality_traits["flexibility"]

            if satisfaction < 0.4:
                adjustment = 0.3 * flexibility
                status = "🔴 Ajuste forte"
            elif satisfaction < 0.6:
                adjustment = 0.15 * flexibility
                status = "🟡 Ajuste moderado"
            else:
                adjustment = 0.05 * flexibility
                status = "🟢 Ajuste mínimo"

            print(f"  {agent.emoji} {agent.name:<12} {status} ({adjustment:.1%})")

            new_ranking = self._adjust_ranking_smart(ranking, consensus, adjustment)
            new_rankings.append(new_ranking)

        return new_rankings

    def _adjust_ranking_smart(self, agent_ranking: List[int],
                             consensus: List[int], factor: float) -> List[int]:
        """Ajuste inteligente de ranking"""
        n = len(agent_ranking)

        agent_scores = np.array([n - i for i in range(n)])
        consensus_scores = np.zeros(n)

        for i, sol_idx in enumerate(consensus):
            if sol_idx in agent_ranking:
                idx = agent_ranking.index(sol_idx)
                consensus_scores[idx] = n - i

        hybrid = (1 - factor) * agent_scores + factor * consensus_scores
        hybrid += np.random.uniform(-0.01, 0.01, n)

        sorted_indices = np.argsort(hybrid)[::-1]
        return [agent_ranking[i] for i in sorted_indices]

    def _display_round_results(self, round_num: int, consensus: List[int],
                              satisfactions: List[float]) -> None:
        """Exibe resultados detalhados da rodada"""
        print(f"\n📊 RESULTADOS DA RODADA {round_num}")
        print("─" * 60)

        # Top 3 do consenso
        print("\n🏆 Consenso Parcial:")
        medals = ["🥇", "🥈", "🥉"]
        for pos, sol_idx in enumerate(consensus[:3]):
            medal = medals[pos] if pos < 3 else f"{pos+1}º"
            solution = self.solutions[sol_idx]
            score_avg = np.mean(list(solution.scores.values()))
            print(f"  {medal} Solução {sol_idx+1} (Score: {score_avg:.2f})")
            print(f"     └─ {solution.description[:50]}...")

        # Satisfação dos agentes
        print("\n😊 Níveis de Satisfação:")
        for agent, sat in zip(self.agents, satisfactions):
            bar_length = int(sat * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            color = "🟢" if sat > 0.7 else "🟡" if sat > 0.4 else "🔴"
            print(f"  {agent.emoji} {agent.name:<12} {bar} {sat:.1%} {color}")

        # Estatísticas
        avg_sat = np.mean(satisfactions)
        std_sat = np.std(satisfactions)
        min_sat = np.min(satisfactions)
        max_sat = np.max(satisfactions)

        print(f"\n📈 Estatísticas:")
        print(f"  Média: {avg_sat:.1%} | Desvio: {std_sat:.1%}")
        print(f"  Mín: {min_sat:.1%} | Máx: {max_sat:.1%}")

    def _show_final_results(self, final_consensus: List[int]) -> None:
        """Exibe resultados finais com visualizações completas"""
        print(f"\n{'═' * 80}")
        print(f"{'🎊 CONSENSO FINAL ALCANÇADO 🎊':^80}")
        print(f"{'═' * 80}\n")

        # Ranking final completo
        print("🏁 RANKING FINAL CONSENSUADO:\n")

        for pos, sol_idx in enumerate(final_consensus):
            solution = self.solutions[sol_idx]

            # Visual diferenciado por posição
            if pos == 0:
                icon = "🥇"
                box = "█" * 80
                print(f"\n{box}")
            elif pos == 1:
                icon = "🥈"
                box = "▓" * 80
                print(f"\n{box}")
            elif pos == 2:
                icon = "🥉"
                box = "▒" * 80
                print(f"\n{box}")
            else:
                icon = f"{pos+1}º"
                box = "░" * 80
                print(f"\n{box}")

            print(f"{icon} POSIÇÃO {pos + 1}")
            print(f"📝 {solution.description}")

            # Detalhamento de scores
            print(f"\n📊 Análise Detalhada:")
            for metric in CreativityMetric:
                score = solution.scores.get(metric.value, 0)
                bar = '█' * int(score * 10) + '░' * (10 - int(score * 10))
                print(f"   {metric.icon} {metric.value:<15} {bar} {score:.2f}")

            avg_score = np.mean(list(solution.scores.values()))
            print(f"\n   📈 Score médio: {avg_score:.2%}")

            if self.use_llm and solution.llm_evaluations:
                llm_scores = [e["profile_alignment"] for e in solution.llm_evaluations.values()]
                print(f"   🧠 Alinhamento LLM: {np.mean(llm_scores):.2%}")

        # Estatísticas do processo
        print(f"\n{'─' * 80}")
        print("📊 ESTATÍSTICAS DO PROCESSO:")
        print(f"  ⏱️  Tempo total: {self.stats['total_time']:.2f}s")
        print(f"  🔄 Rodadas completadas: {self.stats['rounds_completed']}")
        print(f"  📊 Operações vetorizadas: {self.stats.get('vectorized_ops', 0)}")
        print(f"  🚀 Operações paralelas: {self.stats.get('parallel_ops', 0)}")

        if self.use_llm:
            print(f"  🧠 Chamadas LLM: {self.stats.get('llm_calls', 0)}")
            print(f"  💾 Cache hits: {self.stats.get('cache_hits', 0)}")
            if self.cache.hit_rate > 0:
                print(f"  ⚡ Taxa de cache: {self.cache.hit_rate:.1%}")

        # Throughput
        if self.stats['total_time'] > 0:
            throughput = self.stats['rounds_completed'] / self.stats['total_time']
            print(f"  ⚡ Throughput: {throughput:.2f} rodadas/s")

        # Visualizações finais interativas
        if Config.SHOW_ANIMATIONS and VISUALIZATION_AVAILABLE:
            print("\n📈 Gerando visualizações finais interativas...")

            # Evolução do consenso
            self.visualizer.show_consensus_evolution(self.consensus_history)

            # Dinâmica de negociação
            self.visualizer.show_negotiation_dynamics(self.agents)

            # Comparação final
            self.visualizer.show_solution_comparison(self.solutions, self.agents)

            # Breakdown das top soluções
            self.visualizer.show_final_consensus_breakdown(
                self.solutions, final_consensus, self.agents
            )

    def generate_detailed_report(self) -> str:
        """Gera relatório HTML detalhado e estilizado"""
        html = f"""
        <html>
        <head>
            <title>Relatório - Framework Híbrido Multiagente v5.0</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    margin: 40px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: #333;
                }}
                .container {{
                    background: white;
                    border-radius: 15px;
                    padding: 30px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    font-size: 2.5em;
                    margin-bottom: 30px;
                }}
                h2 {{
                    color: #34495e;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                    margin-top: 30px;
                }}
                .stat-box {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 10px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                .solution {{
                    background: #fff;
                    padding: 20px;
                    margin: 20px 0;
                    border-left: 5px solid #3498db;
                    border-radius: 5px;
                    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                    transition: transform 0.3s;
                }}
                .solution:hover {{
                    transform: translateX(5px);
                }}
                .gold {{ border-left-color: #FFD700; background: #FFFACD; }}
                .silver {{ border-left-color: #C0C0C0; background: #F5F5F5; }}
                .bronze {{ border-left-color: #CD7F32; background: #FFF8DC; }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    font-weight: bold;
                }}
                tr:nth-child(even) {{ background: #f9f9f9; }}
                tr:hover {{ background: #f5f5f5; }}
                .metric-bar {{
                    display: inline-block;
                    width: 100px;
                    height: 20px;
                    background: #e0e0e0;
                    border-radius: 10px;
                    overflow: hidden;
                    margin-left: 10px;
                }}
                .metric-fill {{
                    height: 100%;
                    background: linear-gradient(90deg, #4CAF50, #8BC34A);
                    border-radius: 10px;
                }}
                .agent-card {{
                    display: inline-block;
                    background: white;
                    border: 2px solid #3498db;
                    border-radius: 10px;
                    padding: 15px;
                    margin: 10px;
                    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                }}
                .emoji {{ font-size: 2em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Relatório do Framework Híbrido Multiagente v5.0</h1>

                <div class="stat-box">
                    <h2 style="color: white; border: none;">⚙️ Configurações da Execução</h2>
                    <p><strong>Modo:</strong> {Config.SIMULATION_MODE}</p>
                    <p><strong>LLM:</strong> {'✅ Habilitado' if self.use_llm else '❌ Desabilitado'}</p>
                    <p><strong>Agentes:</strong> {len(self.agents)}</p>
                    <p><strong>Soluções:</strong> {len(self.solutions)}</p>
                    <p><strong>Rodadas:</strong> {self.stats['rounds_completed']}</p>
                    <p><strong>Tempo Total:</strong> {self.stats['total_time']:.2f}s</p>
                </div>

                <h2>👥 Agentes Participantes</h2>
                <div style="text-align: center;">
        """

        for agent in self.agents:
            html += f"""
                <div class="agent-card">
                    <div class="emoji">{agent.emoji}</div>
                    <h4>{agent.name}</h4>
                    <p style="font-size: 0.9em; color: #666;">{agent.profile[:50]}...</p>
                </div>
            """

        html += """
                </div>

                <h2>🏆 Ranking Final Consensuado</h2>
        """

        if self.consensus_history:
            final = self.consensus_history[-1]['consensus']
            for pos, sol_idx in enumerate(final):
                sol = self.solutions[sol_idx]
                avg_score = np.mean(list(sol.scores.values()))

                medal_class = ""
                medal_icon = f"{pos+1}º"
                if pos == 0:
                    medal_class = "gold"
                    medal_icon = "🥇"
                elif pos == 1:
                    medal_class = "silver"
                    medal_icon = "🥈"
                elif pos == 2:
                    medal_class = "bronze"
                    medal_icon = "🥉"

                html += f"""
                <div class="solution {medal_class}">
                    <h3>{medal_icon} Posição {pos+1}</h3>
                    <p><strong>Descrição:</strong> {sol.description}</p>
                    <p><strong>Score Médio:</strong> {avg_score:.2%}</p>
                    <div style="margin-top: 10px;">
                """

                for metric in CreativityMetric:
                    score = sol.scores.get(metric.value, 0)
                    html += f"""
                        <div style="margin: 5px 0;">
                            {metric.icon} {metric.value}:
                            <div class="metric-bar">
                                <div class="metric-fill" style="width: {score*100}%;"></div>
                            </div>
                            <span style="margin-left: 10px;">{score:.2f}</span>
                        </div>
                    """

                html += """
                    </div>
                </div>
                """

        html += f"""
                <h2>📈 Estatísticas de Performance</h2>
                <table>
                    <tr><th>Métrica</th><th>Valor</th></tr>
                    <tr><td>⏱️ Tempo Total</td><td>{self.stats['total_time']:.3f}s</td></tr>
                    <tr><td>🔄 Rodadas</td><td>{self.stats['rounds_completed']}</td></tr>
                    <tr><td>📊 Operações Vetorizadas</td><td>{self.stats.get('vectorized_ops', 0)}</td></tr>
                    <tr><td>🚀 Operações Paralelas</td><td>{self.stats.get('parallel_ops', 0)}</td></tr>
        """

        if self.use_llm:
            html += f"""
                    <tr><td>🧠 Chamadas LLM</td><td>{self.stats.get('llm_calls', 0)}</td></tr>
                    <tr><td>💾 Cache Hits</td><td>{self.stats.get('cache_hits', 0)}</td></tr>
            """

        if self.stats['total_time'] > 0:
            throughput = self.stats['rounds_completed'] / self.stats['total_time']
            html += f"""
                    <tr><td>⚡ Throughput</td><td>{throughput:.2f} rodadas/s</td></tr>
            """

        html += """
                </table>

                <h2>📊 Evolução da Satisfação</h2>
                <p>A satisfação média dos agentes evoluiu ao longo das rodadas:</p>
                <table>
                    <tr><th>Rodada</th><th>Satisfação Média</th><th>Mínimo</th><th>Máximo</th></tr>
        """

        for history in self.consensus_history:
            avg_sat = np.mean(history['satisfactions'])
            min_sat = np.min(history['satisfactions'])
            max_sat = np.max(history['satisfactions'])
            html += f"""
                    <tr>
                        <td>{history['round']}</td>
                        <td>{avg_sat:.1%}</td>
                        <td>{min_sat:.1%}</td>
                        <td>{max_sat:.1%}</td>
                    </tr>
            """

        html += f"""
                </table>

                <div style="text-align: center; margin-top: 40px; padding: 20px; background: #f0f0f0; border-radius: 10px;">
                    <p style="color: #666; font-size: 0.9em;">
                        Relatório gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}<br>
                        Framework Híbrido de Decisão Multiagente v5.0<br>
                        Performance Edition com Visualizações Completas
                    </p>
                </div>
            </div>
        </body>
        </html>
        """

        return html