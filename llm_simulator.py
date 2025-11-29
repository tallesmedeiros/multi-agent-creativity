import hashlib
import re
import time
from typing import Dict, List

import numpy as np

from cache import OptimizedCache
from config import Config


class OptimizedLLMSimulator:
    """🤖 Simulador de LLM otimizado com cache e batch processing"""

    def __init__(self):
        self.cache = OptimizedCache(Config.CACHE_SIZE)
        self.call_count = 0
        self.stats = {"llm_calls": 0, "cache_hits": 0}
        self.patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict:
        profiles_keywords = {
            "engenheiro": {
                "positive": ["ferramenta", "precisos", "calibrar", "circuito", "estabilizar"],
                "negative": ["arte", "sons", "decorativo"],
                "boost_metrics": ["adequação", "elaboração"],
            },
            "artista": {
                "positive": ["arte", "criar", "texturas", "móbile", "sons"],
                "negative": ["ferramenta", "precisos"],
                "boost_metrics": ["originalidade", "flexibilidade"],
            },
            "cientista": {
                "positive": ["precisos", "calibrar", "circuito", "elétrico"],
                "negative": ["improvisada", "decorativo"],
                "boost_metrics": ["elaboração", "impacto"],
            },
        }

        compiled = {}
        for profile, keywords in profiles_keywords.items():
            compiled[profile] = {
                "positive": re.compile("|".join(keywords["positive"]), re.IGNORECASE),
                "negative": re.compile("|".join(keywords["negative"]), re.IGNORECASE),
                "boost_metrics": keywords["boost_metrics"],
            }
        return compiled

    def evaluate_solution(self, solution: str, agent_profile: str) -> Dict:
        cache_key = hashlib.md5(f"{solution}_{agent_profile}".encode()).hexdigest()
        cached = self.cache.get(cache_key)

        if cached:
            self.stats["cache_hits"] += 1
            return cached

        result = self._evaluate_single(solution, agent_profile)
        self.cache.put(cache_key, result)
        self.stats["llm_calls"] += 1

        if Config.SHOW_ANIMATIONS:
            time.sleep(0.1)

        return result

    def _evaluate_single(self, solution: str, agent_profile: str) -> Dict:
        self.call_count += 1

        profile_key = None
        for key in self.patterns.keys():
            if key in agent_profile.lower():
                profile_key = key
                break

        if not profile_key:
            profile_key = "engenheiro"

        profile_data = self.patterns[profile_key]

        positive_matches = len(profile_data["positive"].findall(solution))
        negative_matches = len(profile_data["negative"].findall(solution))

        base_score = np.random.uniform(0.4, 0.6)
        adjusted_score = np.clip(base_score + (positive_matches * 0.15) - (negative_matches * 0.1), 0.1, 1.0)

        metrics = ["fluência", "originalidade", "flexibilidade", "elaboração", "adequação", "impacto"]
        boost_mask = np.array([m in profile_data["boost_metrics"] for m in metrics])
        base_scores = np.full(len(metrics), adjusted_score)
        noise = np.random.uniform(-0.1, 0.2, len(metrics))
        noise[~boost_mask] = np.random.uniform(-0.1, 0.1, (~boost_mask).sum())

        final_scores = np.clip(base_scores + noise, 0, 1)
        scores = dict(zip(metrics, final_scores))

        justification = self._generate_justification(solution, profile_key, scores)

        return {
            "scores": scores,
            "reasoning": justification,
            "confidence": np.random.uniform(0.7, 0.95),
            "profile_alignment": adjusted_score,
        }

    def _generate_justification(self, solution: str, profile: str, scores: Dict) -> str:
        justifications = {
            "engenheiro": "Solução prática com foco em funcionalidade e eficiência técnica",
            "artista": "Abordagem criativa valorizando expressão e originalidade estética",
            "cientista": "Método sistemático baseado em princípios científicos fundamentados",
        }
        return justifications.get(profile, "Análise baseada em critérios múltiplos")

    def mediate_negotiation(self, agents: List, solutions: List, round_num: int) -> str:
        self.call_count += 1
        self.stats["llm_calls"] += 1

        mediations = [
            "Os agentes identificaram pontos de convergência nas soluções práticas",
            "Há consenso emergente sobre a importância da viabilidade técnica",
            "As perspectivas criativas estão sendo valorizadas pelo grupo",
            "O debate revelou critérios compartilhados não percebidos inicialmente",
        ]

        return mediations[min(round_num, len(mediations) - 1)]
