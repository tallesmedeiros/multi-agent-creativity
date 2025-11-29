import numpy as np

from config import Config, NUMBA_AVAILABLE, njit


if NUMBA_AVAILABLE and njit is not None:

    @njit
    def kendall_tau_numba(r1: np.ndarray, r2: np.ndarray) -> float:  # pragma: no cover - numba compiled
        """Kendall tau compilado com Numba - até 10x mais rápido"""
        n = len(r1)
        if n <= 1:
            return 1.0

        concordant = 0
        discordant = 0

        for i in range(n):
            for j in range(i + 1, n):
                if (r1[i] < r1[j] and r2[i] < r2[j]) or (r1[i] > r1[j] and r2[i] > r2[j]):
                    concordant += 1
                else:
                    discordant += 1

        total = n * (n - 1) / 2
        return (concordant - discordant) / total if total > 0 else 1.0
else:

    def kendall_tau_numba(r1, r2):
        return None


def kendall_tau_fast(r1, r2) -> float:
    """Escolhe a implementação mais rápida disponível."""
    if Config.USE_NUMBA and NUMBA_AVAILABLE:
        return kendall_tau_numba(np.array(r1), np.array(r2))  # type: ignore[arg-type]

    n = len(r1)
    if n <= 1:
        return 1.0

    pos2 = {v: i for i, v in enumerate(r2)}

    concordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if r1[i] in pos2 and r1[j] in pos2 and pos2[r1[i]] < pos2[r1[j]]:
                concordant += 1

    total = n * (n - 1) / 2
    return (2 * concordant / total - 1) if total > 0 else 1.0
