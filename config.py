import warnings
from multiprocessing import cpu_count

try:
    from numba import jit, njit
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    NUMBA_AVAILABLE = False
    jit = None
    njit = None
    print("ℹ️ Numba não disponível - usando implementação Python pura")


class Config:
    """🔧 Configurações centralizadas do sistema otimizado"""

    # Modo de operação
    USE_LLM = False
    OPENAI_API_KEY = ""

    # Parâmetros de simulação
    SIMULATION_MODE = "hybrid"
    CACHE_LLM_RESPONSES = True
    MAX_LLM_CALLS = 10

    # Performance
    USE_PARALLEL = True
    MAX_WORKERS = min(4, cpu_count())
    USE_VECTORIZATION = True
    CACHE_SIZE = 1000

    # Otimizações
    USE_NUMBA = NUMBA_AVAILABLE
    BATCH_SIZE = 32
    USE_SIMULATED_ANNEALING = True

    # Visualização
    SHOW_ANIMATIONS = True
    PLOT_STYLE = "interactive"

    # Debug
    VERBOSE = True
    SHOW_TIMING = True
    PROFILE = False

warnings.filterwarnings("ignore")
