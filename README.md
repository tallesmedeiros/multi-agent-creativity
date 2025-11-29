# Framework Híbrido de Criatividade Multiagente

Este repositório demonstra um pipeline completo para avaliar ideias criativas usando agentes artificiais com perfis distintos. O sistema combina heurísticas algorítmicas, negociação multiagente e mediação por LLM (simulada) para chegar a consensos sobre soluções criativas.

## Visão geral rápida
- **Execução guiada:** `main.py` inicia uma experiência interativa, permite escolher entre modo puramente algorítmico ou híbrido com LLM simulado e executa todas as etapas de coleta, avaliação e consenso.
- **Framework central:** `CompleteHybridFramework` (`framework.py`) encapsula análise das soluções, configuração dos agentes, cálculo de rankings e rodadas de consenso híbrido com visualizações opcionais.
- **Modelos de domínio:** as classes `Agent` e `Solution` (`models/agent.py`, `models/solution.py`) armazenam preferências, traços de personalidade, metadados e históricos de consenso para cada entidade.
- **Parâmetros e otimizações:** `config.py` define toggles de desempenho (paralelização, vetorização NumPy, JIT com Numba, cache) e opções de simulação.
- **Enums de negócio:** `enums.py` centraliza métricas de criatividade, níveis de preferência e métodos de consenso disponíveis.
- **Suporte analítico:** `analysis.py` traz um Kendall Tau otimizado para medir divergência entre rankings; `llm_simulator.py` (simulação de LLM) e `visualization.py` fornecem mediação textual e gráficos interativos.

## Estrutura de diretórios
```
├── main.py                   # Ponto de entrada interativo
├── framework.py              # Orquestração de análise, negociação e consenso
├── config.py                 # Flags de execução e otimizações
├── enums.py                  # Métricas de criatividade e métodos de consenso
├── analysis.py               # Estatísticas de correlação (Kendall Tau)
├── llm_simulator.py          # Mediador LLM simulado para negociação
├── visualization.py          # Gráficos e relatórios HTML
├── models/
│   ├── agent.py              # Representação de agentes e suas preferências
│   └── solution.py           # Representação de ideias avaliadas
└── cache.py, framework auxiliares
```

## Fluxo de execução
1. **Carregamento de configurações:** `Config` aplica preferências de modo (LLM ou não), paralelização (`multiprocessing.Pool`), vetorização NumPy e cache de respostas.
2. **Cadastro de soluções:** `CompleteHybridFramework.add_solutions_batch` analisa descrições em paralelo, gera *scores* heurísticos por métrica e metadados (complexidade, contagem de palavras, timestamp).
3. **Criação de agentes:** `add_agent` converte preferências qualitativas em vetores NumPy e inicializa traços de personalidade; com LLM ativo, pré-calcula avaliações alinhadas ao perfil.
4. **Rankeamento inicial:** `_get_initial_rankings_vectorized` calcula, via produto matricial, a adequação de cada solução aos pesos de cada agente.
5. **Rodadas de consenso:** `run_hybrid_consensus` combina métodos algorítmicos e mediação LLM para negociar rankings ao longo de várias rodadas, registrando histórico e satisfação dos agentes.
6. **Relatórios e visualizações:** `generate_detailed_report` produz um HTML estilizado com estatísticas, rankings e comparações; `InteractiveVisualizer` (em `visualization.py`) oferece gráficos interativos quando habilitado.

## Métricas de criatividade
O sistema avalia cada solução segundo seis dimensões inspiradas em literatura de criatividade:
- **Fluência (💡):** volume de ideias e ações descritas.
- **Originalidade (🎨):** ineditismo e novidade da proposta.
- **Flexibilidade (🔄):** variedade de abordagens ou contextos de uso.
- **Elaboração (🔬):** nível de detalhe e refinamento técnico.
- **Adequação (✅):** viabilidade prática e aplicabilidade.
- **Impacto (💥):** potencial transformador ou de geração de valor.

Cada agente atribui pesos (de *muito baixa* a *muito alta*) a essas métricas usando `MetricPreference`, refletindo prioridades distintas durante a negociação.

## Teoria de decisão multiagente aplicada
A lógica de consenso combina vários paradigmas de tomada de decisão coletiva:
- **Soma ponderada / produto de Nash:** as preferências são vetorizadas e multiplicadas pelos *scores* das soluções; o produto de Nash é otimizado com *simulated annealing* para equilibrar utilidade entre agentes.
- **Métodos de votação social:** contagem de Borda, vencedor de Condorcet e valor de Shapley permitem comparar rankings individuais e construir uma ordem agregada.
- **Mediação adaptativa por LLM:** em cenários de alta divergência, o `OptimizedLLMSimulator` sugere compromissos textuais para guiar a negociação, combinando julgamento qualitativo com heurísticas algorítmicas.
- **Análise de divergência:** o coeficiente Kendall Tau mede proximidade entre rankings, orientando quando acionar mediação LLM ou ajustes de pesos.

Esse arranjo híbrido ilustra como agentes com preferências heterogêneas podem iterar entre algoritmos de escolha social e insights linguísticos para avaliar soluções criativas de forma transparente e explicável.

## Como executar
1. Certifique-se de ter Python 3.10+ e as dependências padrão instaladas (`numpy`, `matplotlib`, `seaborn` opcionalmente). Para acelerar o Kendall Tau, instale `numba`.
2. (Opcional) Defina `OPENAI_API_KEY` em `config.py` para substituir o simulador por uma chamada real de LLM.
3. Rode o fluxo interativo:
   ```bash
   python main.py
   ```
   Escolha o modo (algorítmico puro ou híbrido com LLM simulado), acompanhe as rodadas e, ao final, gere um relatório HTML se desejar.

## Extensões possíveis
- **Novas métricas:** adicione membros em `CreativityMetric` e ajuste `_analyze_solution_algorithmic` para incluir novos padrões e *scores*.
- **Novos métodos de consenso:** implemente uma função agregadora e registre em `ConsensusMethod` e `_algorithmic_consensus`.
- **Integração LLM real:** troque `OptimizedLLMSimulator` por chamadas para seu provedor preferido, mantendo a interface `mediate_negotiation`.

Bom experimento! 🚀
