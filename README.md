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
O sistema avalia cada solução segundo seis dimensões inspiradas em literatura de criatividade. A coleta inicial ocorre em `CompleteHybridFramework._analyze_solution_algorithmic`, que extrai sinais quantitativos (contagem de tokens, padrões de palavras-chave e normalizações por z-score) e retorna um vetor NumPy com *scores* heurísticos. Esses *scores* são então ponderados pelo vetor de preferência dos agentes (`Agent.preferences`) para gerar os rankings iniciais.

- **Fluência (💡):** estimada a partir da contagem de ideias/ações distintas em uma descrição. A função identifica verbos e conectores de ações, normaliza pela extensão do texto e aumenta a pontuação para descrições que apresentam várias proposições autônomas.
- **Originalidade (🎨):** medida por raridade de palavras-chave e combinações semânticas pouco usuais. O algoritmo utiliza dicionários de referência e detecção de *n-grams* incomuns; soluções com termos menos frequentes recebem *score* maior.
- **Flexibilidade (🔄):** avaliada pelo número de domínios ou contextos presentes. São detectadas categorias (ex.: educação, saúde, indústria) e, quanto maior a diversidade entre elas, maior a pontuação.
- **Elaboração (🔬):** capturada via densidade de detalhes técnicos e presencia de etapas ou parâmetros concretos. Mais números, descrições de processos e especificações técnicas elevam o *score*.
- **Adequação (✅):** estimada pela presença de restrições realistas (custo, tempo, recursos) e alinhamento com metas práticas. A heurística verifica menções a viabilidade, implementação e conformidade com requisitos.
- **Impacto (💥):** calculado por sinais de escala e transformação (ex.: alcance global, efeitos sistêmicos, geração de valor econômico/social). Menções a benefícios amplos ou disruptivos elevam a nota.

Cada agente atribui pesos (de *muito baixa* a *muito alta*) a essas métricas usando `MetricPreference`, refletindo prioridades distintas durante a negociação. Durante o consenso, esses pesos influenciam tanto o ranking algorítmico quanto as sugestões do simulador LLM.

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
