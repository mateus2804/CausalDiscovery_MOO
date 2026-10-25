# Differentiable Causal Discovery: Classic vs. Continuous Multiobjective Optimization

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Sobre o Projeto

Este repositório busca reproduzir e analisar métodos clássicos de descoberta causal baseados em otimização contínua diferenciável (score-based), especificamente em suas formulações lineares. O principal objetivo é comparar as abordagens tradicionais — que utilizam multiplicadores de Lagrange ou penalidades para garantir a aciclicidade — com novas formulações estruturadas como **problemas de otimização contínua multiobjetivo**.

Ao invés de tratar a penalidade de aciclicidade $h(W) = 0$ e a esparsidade (ex: penalidade $L_1$) apenas como restrições em um único objetivo escalar, este projeto explora o trade-off ótimo no espaço de Pareto entre o ajuste aos dados (score) e a restrição de Grafo Acíclico Direcionado (DAG).

## 🚀 Métodos Implementados

As seguintes abordagens clássicas (versões lineares) estão implementadas e servem como *baselines*:

*   **NOTEARS** (Zheng et al., 2018): Transforma a busca combinatória por DAGs em otimização contínua usando o traço da matriz exponencial: $h(W) = \text{tr}(e^{W \circ W}) - d = 0$.
*   **GOLEM** (Ng et al., 2020): Relaxa a restrição rígida (hard constraint) do NOTEARS, focando em otimização sem restrições baseada em *likelihood* com penalização suave, reduzindo problemas de otimização em paisagens mal condicionadas.
*   **DAGMA** (Bello et al., 2022): Utiliza uma caracterização baseada em log-determinante para aciclicidade, oferecendo gradientes mais bem comportados e convergência superior às formulações baseadas no traço da exponencial.

### Formulação Multiobjetivo (Moo-Causal)
A expansão do repositório contrapõe os métodos acima a algoritmos de otimização multiobjetivo, avaliando simultaneamente:
1. $f_1(W)$: Erro de reconstrução / Negative Log-Likelihood.
2. $f_2(W)$: Violação de Aciclicidade ($h(W)$).
3. $f_3(W)$: Esparsidade (norma $\Vert{}W\Vert{}_1$).

## ⚙️ Instalação

Clone este repositório e instale as dependências. Recomenda-se o uso de um ambiente virtual (Conda ou venv).

```bash
git clone [https://github.com/seu-usuario/causal-discovery-optimization.git](https://github.com/seu-usuario/causal-discovery-optimization.git)
cd causal-discovery-optimization
pip install -r requirements.txt
