# Modelagem Preditiva — APS

Atividade Prática Supervisionada (APS) da disciplina **Modelagem Preditiva** — Insper, 2º semestre de 2026.
Professor: Paulo C. Marques F.

## Integrantes

- Antonio Irnailson Honorato
- Esthefanny Sousa Farias
- Yuri Henrique
- Eduardo Marcelino

## Sobre a APS

O trabalho tem uma parte teórica sobre os modelos vistos no curso e duas aplicações com dados reais.

### Parte teórica (7,0 pontos)

1. **Regressão logística vs. árvore de classificação** (1,5): treinar os dois modelos em `Q1_training.csv` e comparar os resultados em `Q1_test.csv`.
2. **Bagging de árvores de regressão** (2,0): explicar o bootstrap e o bagging, programar um bagging com `library(tree)` nos dados *California housing* (divisão 50/50) e verificar se o método do subespaço aleatório melhora a performance.
3. **Random Forest** (1,0): explicar a aleatorização dos splits proposta por Breiman, o trade-off entre viés e variância e a relação com o método do subespaço aleatório de Ho.
4. **Erro out-of-bag** (1,5): explicar o cálculo do erro OOB em regressão e mostrar, com `library(ranger)`, que ele se aproxima do erro de teste conforme o número de árvores aumenta (*California housing*, divisão 50/50).
5. **Importância das variáveis** (1,0): explicar como a tabela de importância das variáveis explicativas é construída em uma Random Forest.

### Aplicação 1 — Churn (1,5 ponto)

Prever `Exited` em `churn.csv` (divisão 50/50) com k-NN (k escolhido por uma regra de bolso), Regressão Logística, Árvore de Classificação, Random Forest e Boosting. Comparar a acurácia com corte de 50%, as curvas ROC e as AUCs.

### Aplicação 2 — Preço de carros usados (1,5 ponto)

Prever `price` em `used_cars.csv` (Mercedes, divisão 50/50) com Regressão Linear Múltipla, Árvore de Regressão, Random Forest e Boosting. Comparar o RMSE de teste e fazer gráficos de valores previstos vs. observados.

## Ambiente Python (uv)

O projeto usa o [uv](https://docs.astral.sh/uv/) para gerenciar o Python (3.12+) e as dependências. Elas estão declaradas no `pyproject.toml`, e as versões exatas ficam travadas no `uv.lock`.

### Instalar o uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Configurar o ambiente

Depois de clonar o repositório, rode na raiz do projeto:

```bash
uv sync
```

O comando cria o `.venv` e instala as versões que estão no `uv.lock`. Se o Python 3.12 não estiver instalado, o uv baixa sozinho.

### Uso no dia a dia

| Tarefa | Comando |
| --- | --- |
| Rodar um script | `uv run python script.py` |
| Abrir o Jupyter | `uv run --with jupyter jupyter lab` |
| Adicionar uma biblioteca | `uv add nome-da-biblioteca` |
| Remover uma biblioteca | `uv remove nome-da-biblioteca` |
| Atualizar o ambiente após um `git pull` | `uv sync` |

No VS Code, selecione o interpretador `.venv/bin/python` (ou o kernel `.venv` nos notebooks).

> Sempre commite o `pyproject.toml` e o `uv.lock` depois de adicionar ou remover bibliotecas, para que toda a equipe use o mesmo ambiente. A pasta `.venv` não vai para o repositório (já está no `.gitignore`).

## Estrutura do repositório

```
.
├── data/
│   ├── Q1_training.csv
│   ├── Q1_test.csv
│   ├── california.csv
│   ├── churn.csv
│   └── used_cars.csv
├── pyproject.toml
├── uv.lock
├── requirements.txt
└── README.md
```

## Entrega

- Relatório em Word (Times New Roman 12pt, espaçamento simples), sem código no corpo do texto.
- Código da APS em **R**.
- Envio de um `.zip` (relatório + código) via Blackboard até **24/11/2026, 23h59**.
- Prova Final em **25/11/2026**, com questões sobre o conteúdo da APS.
