# Projeto Bootcamp de Ciência de Dados e IA: Análise Preditiva de Defeitos em Chapas de Aço

Este projeto foi desenvolvido como parte do Bootcamp de Ciência de Dados e IA, com o objetivo de criar um modelo de machine learning para prever diferentes tipos de defeitos em chapas de aço inoxidável e apresentar os resultados e insights através de um dashboard interativo.

## Estrutura do Projeto

O projeto está organizado da seguinte forma:

- `bootcamp_train.csv`: Conjunto de dados de treino original.
- `bootcamp_test.csv`: Conjunto de dados de teste original (para submissão final, se aplicável).
- `bootcamp_submission_example.csv`: Exemplo do formato de ficheiro para submissão.
- `eda_script_01.py`: Script inicial para carregamento e primeira análise dos dados.
- `eda_script_02.py`: Script para padronização de colunas de falha e tipo de aço, e visualização da distribuição de falhas.
- `eda_script_03.py`: Script para análise de outliers, distribuições de variáveis numéricas e correlações.
- `eda_script_04_preprocessing.py`: Script para pré-processamento dos dados (tratamento de valores em falta, scaling).
- `bootcamp_train_processed.csv`: Conjunto de dados de treino após pré-processamento.
- `model_training_evaluation_01.py`: Script para treino e avaliação inicial de modelos (Random Forest e Regressão Logística).
- `model_evaluation_results.csv`: Resultados da avaliação dos modelos.
- `feature_importance_analysis.py`: Script para extrair e visualizar a importância das features do modelo Random Forest.
- `feature_importances_random_forest.png`: Gráfico com a importância das features.
- `generate_predictions_01.py`: Script para treinar o modelo final (Random Forest) com todos os dados de treino, pré-processar o conjunto de teste e gerar o ficheiro de submissão.
- `submission_random_forest.csv`: Ficheiro de previsões gerado para o conjunto de teste.
- `dashboard_app_v4.py`: Script da aplicação Streamlit para o dashboard interativo.
- `insights_apresentacao.md`: Esboço e insights para a apresentação do projeto.
- `requirements.txt`: Lista de dependências Python para o projeto.
- `todo.md`: Lista de tarefas e progresso do projeto.

## Como Executar

### Pré-requisitos

- Python 3.9 ou superior
- `pip` (gestor de pacotes Python)

### 1. Configurar o Ambiente

Clone o repositório (ou copie os ficheiros para uma pasta local) e navegue até à pasta do projeto.
Crie um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate  # No Linux/macOS
# .venv\Scripts\activate    # No Windows
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

### 2. Executar os Scripts de Análise e Modelagem (Opcional, para reproduzir os passos)

Os scripts foram nomeados sequencialmente (de `eda_script_01.py` a `generate_predictions_01.py`). Executá-los pela ordem para seguir o processo de desenvolvimento.

Exemplo:

```bash
python eda_script_01.py
python eda_script_02.py
# ... e assim por diante
```

### 3. Executar o Dashboard Interativo

Para visualizar o dashboard Streamlit:

```bash
streamlit run dashboard_app_v4.py
```

O dashboard deverá abrir automaticamente no seu navegador web.
