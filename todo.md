# Lista de Tarefas do Projeto Bootcamp CDIA

## Fase 1: Análise e Preparação

- [X] **Análise Exploratória de Dados (EDA)**
    - [X] Carregar o conjunto de treino (`bootcamp_train.csv`)
    - [X] Verificar informações básicas do dataset (dimensões, tipos de dados, primeiras linhas)
    - [X] Análise estatística descritiva das variáveis numéricas
    - [X] Análise da distribuição das variáveis alvo (falhas)
    - [X] Identificar e analisar valores em falta (missing values)
    - [X] Identificar e analisar outliers
    - [X] Analisar correlações entre variáveis
    - [X] Criar visualizações para entender as distribuições e relações (histogramas, boxplots, heatmaps)
- [X] **Pré-processamento dos Dados**
    - [X] Tratar valores em falta (imputação ou remoção)
    - [X] Tratar outliers (se necessário) - Decidido adiar para modelagem/discussão
    - [X] Normalizar/Padronizar variáveis numéricas (se necessário para os modelos escolhidos)
    - [X] Codificar variáveis categóricas (se houver, embora pareçam ser todas numéricas ou binárias) - Verificado e realizado
- [ ] **Engenharia de Atributos (Opcional)**
    - [ ] Avaliar a necessidade de criar novas features a partir das existentes

## Fase 2: Modelagem e Avaliação

- [X] **Seleção da Abordagem de Modelagem**
    - [X] Confirmar se a abordagem multirrótulo é a mais adequada com base na EDA
- [X] **Divisão dos Dados**
    - [X] Separar o conjunto de treino em treino e validação (ex: usando `train_test_split`)
- [X] **Seleção e Treino de Modelos**
    - [X] Experimentar diferentes algoritmos de classificação (ex: Regressão Logística, Árvores de Decisão, Random Forest, Gradient Boosting, SVM)
    - [X] Treinar os modelos com os dados de treino
- [X] **Avaliação de Modelos**
    - [X] Definir e calcular métricas de avaliação apropriadas (ex: precisão, recall, F1-score por classe, AUC-ROC, Hamming Loss para multirrótulo)
    - [X] Usar validação cruzada para uma avaliação mais robusta
    - [X] Comparar o desempenho dos diferentes modelos
- [ ] **Otimização do Modelo**
    - [ ] Ajustar hiperparâmetros do(s) melhor(es) modelo(s) (ex: usando GridSearchCV ou RandomizedSearchCV)

## Fase 3: Resultados e Preparação da Apresentação

- [X] **Geração de Previsões**
    - [X] Usar o modelo final treinado para gerar previsões no conjunto de teste (`bootcamp_test.csv`)
    - [X] Formatar as previsões de acordo com `bootcamp_submission_example.csv`
- [-] **Avaliação via API (Opcional, mas recomendado)** - Decidido não utilizar
    - [-] Registar na API (se ainda não feito)
    - [-] Obter o token de autenticação
    - [-] Submeter o ficheiro de previsões à API e analisar as métricas retornadas
- [X] **Interpretação dos Resultados e Insights**
    - [X] Analisar os resultados do modelo final
    - [X] Identificar as features mais importantes para a predição dos defeitos
    - [X] Extrair insights relevantes sobre os tipos de defeitos e a operação
- [X] **Desenvolvimento do Dashboard com Streamlit (Opcional)**
    - [X] Planear o conteúdo e a estrutura do dashboard
    - [X] Desenvolver o dashboard para apresentar as análises e/ou permitir a interação com o modelo
- [X] **Refinamento Visual do Dashboard e Preparação da Apresentação (Pitch de 10 minutos)**
    - [X] Melhorar o aspeto visual do dashboard (cores, layout, gráficos)
    - [X] Estruturar o conteúdo da apresentação (problema, dados, metodologia, resultados, conclusões, próximos passos) usando o dashboard como apoio
    - [X] Criar os slides (se necessário, como complemento ao dashboard) - Esboço `insights_apresentacao.md` criado
    - [X] Preparar o discurso e ensaiar a apresentação - Utilizador irá focar-se no dashboard
- [ ] **Organização do Código e Repositório**
    - [ ] Limpar e comentar o código Python
    - [ ] Organizar os ficheiros do projeto
    - [ ] Criar um repositório público no GitHub e subir o projeto

## Fase 4: Revisão Final e Entrega

- [ ] Rever a apresentação, o código e a documentação
- [ ] Garantir que todos os requisitos do projeto foram cumpridos
- [ ] Enviar o link do repositório GitHub para o email indicado

