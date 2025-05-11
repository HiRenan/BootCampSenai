# Esboço da Apresentação e Insights do Projeto

## 1. Introdução e Entendimento do Problema (Slide 1-2)

*   **Contexto:** Empresa siderúrgica busca um sistema inteligente de controlo de qualidade para chapas de aço inoxidável.
*   **Desafio:** Detetar e classificar 7 tipos de defeitos automaticamente a partir de 31 indicadores extraídos de imagens de superfície.
*   **Objetivo do Projeto:** Desenvolver um modelo de Machine Learning para prever a classe do defeito, retornar a probabilidade associada, extrair insights da operação e gerar visualizações.
*   **Impacto Potencial:** Melhoria da eficiência do controlo de qualidade, redução de custos, aumento da satisfação do cliente, otimização de processos na indústria siderúrgica.

## 2. Análise Exploratória dos Dados (EDA) - O Que os Dados Nos Contaram? (Slide 3-4)

*   **Dataset:** `bootcamp_train.csv` com 3390 amostras e 39 colunas (incluindo ID e 7 colunas alvo para os defeitos).
*   **Limpeza Inicial:**
    *   Padronização dos tipos de dados das colunas de falha (para booleano/numérico).
    *   Padronização das colunas `tipo_do_aço_A300` e `tipo_do_aço_A400`.
*   **Principais Achados da EDA:**
    *   **Distribuição das Falhas:**
        *   `falha_5` é a mais comum, seguida por `falha_outros` e `falha_6`.
        *   `falha_4` é a menos comum.
        *   (Referenciar o gráfico `distribuicao_falhas.png`)
    *   **Coocorrência de Falhas:** A maioria das amostras apresenta 1 ou 2 tipos de falha simultaneamente, justificando a abordagem multirrótulo.
    *   **Valores em Falta:** Identificadas e tratadas colunas como `x_maximo`, `soma_da_luminosidade`, `espessura_da_chapa_de_aço`, etc., usando imputação pela mediana.
    *   **Outliers:** Observados em várias variáveis numéricas (ex: `area_pixels`, `perimetro_x`, `perimetro_y`). Decisão de não os remover inicialmente, mas monitorar o impacto nos modelos.
        *   (Referenciar o gráfico `boxplots_variaveis_numericas.png`)
    *   **Distribuição das Variáveis:** Muitas variáveis não seguem distribuição normal, apresentando assimetrias.
        *   (Referenciar o gráfico `histogramas_variaveis_numericas.png`)
    *   **Correlações Relevantes (antes da modelagem):**
        *   `falha_3` mostrou forte correlação positiva com `log_indice_x`, `log_das_areas`, `index_externo_x`, `soma_da_luminosidade`. E correlação negativa com `indice_de_bordas_y`.
        *   `falha_1` com `indice_de_orientaçao`.
        *   `falha_2` com `indice_de_variacao_x`.
        *   `falha_6` com `tipo_do_aço_A300`.
        *   (Referenciar o gráfico `matriz_correlacao.png`)
    *   **Variável Constante:** `peso_da_placa` mostrou-se constante e foi removida no pré-processamento.

## 3. Metodologia de Modelagem (Slide 5-6)

*   **Abordagem do Problema:** Classificação Multirrótulo. Cada amostra pode ter um ou mais dos 7 tipos de defeitos. Esta abordagem é mais flexível e reflete a natureza do problema, onde múltiplos defeitos podem coexistir.
*   **Pré-processamento Adicional:**
    *   Imputação de valores em falta restantes com a mediana (para garantir que o modelo não receba NaNs).
    *   Padronização das features numéricas com `StandardScaler` (média 0, desvio padrão 1) para ajudar algoritmos sensíveis à escala das features.
*   **Divisão dos Dados:** Conjunto de treino dividido em 80% para treino e 20% para validação (`train_test_split`, `random_state=42` para reprodutibilidade).
*   **Modelos Experimentados:**
    *   Regressão Logística (como baseline).
    *   Random Forest (conhecido pela sua robustez e bom desempenho em diversos problemas, além de fornecer importância das features).
*   **Métricas de Avaliação (para Multirrótulo):**
    *   **Hamming Loss:** Fração de rótulos previstos incorretamente (quanto menor, melhor).
    *   **Accuracy (Exact Match Ratio):** Percentagem de amostras onde *todos* os rótulos são previstos corretamente (métrica rigorosa).
    *   **F1-score (Macro e Samples):** Média harmónica de precisão e recall. "Macro" calcula a métrica para cada classe e encontra a média não ponderada (bom para classes desbalanceadas). "Samples" calcula para cada instância.
    *   **AUC-ROC (Macro):** Capacidade do modelo de distinguir entre classes positivas e negativas, agregada para todas as classes.

## 4. Resultados e Discussão (Slide 7-8)

*   **Desempenho dos Modelos (no conjunto de validação):**
    *   **Random Forest:**
        *   Hamming Loss: 0.0839
        *   Accuracy (Exact Match Ratio): 0.5398
        *   F1-score (Macro): 0.5939
        *   AUC-ROC (Macro): 0.9078
    *   **Regressão Logística:**
        *   Hamming Loss: 0.1220
        *   F1-score (Macro): 0.5388
        *   AUC-ROC (Macro): 0.8819
    *   (Referenciar a tabela `model_evaluation_results.csv`)
*   **Escolha do Modelo Final:** Random Forest demonstrou ser superior, especialmente em F1-score Macro e Hamming Loss.
*   **Importância das Features (Modelo Random Forest - Top 5 Médio):**
    1.  `indice_de_variacao_x` (dispersão das coordenadas X do contorno do defeito)
    2.  `log_das_areas` (logaritmo da área do defeito)
    3.  `soma_da_luminosidade` (brilho total do defeito)
    4.  `sigmoide_das_areas` (transformação da área do defeito)
    5.  `index_externo_x` (pixels fora do defeito no eixo X)
    *   (Referenciar o gráfico `feature_importances_random_forest.png`)
*   **Insights Chave a partir das Features Mais Importantes:**
    *   **Geometria e Tamanho do Defeito:** Características como `log_das_areas`, `sigmoide_das_areas`, `indice_de_variacao_x` e `index_externo_x` são cruciais. Isto sugere que a forma, o tamanho e a dispersão espacial dos defeitos são fortes indicadores do tipo de falha.
        *   *Exemplo prático:* Defeitos muito espalhados no eixo X (`indice_de_variacao_x`) ou com grandes áreas (`log_das_areas`) podem ser mais facilmente identificados ou podem ser característicos de certos tipos de falhas.
    *   **Luminosidade:** A `soma_da_luminosidade` também é muito importante, indicando que o brilho ou contraste do defeito em relação à superfície da chapa é um diferenciador chave.
        *   *Exemplo prático:* Defeitos mais escuros ou mais claros que o normal podem ser rapidamente sinalizados pelo modelo.
    *   **Relação com a EDA:**
        *   A importância de `log_das_areas` e `soma_da_luminosidade` no modelo Random Forest reforça as correlações observadas na EDA, especialmente para `falha_3`. Isto sugere que o modelo está a aprender padrões consistentes com as observações iniciais.
        *   Se `indice_de_variacao_x` é a feature mais importante, isso pode indicar que variações ao longo do comprimento da chapa (ou na direção do transporte) são particularmente críticas para identificar defeitos.
*   **Implicações para a Empresa:**
    *   **Foco no Processo:** As features mais importantes podem dar pistas sobre quais etapas do processo de produção ou inspeção são mais críticas ou onde os sensores precisam ser mais precisos.
    *   **Manutenção Preditiva:** Se certos padrões de features consistentemente levam a defeitos específicos, isso pode, a longo prazo, ajudar a prever problemas nas máquinas ou no processo antes que causem defeitos.
    *   **Otimização da Inspeção:** O sistema pode priorizar a inspeção de chapas com características que o modelo considera de alto risco.

## 5. Conclusões e Próximos Passos (Slide 9-10)

*   **Conclusões Principais:**
    *   O modelo Random Forest demonstrou capacidade promissora para classificar múltiplos tipos de defeitos em chapas de aço, com um F1-score (Macro) de aproximadamente 0.59 e AUC-ROC (Macro) de 0.91 no conjunto de validação.
    *   As características geométricas (tamanho, forma, dispersão) e de luminosidade dos defeitos são os indicadores mais fortes para a classificação.
    *   O projeto cumpriu os requisitos de desenvolver um modelo preditivo e extrair insights.
*   **Limitações:**
    *   O conjunto de dados de teste não foi avaliado externamente via API (decisão do projeto).
    *   Não foi realizada otimização de hiperparâmetros, o que poderia melhorar ainda mais o desempenho.
    *   A interpretabilidade de modelos como Random Forest pode ser mais complexa que modelos lineares, embora a análise de feature importance ajude.
*   **Próximos Passos Sugeridos:**
    *   **Otimização de Hiperparâmetros:** Usar GridSearchCV ou RandomizedSearchCV para encontrar a melhor combinação de hiperparâmetros para o Random Forest.
    *   **Engenharia de Features:** Explorar a criação de novas features a partir das existentes (ex: rácios entre perímetros, densidade de pixels) que possam capturar melhor as características dos defeitos.
    *   **Experimentar Outros Modelos:** Testar algoritmos mais avançados como Gradient Boosting (XGBoost, LightGBM) ou Redes Neurais, especialmente se mais dados estiverem disponíveis.
    *   **Análise de Erros:** Investigar os casos onde o modelo mais erra para entender melhor as suas limitações e onde pode ser melhorado.
    *   **Desenvolvimento de Dashboard (Streamlit):** Criar uma interface interativa para visualização das análises, resultados do modelo e simulação de previsões (conforme interesse do utilizador).
    *   **Deployment:** Se o modelo for considerado suficientemente bom, planear o deployment via API (como FastAPI) para integração com os sistemas da empresa.
*   **Agradecimentos e Perguntas**

## 6. Organização do Código (GitHub)
*   Mencionar que o código será disponibilizado em scripts Python documentados num repositório público.
*   Estrutura sugerida: scripts para EDA, pré-processamento, treino/avaliação, geração de previsões, análise de feature importance.

