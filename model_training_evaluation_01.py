import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

# Carregar o dataset pré-processado
df = pd.read_csv("data/bootcamp_train_processed.csv")

# Definir as colunas de features (X) e as colunas alvo (y)
falha_cols = [f'falha_{i}' for i in range(1, 7)] + ['falha_outros']

# Remover a coluna 'peso_da_placa' pois tem variância zero (constante)
if 'peso_da_placa' in df.columns:
    df = df.drop(columns=['peso_da_placa'])
    print("Coluna 'peso_da_placa' removida.")

feature_cols = [col for col in df.columns if col not in falha_cols + ['id']]

X = df[feature_cols]
y = df[falha_cols].astype(int) # Garantir que as colunas alvo são inteiros (0 ou 1)

print(f"Formato de X: {X.shape}")
print(f"Formato de y: {y.shape}")

# Dividir os dados em conjuntos de treino e validação
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Formato de X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Formato de X_val: {X_val.shape}, y_val: {y_val.shape}")

# --- Modelagem e Avaliação ---

# Modelos a serem testados
models = {
    "Logistic Regression": MultiOutputClassifier(LogisticRegression(solver='liblinear', random_state=42)),
    "Random Forest": MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    # Adicionar mais modelos aqui (ex: Gradient Boosting, SVM)
}

results = {}

print("\nIniciando treino e avaliação dos modelos...")

for model_name, model in models.items():
    print(f"\nTreinando {model_name}...")
    model.fit(X_train, y_train)
    
    print(f"Avaliando {model_name} no conjunto de validação...")
    y_pred_val = model.predict(X_val)
    y_pred_proba_val = model.predict_proba(X_val) # Para AUC-ROC
    
    # Métricas de avaliação
    # Hamming Loss: A fração de rótulos que são incorretamente previstos.
    # Quanto menor, melhor.
    h_loss = hamming_loss(y_val, y_pred_val)
    
    # Accuracy Score (Exact Match Ratio): Percentagem de amostras onde todos os rótulos são corretos.
    # É uma métrica rigorosa para multirrótulo.
    acc_score = accuracy_score(y_val, y_pred_val)
    
    # Métricas por classe (calculadas para cada classe e depois agregadas)
    # Usaremos 'macro' para dar igual peso a cada classe, bom para desbalanceamento.
    # Usaremos 'samples' para calcular métricas para cada instância e depois encontrar a sua média.
    precision_macro = precision_score(y_val, y_pred_val, average='macro', zero_division=0)
    recall_macro = recall_score(y_val, y_pred_val, average='macro', zero_division=0)
    f1_macro = f1_score(y_val, y_pred_val, average='macro', zero_division=0)
    
    precision_samples = precision_score(y_val, y_pred_val, average='samples', zero_division=0)
    recall_samples = recall_score(y_val, y_pred_val, average='samples', zero_division=0)
    f1_samples = f1_score(y_val, y_pred_val, average='samples', zero_division=0)

    # AUC-ROC por classe e depois média
    # predict_proba para MultiOutputClassifier retorna uma lista de arrays de probabilidade, um por output/classe
    # Precisamos calcular AUC para cada classe e depois fazer a média
    auc_scores_per_class = []
    for i in range(y_val.shape[1]):
        try:
            # y_pred_proba_val[i] é a probabilidade para a classe i
            # Acessamos a probabilidade da classe positiva [:, 1]
            auc = roc_auc_score(y_val.iloc[:, i], y_pred_proba_val[i][:, 1])
            auc_scores_per_class.append(auc)
        except ValueError as e:
            # Pode acontecer se uma classe não tiver amostras positivas no conjunto de validação
            print(f"Não foi possível calcular AUC para a classe {y_val.columns[i]}: {e}")
            auc_scores_per_class.append(np.nan) # Adicionar NaN se não puder ser calculado
    
    roc_auc_macro = np.nanmean(auc_scores_per_class) # Média ignorando NaNs

    results[model_name] = {
        'Hamming Loss': h_loss,
        'Accuracy (Exact Match Ratio)': acc_score,
        'Precision (Macro)': precision_macro,
        'Recall (Macro)': recall_macro,
        'F1-score (Macro)': f1_macro,
        'Precision (Samples)': precision_samples,
        'Recall (Samples)': recall_samples,
        'F1-score (Samples)': f1_samples,
        'AUC-ROC (Macro)': roc_auc_macro
    }
    
    print(f"Resultados para {model_name}:")
    for metric, value in results[model_name].items():
        print(f"  {metric}: {value:.4f}")

# Exibir resultados comparativos
results_df = pd.DataFrame(results).T.sort_values(by='F1-score (Macro)', ascending=False)
print("\nResultados Comparativos dos Modelos (ordenado por F1-score Macro):")
print(results_df)

results_df.to_csv("data/model_evaluation_results.csv")
print("\nResultados da avaliação dos modelos guardados em model_evaluation_results.csv")

print("\nModelagem e avaliação inicial concluídas.")

#Em resumo, o model_training_evaluation_01.py permitiu-nos:
# Treinar dois tipos diferentes de modelos de machine learning (um simples e um mais complexo).
# Avaliar o desempenho de cada modelo de forma objetiva usando várias métricas apropriadas para classificação multirrótulo.
# Comparar os modelos e identificar qual deles teve o melhor desempenho no nosso conjunto de validação (o Random Forest).
# Estabelecer uma base para os próximos passos, como otimizar o melhor modelo ou usá-lo para gerar as previsões finais.


