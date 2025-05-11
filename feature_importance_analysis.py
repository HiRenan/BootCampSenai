import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
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
y = df[falha_cols].astype(int)

print(f"Formato de X: {X.shape}")
print(f"Formato de y: {y.shape}")

# Treinar o modelo Random Forest final com todos os dados de treino
print("\nTreinando modelo Random Forest para extrair feature importances...")
# O MultiOutputClassifier treina um classificador por alvo.
# Cada classificador (RandomForestClassifier) terá suas próprias feature_importances_.
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
model.fit(X, y)
print("Modelo treinado.")

# Extrair e agregar feature importances
# Vamos calcular a média das importâncias das features em todos os estimadores (um por classe alvo)
all_feature_importances = []
for estimator in model.estimators_:
    all_feature_importances.append(estimator.feature_importances_)

if all_feature_importances:
    # Calcular a média das importâncias das features
    mean_feature_importances = np.mean(all_feature_importances, axis=0)
    
    # Criar um DataFrame para visualização
    importances_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': mean_feature_importances
    }).sort_values(by='Importance', ascending=False)

    print("\nImportância Média das Features (Top 15):")
    print(importances_df.head(15))

    # Visualizar as importâncias das features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importances_df.head(15), palette='viridis') # Top 15
    plt.title('Importância Média das Features (Top 15) - Modelo Random Forest')
    plt.xlabel('Importância Média')
    plt.ylabel('Feature')
    plt.tight_layout()
    feature_importance_plot_path = "images/feature_importances_random_forest.png"
    plt.savefig(feature_importance_plot_path)
    print(f"\nGráfico de importância das features guardado em: {feature_importance_plot_path}")
    plt.close()
else:
    print("Não foi possível extrair as importâncias das features.")

print("\nAnálise de importância das features concluída.")

# Adicional: Importância por classe de falha (opcional, pode gerar muitos gráficos)
# for i, estimator in enumerate(model.estimators_):
#     target_class_name = y.columns[i]
#     importances_class_df = pd.DataFrame({
#         'Feature': feature_cols,
#         'Importance': estimator.feature_importances_
#     }).sort_values(by='Importance', ascending=False)
#     print(f"\nImportância das Features para {target_class_name} (Top 5):")
#     print(importances_class_df.head(5))

#Em resumo, o feature_importance_analysis.py permitiu-nos:
# Identificar quais das muitas características medidas nas chapas de aço são as mais influentes para o modelo Random Forest ao prever os tipos de falha.
# Obter uma medida quantitativa dessa influência.
# Gerar uma visualização clara para comunicar estes achados.
