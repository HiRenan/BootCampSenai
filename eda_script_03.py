import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Carregar o dataset de treino (já com as colunas de falha e tipo de aço padronizadas do script anterior)
# Para manter a consistência, vamos recarregar e aplicar as transformações do script 2 aqui
# Idealmente, salvaríamos o dataframe processado, mas para scripts independentes, repetimos a lógica de limpeza inicial.
train_df = pd.read_csv("data/bootcamp_train.csv")

# Padronizar as colunas de falhas para booleano (True/False)
falha_cols = [f'falha_{i}' for i in range(1, 7)] + ['falha_outros']
map_sim_nao = {
    'Sim': True, 'sim': True, 'True': True, True: True,
    'Não': False, 'nao': False, 'Nao': False, 'False': False, False: False
}
for col in falha_cols:
    if train_df[col].dtype == 'object' or pd.api.types.is_string_dtype(train_df[col]):
        train_df[col] = train_df[col].map(map_sim_nao).astype(bool)
    elif pd.api.types.is_bool_dtype(train_df[col]):
        pass
    else:
        train_df[col] = train_df[col].notna() & (train_df[col] != 0)
    train_df[col] = train_df[col].astype(bool)

# Padronizar colunas tipo_do_aço_A300 e tipo_do_aço_A400
if train_df['tipo_do_aço_A300'].dtype == 'object':
    train_df['tipo_do_aço_A300'] = train_df['tipo_do_aço_A300'].map(map_sim_nao).fillna(False).astype(bool)
if train_df['tipo_do_aço_A400'].dtype == 'object':
    train_df['tipo_do_aço_A400'] = train_df['tipo_do_aço_A400'].map(map_sim_nao).fillna(False).astype(bool)

# Identificar colunas numéricas (excluindo ID e as colunas alvo/booleanas já tratadas)
exclude_cols = ['id'] + falha_cols + ['tipo_do_aço_A300', 'tipo_do_aço_A400'] # num_falhas foi criada no script 2, não está aqui
numerical_cols = [col for col in train_df.columns if train_df[col].dtype in [np.int64, np.float64] and col not in exclude_cols]

# --- Análise de Outliers com Boxplots ---
print("\nGerando boxplots para identificar outliers nas variáveis numéricas...")
plt.figure(figsize=(20, 15))
for i, col in enumerate(numerical_cols):
    plt.subplot(6, 5, i + 1) # Ajustar o layout conforme o número de colunas
    sns.boxplot(y=train_df[col])
    plt.title(col)
    plt.ylabel('')
plt.tight_layout()
plt.savefig('images/boxplots_variaveis_numericas.png')
print("Boxplots guardados em boxplots_variaveis_numericas.png")
plt.close()

# --- Análise de Distribuição com Histogramas ---
print("\nGerando histogramas para as variáveis numéricas...")
plt.figure(figsize=(20, 15))
for i, col in enumerate(numerical_cols):
    plt.subplot(6, 5, i + 1) # Ajustar o layout
    sns.histplot(train_df[col].dropna(), kde=True)
    plt.title(col)
    plt.xlabel('')
    plt.ylabel('')
plt.tight_layout()
plt.savefig('images/histogramas_variaveis_numericas.png')
print("Histogramas guardados em histogramas_variaveis_numericas.png")
plt.close()

# --- Análise de Correlações ---
print("\nCalculando e visualizando a matriz de correlação...")
# Selecionar apenas colunas numéricas para a correlação, excluindo ID
# As colunas booleanas (falhas, tipo_aço) podem ser convertidas para int (0/1) para incluir na correlação
corr_df = train_df.copy()
for col in falha_cols + ['tipo_do_aço_A300', 'tipo_do_aço_A400']:
    corr_df[col] = corr_df[col].astype(int)

# Excluir a coluna 'id' que não é relevante para correlação de features
if 'id' in corr_df.columns:
    corr_df_numeric = corr_df.drop(columns=['id'])
else:
    corr_df_numeric = corr_df

correlation_matrix = corr_df_numeric.corr()

plt.figure(figsize=(28, 24))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".1f") # annot=True pode ser muito denso
plt.title('Matriz de Correlação das Variáveis')
plt.tight_layout()
plt.savefig('images/matriz_correlacao.png')
print("Matriz de correlação guardada em matriz_correlacao.png")
plt.close()

# Mostrar correlações mais altas com as variáveis alvo (falhas)
print("\nPrincipais correlações com as variáveis alvo (falhas):")
for falha in falha_cols:
    correlations_target = correlation_matrix[falha].drop(falha_cols).sort_values(ascending=False)
    print(f"\nCorrelações com {falha}:")
    print(correlations_target.head(5)) # Top 5 positivas
    print(correlations_target.tail(5)) # Top 5 negativas (as menores)

print("\nAnálise de outliers, distribuições e correlações concluída.")

#Aqui foca-se em três aspetos principais:
# Análise de Outliers: Utiliza gráficos chamados boxplots para visualizar a distribuição de cada variável numérica e identificar valores que são muito diferentes do resto (os outliers).
# Distribuição das Variáveis Numéricas: Cria histogramas para cada variável numérica, mostrando como os seus valores estão distribuídos (se são simétricos, assimétricos, se têm picos, etc.).
# Análise de Correlação: Calcula e visualiza uma matriz de correlação, que mostra a força e a direção da relação linear entre pares de variáveis.

#Em Resumo:
# Identificar a presença e a localização de outliers nas nossas medições.
# Compreender a forma da distribuição de cada característica numérica.
# Descobrir relações lineares entre as diferentes medições e, mais importante, entre as medições e os tipos de falha.

