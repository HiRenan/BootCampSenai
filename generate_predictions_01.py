import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import warnings

warnings.filterwarnings("ignore")

# --- Funções de Pré-processamento (Reutilizadas e Adaptadas) ---
def preprocess_data(df, scaler=None, train_medians=None, is_train=True):
    print(f"Pré-processando dados... {'Treino' if is_train else 'Teste'}")
    # Padronizar as colunas de falhas para booleano (True/False) e depois para int (0/1)
    falha_cols = [f'falha_{i}' for i in range(1, 7)] + ['falha_outros']
    map_sim_nao = {
        'Sim': 1, 'sim': 1, 'True': 1, True: 1,
        'Não': 0, 'nao': 0, 'Nao': 0, 'False': 0, False: 0
    }

    # As colunas de falha só existem no conjunto de treino
    if is_train:
        for col in falha_cols:
            if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].apply(lambda x: map_sim_nao.get(x, 0) if isinstance(x, str) else (1 if x else 0) if pd.notnull(x) else 0)
            df[col] = df[col].astype(int)

    # Padronizar colunas tipo_do_aço_A300 e tipo_do_aço_A400 para int (0/1)
    for col_aco in ['tipo_do_aço_A300', 'tipo_do_aço_A400']:
        if col_aco in df.columns:
            if df[col_aco].dtype == 'object':
                # Mapeia e preenche NaNs com 0 (False) antes de converter para int
                df[col_aco] = df[col_aco].map(map_sim_nao).fillna(0).astype(int)
            elif pd.api.types.is_bool_dtype(df[col_aco]):
                 df[col_aco] = df[col_aco].astype(int)
            # Se já for int/float, pode ter NaNs que precisam ser tratados se não foram pelo map
            if df[col_aco].isnull().any():
                 df[col_aco] = df[col_aco].fillna(0).astype(int)
        else:
            print(f"Coluna {col_aco} não encontrada no dataframe.")

    # Tratamento de Valores em Falta (Missing Values) com medianas do treino
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Remover 'id' e colunas de falha da lista de colunas numéricas para imputação
    cols_to_impute = [col for col in numerical_cols if col != 'id' and col not in falha_cols]

    if is_train:
        train_medians = {} # Guardar medianas do treino
        for col in cols_to_impute:
            if df[col].isnull().any():
                median_val = df[col].median()
                train_medians[col] = median_val
                df[col].fillna(median_val, inplace=True)
                print(f"Valores em falta na coluna de treino '{col}' preenchidos com a mediana ({median_val}).")
    else:
        if train_medians:
            for col in cols_to_impute:
                if col in train_medians and df[col].isnull().any():
                    df[col].fillna(train_medians[col], inplace=True)
                    print(f"Valores em falta na coluna de teste '{col}' preenchidos com a mediana do treino ({train_medians[col]}).")
                elif df[col].isnull().any(): # Se a coluna não estava no train_medians mas tem NaNs
                    df[col].fillna(0, inplace=True) # Ou outra estratégia de fallback
                    print(f"Valores em falta na coluna de teste '{col}' preenchidos com 0 (mediana do treino não disponível ou coluna nova).")
        else:
            print("Medianas do treino não fornecidas para imputação no conjunto de teste.")
            # Fallback: preencher com 0 ou média global se necessário
            for col in cols_to_impute:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True) # Mediana do próprio teste como fallback
                    print(f"Alerta: Valores em falta na coluna de teste '{col}' preenchidos com a mediana do TESTE.")

    # No script de pré-processamento, foi visto que 'peso_da_placa' é constante no treino, então a removemos.
    if 'peso_da_placa' in df.columns:
        df = df.drop(columns=['peso_da_placa'])
        print("Coluna 'peso_da_placa' removida.")

    # Feature Scaling (Normalização/Padronização)
    features_to_scale = [col for col in df.columns if df[col].dtype in [np.int64, np.float64, np.int32, np.float32] and col not in falha_cols + ['id']]
    
    if is_train:
        scaler = StandardScaler()
        if features_to_scale: # Apenas escalar se houver colunas para escalar
            df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
            print("Variáveis numéricas do treino padronizadas.")
    else:
        if scaler and features_to_scale:
            df[features_to_scale] = scaler.transform(df[features_to_scale])
            print("Variáveis numéricas do teste padronizadas com scaler do treino.")
        elif not scaler:
            print("Scaler não fornecido para o conjunto de teste.")

    return df, scaler, train_medians

# --- Carregar Dados ---
print("Carregando datasets...")
train_df_orig = pd.read_csv("data/bootcamp_train.csv")
test_df_orig = pd.read_csv("data/bootcamp_test.csv")
submission_example_df = pd.read_csv("data/bootcamp_submission_example.csv")

# Guardar IDs do conjunto de teste para o ficheiro de submissão
test_ids = test_df_orig['id']

# --- Pré-processar Dados de Treino ---
train_df_processed, scaler_fitted, train_medians_fitted = preprocess_data(train_df_orig.copy(), is_train=True)

# Definir X_train e y_train
falha_cols = [f'falha_{i}' for i in range(1, 7)] + ['falha_outros']
X_train = train_df_processed.drop(columns=['id'] + falha_cols)
y_train = train_df_processed[falha_cols]

print(f"Formato final X_train: {X_train.shape}, y_train: {y_train.shape}")

# --- Pré-processar Dados de Teste ---
# Usar o scaler e medianas ajustados no conjunto de treino
test_df_processed, _, _ = preprocess_data(test_df_orig.copy(), scaler=scaler_fitted, train_medians=train_medians_fitted, is_train=False)
X_test = test_df_processed.drop(columns=['id'])

# Garantir que X_test tem as mesmas colunas que X_train (exceto 'id' e alvos)
# E na mesma ordem. Isso é crucial se alguma coluna foi removida (ex: peso_da_placa)
# ou se novas colunas foram criadas (não é o caso aqui, mas boa prática)
missing_cols_test = set(X_train.columns) - set(X_test.columns)
for c in missing_cols_test:
    X_test[c] = 0 # Adicionar colunas em falta com 0 (ou outra imputação apropriada)

extra_cols_test = set(X_test.columns) - set(X_train.columns)
X_test = X_test.drop(columns=list(extra_cols_test))

X_test = X_test[X_train.columns] # Assegurar a mesma ordem e conjunto de colunas

print(f"Formato final X_test: {X_test.shape}")

# --- Treinar Modelo Final (Random Forest) --- 
print("\nTreinando modelo Random Forest final com todos os dados de treino...")
final_model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
final_model.fit(X_train, y_train)
print("Modelo treinado.")

# --- Gerar Previsões no Conjunto de Teste ---
print("Gerando previsões no conjunto de teste...")
y_pred_test = final_model.predict(X_test)

# --- Criar Ficheiro de Submissão ---
print("Criando ficheiro de submissão...")
submission_df = pd.DataFrame(y_pred_test, columns=falha_cols)
submission_df.insert(0, 'id', test_ids)

# Garantir que as colunas de falha são inteiros (0 ou 1)
for col in falha_cols:
    submission_df[col] = submission_df[col].astype(int)

# Verificar se o formato corresponde ao exemplo
print("Primeiras linhas do ficheiro de submissão gerado:")
print(submission_df.head())
print("\nPrimeiras linhas do ficheiro de exemplo de submissão:")
print(submission_example_df.head())

if list(submission_df.columns) == list(submission_example_df.columns) and len(submission_df) == len(test_df_orig):
    print("Formato do ficheiro de submissão parece correto.")
else:
    print("ALERTA: Formato do ficheiro de submissão pode estar incorreto!")
    print(f"Colunas geradas: {list(submission_df.columns)}")
    print(f"Colunas esperadas: {list(submission_example_df.columns)}")
    print(f"Linhas geradas: {len(submission_df)}, Linhas no teste original: {len(test_df_orig)}")

submission_file_path = "data/submission_random_forest.csv"
submission_df.to_csv(submission_file_path, index=False)
print(f"\nFicheiro de submissão guardado em: {submission_file_path}")

print("\nGeração de previsões concluída.")

#Em resumo, o generate_predictions_01.py fez o seguinte:
# Preparou os dados de teste aplicando exatamente as mesmas transformações que foram aplicadas aos dados de treino.
# Usou o teu melhor modelo (Random Forest), treinado com todos os dados de treino, para fazer previsões nos dados de teste.
# Gerou o ficheiro submission_random_forest.csv, que é o produto final do teu esforço de modelagem, pronto para ser avaliado em dados completamente novos.

