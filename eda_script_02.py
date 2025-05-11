import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset de treino
train_df = pd.read_csv("data/bootcamp_train.csv")

# Padronizar as colunas de falhas para booleano (True/False)
# Primeiro, identificar as colunas de falha
falha_cols = ['falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6', 'falha_outros']

# Criar um dicionário de mapeamento para os valores inconsistentes
# Adicionar mais mapeamentos conforme necessário após inspeção
map_sim_nao = {
    'Sim': True, 'sim': True, 'True': True, True: True,
    'Não': False, 'nao': False, 'Nao': False, 'False': False, False: False
}

print("Valores únicos nas colunas de falha ANTES da padronização:")
for col in falha_cols:
    print(f"{col}: {train_df[col].unique()}")
    # Tentar aplicar o mapeamento. Se a coluna já for booleana, pode dar erro, então tratamos.
    if train_df[col].dtype == 'object' or pd.api.types.is_string_dtype(train_df[col]):
        train_df[col] = train_df[col].map(map_sim_nao).astype(bool)
    elif pd.api.types.is_bool_dtype(train_df[col]):
        pass # Já é booleano, não faz nada
    else:
        # Para casos como a falha_5 que pode ter outros valores, vamos inspecionar e decidir
        # Por agora, se não for string/object ou bool, converter para bool considerando não nulo como True
        # Isto pode precisar de ajuste
        print(f"Atenção: Coluna {col} tem tipo {train_df[col].dtype} e será convertida para bool. Valores não nulos/não zero serão True.")
        train_df[col] = train_df[col].notna() & (train_df[col] != 0) # Exemplo, pode precisar de ajuste

# Verificar novamente após a tentativa de padronização
print("\nValores únicos nas colunas de falha DEPOIS da padronização inicial:")
for col in falha_cols:
    print(f"{col}: {train_df[col].unique()}")
    # Garantir que são booleanos
    train_df[col] = train_df[col].astype(bool)

print("\nInformações do dataset DEPOIS da padronização das colunas de falha:")
train_df.info()

# Análise da distribuição das variáveis alvo (falhas)
print("\nDistribuição das classes de falha:")
falhas_dist = train_df[falha_cols].sum().sort_values(ascending=False)
print(falhas_dist)

# Visualização da distribuição das falhas
plt.figure(figsize=(10, 6))
sns.barplot(x=falhas_dist.index, y=falhas_dist.values)
plt.title('Distribuição do Número de Ocorrências por Tipo de Falha')
plt.xlabel('Tipo de Falha')
plt.ylabel('Número de Ocorrências')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('images/distribuicao_falhas.png')
print("\nGráfico da distribuição das falhas guardado em distribuicao_falhas.png")

# Análise de coocorrência de falhas (quantas amostras têm múltiplas falhas)
# Adicionar uma coluna que conta o número de falhas por amostra
train_df['num_falhas'] = train_df[falha_cols].sum(axis=1)
print("\nDistribuição do número de falhas por amostra:")
print(train_df['num_falhas'].value_counts().sort_index())

# Análise de valores em falta (missing values)
print("\nValores em falta por coluna (%):")
missing_values = train_df.isnull().sum()
missing_percentage = (missing_values / len(train_df)) * 100
missing_info = pd.DataFrame({'Valores em Falta': missing_values, 'Percentagem (%)': missing_percentage})
missing_info = missing_info[missing_info['Valores em Falta'] > 0].sort_values(by='Percentagem (%)', ascending=False)
print(missing_info)

# Padronizar colunas tipo_do_aço_A300 e tipo_do_aço_A400
# Assumindo que são binárias (Sim/Não ou True/False ou 0/1)
# tipo_do_aço_A300
print(f"\nValores únicos em 'tipo_do_aço_A300' antes: {train_df['tipo_do_aço_A300'].unique()}")
if train_df['tipo_do_aço_A300'].dtype == 'object':
    train_df['tipo_do_aço_A300'] = train_df['tipo_do_aço_A300'].map(map_sim_nao).fillna(False).astype(bool) # Preenche NaN com False
print(f"Valores únicos em 'tipo_do_aço_A300' depois: {train_df['tipo_do_aço_A300'].unique()}")

# tipo_do_aço_A400
print(f"\nValores únicos em 'tipo_do_aço_A400' antes: {train_df['tipo_do_aço_A400'].unique()}")
if train_df['tipo_do_aço_A400'].dtype == 'object':
    # A coluna tipo_do_aço_A400 tem NaNs, que devem ser tratados. 
    # Se o mapeamento resultar em NaN para valores não reconhecidos, e depois preenchermos com False, está ok.
    train_df['tipo_do_aço_A400'] = train_df['tipo_do_aço_A400'].map(map_sim_nao)
    # Agora, vamos tratar os NaNs que podem ter vindo de valores não mapeados ou já existentes
    # Se a intenção é que NaN signifique 'não é A400', então preencher com False é razoável.
    # No entanto, é importante distinguir NaNs originais de NaNs por falha de mapeamento.
    # Para simplificar, vamos assumir que NaN original significa False e valores não mapeados também se tornam False.
    train_df['tipo_do_aço_A400'] = train_df['tipo_do_aço_A400'].fillna(False).astype(bool)
print(f"Valores únicos em 'tipo_do_aço_A400' depois: {train_df['tipo_do_aço_A400'].unique()}")

print("\nInformações do dataset DEPOIS da padronização das colunas de tipo de aço:")
train_df.info()


# Aqui o código foi crucial para: 
# Limpar e Padronizar as Variáveis Alvo: Garantir que as colunas de falha estão num formato numérico consistente (0/1).
# Entender a Natureza das Falhas: Ver quais são mais frequentes e confirmar que múltiplas falhas podem ocorrer numa mesma amostra, o que justifica a modelagem multirrótulo.
# Preparar o Terreno: Deixar os dados um pouco mais limpos e compreendidos antes de tratamentos de dados faltantes mais complexos e da modelagem em si.