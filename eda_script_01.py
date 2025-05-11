import pandas as pd

# Carregar o dataset de treino
train_df = pd.read_csv("data/bootcamp_train.csv")

# Exibir as primeiras linhas do dataframe
print("Primeiras 5 linhas do dataset de treino:")
print(train_df.head())

# Exibir informações sobre o dataframe (tipos de dados, valores não nulos)
print("\nInformações do dataset de treino:")
train_df.info()

# Exibir estatísticas descritivas das colunas numéricas
print("\nEstatísticas descritivas do dataset de treino:")
print(train_df.describe(include='all'))

# Verificar as dimensões do dataset
print(f"\nO dataset de treino tem {train_df.shape[0]} linhas e {train_df.shape[1]} colunas.")

#Este meu código está pegando os dados inicialmente e analisando eles, bem simples diretamente com pandas.O objetivo da Análise Exploratória de Dados é conhecer os dados, entender a sua estrutura, identificar possíveis problemas (como dados em falta ou erros), e começar a ter uma ideia dos padrões que eles podem conter. É como um detetive a examinar a cena do crime antes de formular qualquer teoria.

