import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

games = pandas.read_csv("3-TreeDecision/games.csv")
plt.hist(games["average_rating"])
#plt.show()
# Visualizando as observações com rating igual a 0
#print(games[games["average_rating"] == 0])
# Retornando a primeira linha do subset do dataframe, onde o índice é igual a 0.
#print(games[games["average_rating"] == 0].iloc[0])
# Removendo as linhas sem avaliação de usuários.
games = games[games["users_rated"] > 0]
# Removendo linhas com valores missing
games = games.dropna(axis = 0)
plt.hist(games["average_rating"])
#plt.show()
#print(games.corr()["average_rating"])
# Obtém todas as colunas do dataframe
colunas = games.columns.tolist()
# Filtra as colunas e remove as que não são relevantes
colunas = [c for c in colunas if c not in ["bayes_average_rating", "average_rating", "type", "name"]]
# Preparando a variável target, a que será prevista
target = "average_rating"
# Gerando os dados de treino
df_treino = games.sample(frac = 0.8, random_state = 101)
# Seleciona tudo que não está no dataset de treino e armazena no dataset de teste
df_teste = games.loc[~games.index.isin(df_treino.index)]


# Criando um Regressor
reg_v1 = LinearRegression()
# Fit the model to the training data.
modelo_v1 = reg_v1.fit(df_treino[colunas], df_treino[target])
# Fazendo previsões
previsoes = modelo_v1.predict(df_teste[colunas])
# Computando os erros entre valores observados e valores previstos
print(mean_squared_error(previsoes, df_teste[target]))

'''
# Criando um regressor Random Forest
reg_v2 = RandomForestRegressor(n_estimators = 100, min_samples_leaf = 10, random_state = 101)
# Criando o modelo
modelo_v2 = reg_v2.fit(df_treino[colunas], df_treino[target])
# Fazendo previsões
previsoes = modelo_v2.predict(df_teste[colunas])
# Computando o erro
print(mean_squared_error(previsoes, df_teste[target]))
'''