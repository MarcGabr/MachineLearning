''' #Visualização
from pandas import read_csv
arquivo = 'pima-data.csv'
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dados = read_csv(arquivo, names = colunas)
print(dados.head(20))
import matplotlib.pyplot as plt
#dados.hist()
#plt.show()
# Vendo se o grafico está 'normal'
#dados.plot(kind = 'density', subplots = True, layout = (3,3), sharex = False)
#plt.show()
# Matriz de Correlação com nomes das variáveis
correlations = dados.corr()
# Plot
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin = -1, vmax = 1)
fig.colorbar(cax)
ticks = np.arange(0, 9, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(colunas)
ax.set_yticklabels(colunas)
plt.show()
'''


''' #Pré-processamento
#   Escala
#  (Mudança de escala para valores 0 e 1)
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
# Carregando os dados
arquivo = 'pima-data.csv'
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(arquivo, names = colunas)
array = df.values
# Separando o array em componentes de input e output
X = array[:,0:8]
Y = array[:,8]
# Gerando a nova escala
scaler = MinMaxScaler(feature_range = (0, 1))
rescaledX = scaler.fit_transform(X)
# Sumarizando os dados transformados
#print(rescaledX)
print(rescaledX[0:5,:])

#Padronização
#Trocar todas as variaveis para um distribuição normal com média igual a 0 e desvio padrão igual a 1.
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
# Carregando os dados
arquivo = 'pima-data.csv'
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(arquivo, names = colunas)
array = df.values
# Separando o array em componentes de input e output
X = array[:,0:8]
Y = array[:,8]
scaler = StandardScaler().fit(X)
standardX = scaler.transform(X)
# Sumarizando os dados transformados
print(standardX[0:5,:])

#Normalização 
#Este método de pré-processamento é útil quando temos datasets esparsos (com muitos zeros) e atributos com escala muito variada.
from pandas import read_csv
from sklearn.preprocessing import Normalizer
# Carregando os dados
arquivo = 'pima-data.csv'
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(arquivo, names = colunas)
array = df.values
# Separando o array em componentes de input e output
X = array[:,0:8]
Y = array[:,8]
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
# Sumarizando os dados transformados
print(normalizedX[0:5,:])
'''


'''  #Escolhe melhores variaveis para predição
#Seleção univariada
from pandas import read_csv
from sklearn.feature_selection import SelectKBest #Utilizando o metodo qui-quadrado
from sklearn.feature_selection import chi2
# Carregando os dados
arquivo = 'pima-data.csv'
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(arquivo, names = colunas)
array = df.values
# Separando o array em componentes de input e output
X = array[:,0:8]
Y = array[:,8]
# Extração de Variáveis
test = SelectKBest(score_func = chi2, k = 4)
fit = test.fit(X, Y)
# Sumarizando o score
print(fit.scores_)
features = fit.transform(X)
# Sumarizando atributos selecionados
print(features[0:5,:])

#Seleção Ensemble de variaveis
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# Carregando os dados
arquivo = 'pima-data.csv'
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(arquivo, names = colunas)
array = df.values
# Separando o array em componentes de input e output
X = array[:,0:8]
Y = array[:,8]
# Criação do Modelo - Feature Selection
modelo = ExtraTreesClassifier()
modelo.fit(X, Y)
# Print dos Resultados
print(df.columns[0:8])
print(modelo.feature_importances_)
'''



'''#Amostragem
#Dados de Treino e de Teste

# Avaliação usando dados de treino e de teste
from pandas import read_csv
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

# Carregando os dados
url = 'pima-data.csv'
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(url, names = colunas)
array = df.values
# Separando o array em componentes de input e output
X = array[:,0:8]
Y = array[:,8]
# Definindo o tamanho das amostras 33% para treino e 77 para teste
teste_size = 0.33
# Garante que os resultados podem ser reproduzidos
seed = 7
# Criando os conjuntos de dados de treino e de teste

X_treino, X_teste, Y_treino, Y_teste = model_selection.train_test_split(X, Y, 
                                                                         test_size = teste_size, 
                                                                         random_state = seed)
# Criação do modelo

modelo = LogisticRegression(max_iter=200)
modelo.fit(X_treino, Y_treino)

# Score
result = modelo.score(X_teste, Y_teste)
print("Acurácia: %.3f%%" % (result * 100.0))


# CrossValidation
#Varios testes para validar o modelo testado
from pandas import read_csv
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
# Carregando os dados
url = 'pima-data.csv'
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(url, names = colunas)
array = df.values
# Separando o array em componentes de input e output
X = array[:,0:8]
Y = array[:,8]
# Definindo os valores para os folds
num_folds = 10
num_instances = len(X)
seed = 7
# Separando os dados em folds
kfold = model_selection.KFold(num_folds, True, random_state = seed)
# Criando o modelo
modelo = LogisticRegression(max_iter=200)
resultado = model_selection.cross_val_score(modelo, X, Y, cv = kfold)
# Usamos a média e o desvio padrão
print("Acurácia: %.3f%% (%.3f%%)" % (resultado.mean()*100.0, resultado.std() * 100.0))
'''

''' Avaliar Performace Classificação


# Acurácia
# Número de previsões corretas. É útil apenas quando existe o mesmo número de observações em cada classe.
# Import dos módulos
from pandas import read_csv
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
# Carregando os dados
url = 'pima-data.csv'
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(url, names = colunas)
array = df.values
# Separando o array em componentes de input e output
X = array[:,0:8]
Y = array[:,8]
# Definindo os valores para o número de folds
num_folds = 10
num_instances = len(X)
seed = 7
# Separando os dados em folds
kfold = model_selection.KFold(num_folds, True, random_state = seed)
# Criando o modelo
modelo = LogisticRegression(max_iter=200)
resultado = model_selection.cross_val_score(modelo, X, Y, cv = kfold, scoring = 'accuracy')
# Print dos resultados
print("Acurácia: %.3f (%.3f)" % (resultado.mean(), resultado.std()))


# Curva ROC 
# A Curva ROC permite analisar a métrica AUC (Area Under the Curve).
# Essa é uma métrica de performance para classificação binária, em que podemos definir as classes em positiavs e negativas.
# Problemas de classificação binária são um trade-off sentre Sensitivity e Specifity.
# Sensitivity é a taxa de verdadeiros positivos (TP). Ese é o número de instâncias positivas da primeira classe que foram previstas corretamente.
# Specifity é a taxa de verdadeiros negativos (TN). Esse é o número de instâncias da segunda classe que foram previstas corretamente.
# Valores acima de 0.5 indicam uma boa taxa de previsão.
# Import dos módulos
from pandas import read_csv
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
# Carregando os dados
url = 'pima-data.csv'
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(url, names = colunas)
array = df.values
# Separando o array em componentes de input e output
X = array[:,0:8]
Y = array[:,8]
# Definindo os valores para o número de folds
num_folds = 10
num_instances = len(X)
seed = 7
# Separando os dados em folds
kfold = model_selection.KFold(num_folds, True, random_state = seed)
# Criando o modelo
modelo = LogisticRegression(max_iter=200)
resultado = model_selection.cross_val_score(modelo, X, Y, cv = kfold, scoring = 'roc_auc')
# Print do resultado
print("AUC: %.3f (%.3f)" % (resultado.mean(), resultado.std()))



# Confusion Matrix
# Permite verificar a acurácia de um modelo com duas classes
# Import dos módulos
from pandas import read_csv
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
# Carregando os dados
url = 'pima-data.csv'
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(url, names = colunas)
array = df.values
# Separando o array em componentes de input e output
X = array[:,0:8]
Y = array[:,8]
# Definindo o tamanho do conjunto de dados
teste_size = 0.33
seed = 7
# Dividindo os dados em treino e teste
X_treino, X_teste, Y_treino, Y_teste = model_selection.train_test_split(X, Y, 
                                                                         test_size = teste_size, 
                                                                         random_state = seed)
# Criando o modelo
model = LogisticRegression(max_iter=200)
model.fit(X_treino, Y_treino)
# Fazendo as previsões e construindo a Confusion Matrix
previsoes = model.predict(X_teste)
matrix = confusion_matrix(Y_teste, previsoes)
# Imprimindo a Confusion Matrix
print(matrix)


# Logarithmic Loss
# Avalia as previsões de probabilidade de um membro pertencer a uma determinada classe. 
# Valores menores indicam boa performance, com 0 representando um perfeito logloss. A função cross_val_score() inverte o valor.
# Import dos módulos
from pandas import read_csv
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
# Carregando os dados
url = 'pima-data.csv'
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(url, names = colunas)
array = df.values
# Separando o array em componentes de input e output
X = array[:,0:8]
Y = array[:,8]
# Definindo os valores para o número de folds
num_folds = 10
num_instances = len(X)
seed = 7
# Separando os dados em folds
kfold = model_selection.KFold(num_folds, True, random_state = seed)
# Criando o modelo
modelo = LogisticRegression(max_iter=200)
resultado = model_selection.cross_val_score(modelo, X, Y, cv = kfold, scoring = 'neg_log_loss')
# Print do resultado
print("Logloss: %.3f (%.3f)" % (resultado.mean(), resultado.std()))


# Relatório de Classificação
# Import dos módulos
from pandas import read_csv
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# Carregando os dados
url = 'pima-data.csv'
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(url, names = colunas)
array = df.values
# Separando o array em componentes de input e output
X = array[:,0:8]
Y = array[:,8]
# Definindo o tamanho do conjunto de dados
teste_size = 0.33
seed = 7
# Dividindo os dados em treino e teste
X_treino, X_teste, Y_treino, Y_teste = model_selection.train_test_split(X, Y, 
                                                                         test_size = teste_size, 
                                                                         random_state = seed)
# Criando o modelo
modelo = LogisticRegression(max_iter=200)
modelo.fit(X_treino, Y_treino)
# Fazendo as previsões e construindo o relatório
previsoes = model.predict(X_teste)
report = classification_report(Y_teste, previsoes)
# Imprimindo o relatório
print(report)
'''


#''' Avaliar Performace Regressão
# MAE
# Mean Absolute Error
# É a soma da diferença absoluta entre previsões e valores reais.
# Fornece uma ideia de quão erradas estão nossas previsões.
# Valor igual a 0 indica que não há erro, sendo a previsão perfeita (a exemplo do Logloss, a função cross_val_score inverte o valor) 
# Import dos módulos
from pandas import read_csv
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
# Carregando os dados
url = "boston-houses.csv"
colunas = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT', 'MEDV']
df = read_csv(url, delim_whitespace = True, names = colunas)
array = df.values
# Separando o array em componentes de input e output
X = array[:,0:13]
Y = array[:,13]
# Definindo os valores para o número de folds
num_folds = 10
num_instances = len(X)
seed = 7
# Separando os dados em folds
kfold = model_selection.KFold(num_folds, True, random_state = seed)
# Criando o modelo
modelo = LinearRegression()
resultado = model_selection.cross_val_score(modelo, X, Y, cv = kfold, scoring = 'neg_mean_absolute_error')
# Print do resultado
print("MAE: %.3f (%.3f)" % (resultado.mean(), resultado.std()))


# MSE
# Mean Squared Error
# Similar ao MAE, fornece a magnitude do erro do modelo.
# Ao extrairmos a raiz quadrada do MSE convertemos as unidades de volta ao original, o que pode ser útil para descrição e apresentação.
# Isso é chamado RMSE (Root Mean Squared Error)
# Import dos módulos
from pandas import read_csv
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
# Carregando os dados
url = "boston-houses.csv"
colunas = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT', 'MEDV']
df = read_csv(url, delim_whitespace = True, names = colunas)
array = df.values
# Separando o array em componentes de input e output
X = array[:,0:13]
Y = array[:,13]
# Definindo os valores para o número de folds
num_folds = 10
num_instances = len(X)
seed = 7
# Separando os dados em folds
kfold = model_selection.KFold(num_folds, True, random_state = seed)
# Criando o modelo
modelo = LinearRegression()
resultado = model_selection.cross_val_score(modelo, X, Y, cv = kfold, scoring = 'neg_mean_squared_error')
# Print do resultado
print("MSE: %.3f (%.3f)" % (resultado.mean(), resultado.std()))


# R^2
# Essa métrica fornece uma indicação do nível de precisão das previsões em relação aos valores observados.
# Também chamado de coeficiente de determinação.
# Valores entre 0 e 1, sendo 0 o valor ideal.
# Import dos módulos
from pandas import read_csv
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
# Carregando os dados
url = "boston-houses.csv"
colunas = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT', 'MEDV']
df = read_csv(url, delim_whitespace = True, names = colunas)
array = df.values
# Separando o array em componentes de input e output
X = array[:,0:13]
Y = array[:,13]
# Definindo os valores para o número de folds
num_folds = 10
num_instances = len(X)
seed = 7
# Separando os dados em folds
kfold = model_selection.KFold(num_folds, True, random_state = seed)
# Criando o modelo
modelo = LinearRegression()
resultado = model_selection.cross_val_score(modelo, X, Y, cv = kfold, scoring = 'r2')
# Print do resultado
print("R^2: %.3f (%.3f)" % (resultado.mean(), resultado.std()))