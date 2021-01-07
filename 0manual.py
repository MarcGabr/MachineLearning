#Manual Machine Learning

# -Inicio
# Identificar o problema eo que deseja buscar. Pré-processe os dados. 

# Fase1 - Importação
    # Carregando arquivo csv usando Pandas
import pandas as pd
arquivo = 'data/pima-data.csv'
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dados = pd.read_csv(arquivo, names = colunas)
print(dados.shape)


# Fase2 - Analise dos dados(Exploratoria e Descritiva)
''' O tipo dos dados é muito importante. Pode ser necessário converter strings ou colunas com 
números inteiros podem representar variáveis categóricas ou valores ordinários.'''
# Visualizando as primeiras 20 linhas
dados.head(20)
# Visualizando as dimensões
dados.shape
# Tipo de dados de cada atributo
dados.dtypes
# Sumário estatístico. 
'''Importante para saber como se encontra a distribuição de dados. valores maximo e minimos e a distribuição dos quartis(25%,50%(Mediana),75%)'''
dados.describe()

'''Em problemas de classificação pode ser necessário balancear as classes. Classes desbalanceadas 
(ou seja, volume maior de um dos tipos das classes) são comuns e precisam ser tratadas durante 
a fase de pré-processamento.'''
# Distribuição das classes
'''Verificar o balanceamento das variaveis preditoras'''
dados.groupby('class').size()

'''A correlação é o relacionamento entre 2 variáveis. O métodos mais comum para calcular correlação é o método de Pearson, que assume uma distribuição normal dos dados. 
Correlação de -1 mostra uma correlação negativa, enquanto uma correlação de +1 mostra uma correlação positiva. 
Uma correlação igual a 0 mostra que não há relacionamento entre as variáveis.'''
# Correlação de Pearson
dados.corr(method = 'pearson')

'''Skew (ou simetria) se refere a distribuição dos dados que é assumida ser normal ou gaussiana (bell curve). 
Muitos algoritmos de Machine Learning consideram que os dados possuem uma distribuição normal. '''
# Verificando o skew de cada atributo
dados.skew()

'''Com o histograma podemos rapidamente avaliar a distribuição de cada atributo. Os histograma agrupam os dados em bins e fornecem uma contagem do número de observações 
em cada bin. Com o histograma, você pode rapidamente verificar a simetria dos dados e se eles estão em distribuição normal ou não. Isso também vai ajudar na identificação
 dos outliers.'''
# Histograma Univariado
dados.hist()
plt.show()
# Density Plot Univariado
dados.plot(kind = 'density', subplots = True, layout = (3,3), sharex = False)
plt.show()
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


# Fase 3 - Pré-processamento
    #Escala
'''Muitos algoritmos de Machine Learning vão se beneficiar disso e produzir resultados melhores. 
Esta etapa também é chamada de normalização e significa colocar os dados em uma escala com range entre 0 e 1.'''
# Transformando os dados para a mesma escala (entre 0 e 1)
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
# Carregando os dados
url = "http://datascienceacademy.com.br/blog/aluno/Python-Spark/Datasets/pima-data.csv"
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(url, names = colunas)
array = df.values
# Separando o array em componentes de input e output
X = array[:,0:8]
Y = array[:,8]
# Gerando a nova escala
scaler = MinMaxScaler(feature_range = (0, 1))
rescaledX = scaler.fit_transform(X)
# Sumarizando os dados transformados
print(rescaledX[0:5,:])

    #Padronização
'''Padronização é a técnica para transformar os atributos com distribuição Gaussiana (normal) e diferentes médias e desvios padrões
 em uma distribuição Gaussiana com a média igual a 0 e desvio padrão igual a 1.'''
 # Padronizando os dados (0 para a média, 1 para o desvio padrão)
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
# Carregando os dados
url = "http://datascienceacademy.com.br/blog/aluno/Python-Spark/Datasets/pima-data.csv"
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(url, names = colunas)
array = df.values
# Separando o array em componentes de input e output
X = array[:,0:8]
Y = array[:,8]
# Gerando o novo padrão
scaler = StandardScaler().fit(X)
standardX = scaler.transform(X)
# Sumarizando os dados transformados
print(standardX[0:5,:])

    #Normalização
'''Normalização se refere a ajustar a escala de cada observação (linha) de modo que ela tenha comprimento igual a 1 (chamado vetor de comprimento 1 em álgebra linear). 
Este método de pré-processamento é útil quando temos datasets esparsos (com muitos zeros) e atributos com escala muito variada.'''
# Normalizando os dados (comprimento igual a 1)
from pandas import read_csv
from sklearn.preprocessing import Normalizer
# Carregando os dados
url = "http://datascienceacademy.com.br/blog/aluno/Python-Spark/Datasets/pima-data.csv"
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(url, names = colunas)
array = df.values
# Separando o array em componentes de input e output
X = array[:,0:8]
Y = array[:,8]
# Gerando os dados normalizados
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
# Sumarizando os dados transformados
print(normalizedX[0:5,:])



# Fase 4 - Feature Selection
''' Fase que deverá selecionar os melhores atributos. 
Atributos irrelevante terão impacto negativo na performance, enquanto atributos colineares podem afetar o grau de acurácia do modelo.
A etapa de Feature Selection é onde selecionamos os atributos (variáveis) que serão melhores candidatas a variáveis preditoras. 
O Feature Selection nos ajuda a reduzir o overfitting (quando o algoritmo aprende demais), aumenta a acurácia do modelo e reduz o tempo de treinamento.
 '''
    #Seleção Univariada
''' Testes estatísticos podem ser usados para selecionar os atributos que possuem forte relacionamento com a variável que estamos tentando prever.'''
# Extração de Variáveis com Testes Estatísticos Univariados (Teste qui-quadrado)
from pandas import read_csv
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# Carregando os dados
url = "http://datascienceacademy.com.br/blog/aluno/Python-Spark/Datasets/pima-data.csv"
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(url, names = colunas)
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
'''Variaveis que possuirem melhor score'''
print(features[0:5,:])

    #Eliminação Recursiva de Atributos
'''Seleção de atributos, que recursivamente remove os atributos e constrói o modelo com os atributos remanescentes. 
Esta técnica utiliza a acurácia do modelo para identificar os atributos que mais contribuem para prever a variável alvo.'''
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# Carregando os dados
url = "http://datascienceacademy.com.br/blog/aluno/Python-Spark/Datasets/pima-data.csv"
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(url, names = colunas)
array = df.values
# Separando o array em componentes de input e output
X = array[:,0:8]
Y = array[:,8]
# Criação do modelo
modelo = LogisticRegression()
# RFE
'''Nesse caso procura-se as 3 melhores variaveis.'''
rfe = RFE(modelo, 3)
fit = rfe.fit(X, Y)
# Print dos resultados
print("Número de Atributos: %d" % fit.n_features_)
print(df.columns[0:8])
print("Atributos Selecionados: %s" % fit.support_)
print("Ranking dos Atributos: %s" % fit.ranking_)

    #Método Ensemble para Seleção de Variáveis
'''Bagged Decision Trees, como o algoritmo RandomForest, podem ser usados para estimar a importância de cada atributo. Esse método retorna um score para cada atributo.'''
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# Carregando os dados
url = "http://datascienceacademy.com.br/blog/aluno/Python-Spark/Datasets/pima-data.csv"
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(url, names = colunas)
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


    #VERIFICANDO SE OS ATRIBUTOS ATINGEM UM TAXA DE SIGNIFICANCIA OU SE ATINGIU UM RESULTADO SATISFATORIO.
        '''07 - MACHINE LEARNING (4.5, 4.6.1, 4.6.2)'''




# Fase 5 - Resampling

'''Divisão dos dados entre Treino e Teste '''
'''Treinamos o algoritmo nos dados de treino e fazemos as previsões nos dados de teste e avaliamos o resultado. 
A divisão dos dados vai depender do seu dataset, mas utiliza-se com frequência tamanhos entre 70/30 (treino/teste) e 65/35 (treino/teste)'''

    # Avaliação usando dados de treino e de teste
from pandas import read_csv
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
# Carregando os dados
url = "http://datascienceacademy.com.br/blog/aluno/Python-Spark/Datasets/pima-data.csv"
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(url, names = colunas)
array = df.values
# Separando o array em componentes de input e output
X = array[:,0:8]
Y = array[:,8]
# Definindo o tamanho das amostras
teste_size = 0.33 
'''Divide os dados entre 33% teste'''
# Garante que os resultados podem ser reproduzidos
# Isso é importante para comparar a acurácia com outros algoritmos de Machine Learning.
seed = 7 
'''Garante que os resutados seja os mesmos'''
# Criando os conjuntos de dados de treino e de teste
X_treino, X_teste, Y_treino, Y_teste = model_selection.train_test_split(X, Y, 
                                                                         test_size = teste_size, 
                                                                         random_state = seed)
# Criação do modelo
modelo = LogisticRegression()
modelo.fit(X_treino, Y_treino)
# Score
'''Acuracia: proximidade entre o valor obtido experimentalmente e o valor verdadeiro na medição'''
result = modelo.score(X_teste, Y_teste)
print("Acurácia: %.3f%%" % (result * 100.0))

    #Cross Validation
''' Dividir os dados em Folds. E apartir da quantidade de folds vai realizando treino e teste.'''
from pandas import read_csv
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
# Carregando os dados
url = "http://datascienceacademy.com.br/blog/aluno/Python-Spark/Datasets/pima-data.csv"
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
kfold = model_selection.KFold(n = num_instances, n_folds = num_folds, random_state = seed)
# Criando o modelo
modelo = LogisticRegression()
resultado = model_selection.cross_val_score(modelo, X, Y, cv = kfold)
# Usamos a média e o desvio padrão
print("Acurácia: %.3f%% (%.3f%%)" % (resultado.mean()*100.0, resultado.std() * 100.0))


'''Metricas utilizadas para avaliar a performace do modelo '''
    #Métricas para Algoritmos de Classificação
        #Acuracia
'''Número de previsões corretas. É útil apenas quando existe o mesmo número de observações em cada classe'''
from pandas import read_csv
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
# Carregando os dados
url = "http://datascienceacademy.com.br/blog/aluno/Python-Spark/Datasets/pima-data.csv"
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
modelo = LogisticRegression()
resultado = model_selection.cross_val_score(modelo, X, Y, cv = kfold, scoring = 'accuracy')
# Print dos resultados
print("Acurácia: %.3f (%.3f)" % (resultado.mean(), resultado.std()))

        #Curva ROC
''' A Curva ROC permite analisar a métrica AUC (Area Under the Curve).
 Essa é uma métrica de performance para classificação binária, em que podemos definir as classes em positiavs e negativas.
 Problemas de classificação binária são um trade-off sentre Sensitivity e Specifity.
 Sensitivity é a taxa de verdadeiros positivos (TP). Ese é o número de instâncias positivas da primeira classe que foram previstas corretamente.
 Specifity é a taxa de verdadeiros negativos (TN). Esse é o número de instâncias da segunda classe que foram previstas corretamente.
 Valores acima de 0.5 indicam uma boa taxa de previsão. '''
from pandas import read_csv
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
# Carregando os dados
url = "http://datascienceacademy.com.br/blog/aluno/Python-Spark/Datasets/pima-data.csv"
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
model = LogisticRegression()
resultado = model_selection.cross_val_score(model, X, Y, cv = kfold, scoring = 'roc_auc')
# Print do resultado
print("AUC: %.3f (%.3f)" % (resultado.mean(), resultado.std()))

        # Confusion Matrix
''' Verdadeiros positivos, VN, FP,FN '''
from pandas import read_csv
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
# Carregando os dados
url = "http://datascienceacademy.com.br/blog/aluno/Python-Spark/Datasets/pima-data.csv"
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
model = LogisticRegression()
model.fit(X_treino, Y_treino)
# Fazendo as previsões e construindo a Confusion Matrix
previsoes = model.predict(X_teste)
matrix = confusion_matrix(Y_teste, previsoes)
# Imprimindo a Confusion Matrix
print(matrix)


    #Métricas para Algoritmos de Regressão
        # MAE
'''Mean Absolute Error
É a soma da diferença absoluta entre previsões e valores reais.
Fornece uma ideia de quão erradas estão nossas previsões.
Valor igual a 0 indica que não há erro, sendo a previsão perfeita (a exemplo do Logloss, a função cross_val_score inverte o valor) '''
from pandas import read_csv
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
# Carregando os dados
url = "http://datascienceacademy.com.br/blog/aluno/Python-Spark/Datasets/boston-houses.csv"
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
'''Mean Squared Error
Similar ao MAE, fornece a magnitude do erro do modelo.
Ao extrairmos a raiz quadrada do MSE convertemos as unidades de volta ao original, o que pode ser útil para descrição e apresentação.
Isso é chamado RMSE (Root Mean Squared Error)'''
from pandas import read_csv
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
# Carregando os dados
url = "http://datascienceacademy.com.br/blog/aluno/Python-Spark/Datasets/boston-houses.csv"
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

        #R^2
'''Essa métrica fornece uma indicação do nível de precisão das previsões em relação aos valores observados.
 Também chamado de coeficiente de determinação.
 Valores entre 0 e 1, sendo 0 o valor ideal.'''
from pandas import read_csv
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
# Carregando os dados
url = "http://datascienceacademy.com.br/blog/aluno/Python-Spark/Datasets/boston-houses.csv"
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


# FASE - 5 APLICAÇÃO DOS ALGORITMOS


#FASE - 6 OTIMIZAÇÃO E AJUSTE DE PARAMETROS
    # Grid Search Parameter Tuning
'''Este método realiza metodicamente combinações entre todos os parâmetros do algoritmo, criando um grid.
Encontrar melhor paramentro e os algoritmos'''
from pandas import read_csv
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
url = "http://datascienceacademy.com.br/blog/aluno/Python-Spark/Datasets/pima-data.csv"
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(url, names = colunas)
array = df.values
# Separando o array em componentes de input e output
X = array[:,0:8]
Y = array[:,8]
# Definindo os valores que serão testados
valores_alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
valores_grid = dict(alpha = valores_alphas)
# Criando o modelo
modelo = Ridge()
# Criando o grid
grid = GridSearchCV(estimator = modelo, param_grid = valores_grid)
grid.fit(X, Y)
# Print do resultado
print(grid.best_score_)
print(grid.best_estimator_.alpha)

    # Random Search Parameter Tuning
'''Este método gera amostras dos parâmetros dos algoritmos a partir de uma distribuição randômica uniforme para um número fixo de interações. 
Um modelo é construído e testado para cada combinação de parâmetros. '''
# Import dos módulos
from pandas import read_csv
import numpy as np
from scipy.stats import uniform
from sklearn.linear_model import Ridge
from sklearn.grid_search import RandomizedSearchCV
# Carregando os dados
url = "http://datascienceacademy.com.br/blog/aluno/Python-Spark/Datasets/pima-data.csv"
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(url, names = colunas)
array = df.values
# Separando o array em componentes de input e output
X = array[:,0:8]
Y = array[:,8]
# Definindo os valores que serão testados
valores_grid = {'alpha': uniform()}
seed = 7
# Criando o modelo
modelo = Ridge()
iterations = 100
rsearch = RandomizedSearchCV(estimator = modelo, 
                             param_distributions = valores_grid, 
                             n_iter = iterations, 
                             random_state = seed)
rsearch.fit(X, Y)
# Print dos resultados
print(rsearch.best_score_)
print(rsearch.best_estimator_.alpha)


    #SALVAR OS MODELOS
from pandas import read_csv
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
# Carregando os dados
url = "http://datascienceacademy.com.br/blog/aluno/Python-Spark/Datasets/pima-data.csv"
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names = colunas)
array = dataframe.values
# Separando o array em componentes de input e output
X = array[:,0:8]
Y = array[:,8]
# Definindo o tamanho dos dados de treino e de teste
teste_size = 0.33
seed = 7
# Criando o dataset de treino e de teste
X_treino, X_teste, Y_treino, Y_teste = model_selection.train_test_split(X, Y, 
                                                                         test_size = teste_size, 
                                                                         random_state = seed)
# Criando o modelo
modelo = LogisticRegression()
modelo.fit(X_treino, Y_treino)
# Salvando o modelo
arquivo = 'modelo_v1.sav'
pickle.dump(modelo, open(arquivo, 'wb'))
# Carregando o arquivo
modelo_v1 = pickle.load(open(arquivo, 'rb'))
resultado = modelo_v1.score(X_teste, Y_teste)
# Print do resultado
print(resultado)