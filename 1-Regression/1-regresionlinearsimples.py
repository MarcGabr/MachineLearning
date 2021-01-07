
from sklearn.datasets import load_boston
boston = load_boston() 
# Carregando Bibliotecas Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

dataset = pd.DataFrame(boston.data, columns = boston.feature_names)
dataset['target'] = boston.target
print(dataset.head(5))
'''
#Calculando o erro quadrado Medio SSE
valor_medio_esperado_na_previsao = dataset['target'].mean()
squared_errors = pd.Series(valor_medio_esperado_na_previsao - dataset['target'])**2 
SSE = np.sum(squared_errors)
print ('Soma dos Quadrados dos Erros (SSE): %01.f' % SSE)
#hist_plot = squared_errors.plot(kind = 'hist')
#plt.show()

#Calculando o desvio padrão de uma variavel
print ('Resultado do Numpy: %0.5f' % np.std(dataset['RM']))

#Calculando a correlação entre a variavel RM com a target
from scipy.stats.stats import pearsonr
print ('Correlação a partir da função pearsonr do SciPy: %0.5f' % pearsonr(dataset['RM'], dataset['target'])[0])

#Demonstrando a correlação(Entre o numero médio quantidade de quartos (RM), com a ocupação de casas(target).)
x_range = [dataset['RM'].min(),dataset['RM'].max()]
y_range = [dataset['target'].min(),dataset['target'].max()]
scatter_plot = dataset.plot(kind = 'scatter', x = 'RM', y = 'target', xlim = x_range, ylim = y_range)
meanY = scatter_plot.plot(x_range, [dataset['target'].mean(),dataset['target'].mean()], '--', color = 'red', linewidth = 1)
meanX = scatter_plot.plot([dataset['RM'].mean(), dataset['RM'].mean()], y_range, '--', color = 'red', linewidth = 1)
plt.show()
'''

'''
#Regressão com Scikit-Learn( Não foram tratados os dados)
from sklearn import linear_model
modelo = linear_model.LinearRegression(normalize = False, fit_intercept = True)
# Define os valores de x e y
num_observ = len(dataset)
X = dataset['RM'].values.reshape((num_observ, 1)) # X deve sempre ser uma matriz e nunca um vetor
y = dataset['target'].values # y pode ser um vetor
modelo.fit(X,y)
print (modelo.coef_)
print (modelo.intercept_)
print (modelo.predict(X)[:10])

# Minimizar a função de custo(A linha de regressão)
#Gradient Descent
observations = len(dataset)
X = dataset['RM'].values.reshape((observations,1)) 
X = np.column_stack((X,np.ones(observations))) 
y = dataset['target'].values 

import random
def random_w( p ):
    return np.array([np.random.normal() for j in range(p)])

def hypothesis(X,w):
    return np.dot(X,w)

def loss(X,w,y):
    return hypothesis(X,w) - y

def squared_loss(X,w,y):
    return loss(X,w,y)**2

def gradient(X,w,y):
    gradients = list()
    n = float(len( y ))
    for j in range(len(w)):
        gradients.append(np.sum(loss(X,w,y) * X[:,j]) / n)
    return gradients

def update(X,w,y, alpha = 0.01):
    return [t - alpha*g for t, g in zip(w, gradient(X,w,y))]

def optimize(X,y, alpha = 0.01, eta = 10**-12, iterations = 1000):
    w = random_w(X.shape[1])
    path = list()
    for k in range(iterations):
        SSL = np.sum(squared_loss(X,w,y))
        new_w = update(X,w,y, alpha = alpha)
        new_SSL = np.sum(squared_loss(X,new_w,y))
        w = new_w
        if k>=5 and (new_SSL - SSL <= eta and new_SSL - SSL >= -eta):
            path.append(new_SSL)
            return w, path
        if k % (iterations / 20) == 0:
            path.append(new_SSL)
    return w, path

# Definindo o valor de alfa
# Alfa é chamado de taxa de aprendizagem
alpha = 0.048
# Otimizando a Cost Function
w, path = optimize(X, y, alpha, eta = 10**-12, iterations = 25000)
# Imprimindo o resultado
print ("Valor Final dos Coeficientes: %s" % w)
print ("Percorrendo o Caminho do Gradiente em que o erro ao quadrado era %s" % path)
'''