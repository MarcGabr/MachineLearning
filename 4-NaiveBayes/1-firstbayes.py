
'''
#Exemplo 1 de Naive Bayes Gaussiano
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
# Dataset
iris = datasets.load_iris()
# Classificador
clf = GaussianNB()
# Modelo
modelo = clf.fit(iris.data, iris.target)
# Previsões
y_pred = modelo.predict(iris.data)
# Imprime o resultado
print("Total de Observações: %d - Total de Previsões Incorretas : %d" 
      % (iris.data.shape[0],(iris.target != y_pred).sum()))


#Exemplo 2 de Naive Bayes Gaussiano
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
# Dataset
dataset = datasets.load_iris()
# Classificador 
clf = GaussianNB()
# Modelo
modelo = clf.fit(dataset.data, dataset.target)
print(modelo)
# Previsões
observado = dataset.target
previsto = modelo.predict(dataset.data)
# Sumário( Mostrando que acertou 50 da primeira classe, 47 da segunda(com 3 erros) e 47 da terceira(com 3 erro))
print(metrics.classification_report(observado, previsto))
print(metrics.confusion_matrix(observado, previsto))
'''
'''
#Exemplo 3 de Naive Bayes Gaussiano
import numpy as np
from sklearn.naive_bayes import GaussianNB
from astroML.plotting import setup_text_plots
from matplotlib import pyplot as plt
from matplotlib import colors

setup_text_plots(fontsize = 8, usetex = True)
# Criando massa de dados 
np.random.seed(0) #Deixando os numeros aleatorio padrão
mu1 = [1, 1]
cov1 = 0.3 * np.eye(2)
mu2 = [5, 3]
cov2 = np.eye(2) * np.array([0.4, 0.1])
# Concatenando
X = np.concatenate([np.random.multivariate_normal(mu1, cov1, 100),
                    np.random.multivariate_normal(mu2, cov2, 100)])
y = np.zeros(200)
y[100:] = 1
# Criação do Modelo
clf = GaussianNB()
clf.fit(X, y)
# Previsões
xlim = (-1, 8)
ylim = (-1, 5)
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 71), np.linspace(ylim[0], ylim[1], 81))
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:, 1].reshape(xx.shape)
# Plot dos resultados
fig = plt.figure(figsize = (5, 3.75))
ax = fig.add_subplot(111)
ax.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.binary, zorder = 2)
ax.contour(xx, yy, Z, [0.5], colors = 'k')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.show()
'''

import numpy as np
from random import random
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pylab as pl
import matplotlib
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
# Massa de dados representando 3 classes
leopardo_features = [(random() * 5 + 8, random() * 7 + 12) for x in range(5)]
urso_features = [(random() * 4 + 3, random() * 2 + 30) for x in range(4)]
elefante_features = [(random() * 3 + 20, (random() - 0.5) * 4 + 23) for x in range(6)]
# X
x = urso_features + elefante_features + leopardo_features
# Y
y = ['urso'] * len(urso_features) + ['elefante'] * len(elefante_features) + ['leopardo'] * len(leopardo_features)

# Plot dos dados
fig, axis = plt.subplots(1, 1)
# Classe 1
urso_weight, urso_height = zip(*urso_features)
axis.plot(urso_weight, urso_height, 'ro', label = 'Ursos')
# Classe 2
elefante_weight, elefante_height = zip(*elefante_features)
axis.plot(elefante_weight, elefante_height, 'bo', label = 'Elefantes')
# Classe 3
leopardo_weight, leopardo_height = zip(*leopardo_features)
axis.plot(leopardo_weight, leopardo_height, 'yo', label = 'Leopardos')
# Eixos
axis.legend(loc = 4)
axis.set_xlabel('Peso')
axis.set_ylabel('Altura')
# Plot
plt.show()

# Criando o Modelo com os dados de treino
clf = GaussianNB()
clf.fit(x, y)
# Criando dados de teste
new_xses = [[2, 3], [3, 31], [21, 23], [12, 16]]
# Previsões
print (clf.predict(new_xses))
print (clf.predict_proba(new_xses))

def plot_classification_results(clf, X, y, title):
    # Divide o dataset em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    # Fit dos dados com o classificador
    clf.fit(X_train, y_train)
    # Cores para o gráfico
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    h = .02  # step size in the mesh
    # Plot da fronteira de decisão.
    # Usando o meshgrid do NumPy e atribuindo uma cor para cada ponto 
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    print(xx,yy)
    # Previsões
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Resultados em cada cor do plot
    Z = Z.reshape(xx.shape)
    pl.figure()
    pl.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # Plot dos pontos de dados de treino
    pl.scatter(X_train[:, 0], X_train[:, 1], c = y_train, cmap = cmap_bold)

    y_predicted = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    pl.scatter(X_test[:, 0], X_test[:, 1], c = y_predicted, alpha = 0.5, cmap = cmap_bold)
    pl.xlim(xx.min(), xx.max())
    pl.ylim(yy.min(), yy.max())
    pl.title(title)
    pl.show()
    return score

xs = np.array(x)
ys = [0] * len(urso_features) + [1] * len(elefante_features) + [2] * len(leopardo_features)

score = plot_classification_results(clf, xs, ys, "Multiclass Classification")
print ("Classification score was: %s" % score)