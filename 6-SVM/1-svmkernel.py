# Importando os módulos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import matplotlib
from sklearn import svm, datasets
# Carregando o dataset iris
iris = datasets.load_iris()
# Extraindo os 2 primeiros atributos para variáveis preditoras (x) e a variável target (y)
X = iris.data[:, :2] 
y = iris.target

#Kernel Linear --
# Criamos o modelo SVC (Support Vector Classification) e então fazemos fit dos dados
svc_model = svm.SVC(kernel = 'linear').fit(X, y)
# Criando um meshgrid para o Plot
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Plot
plt.subplot(1, 1, 1)
Z = svc_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap = plt.cm.Paired, alpha = 0.5)
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC com Kernel Linear')
plt.show()

#Kernel Não Linear --
svc_model = svm.SVC(kernel = 'rbf').fit(X, y)
# Plot
plt.subplot(1, 1, 1)
Z = svc_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap = plt.cm.Paired, alpha = 0.8)
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC com Kernel RBF')
plt.show()