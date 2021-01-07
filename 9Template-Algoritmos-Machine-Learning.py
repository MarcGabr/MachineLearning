#!/usr/bin/env python
# coding: utf-8

# # <font color='blue'>Data Science Academy Big Data Real-Time Analytics com Python e Spark</font>
# 
# # <font color='blue'>Capítulo 5</font>

# ## Algoritmos de Machine Learning - Template

# Este notebook contém um template do código necessário para criar os principais algoritmos de Machine Learning.

# # Regressão Linear

# In[ ]:


# Import do módulo
from sklearn import linear_model

# Datasets de treino e de teste
x_treino = dataset_treino_variaveis_preditoras
y_treino = dataset_treino_variavel_prevista
x_teste = dataset_teste_variaveis_preditoras   

# Criando o objeto linear regression
linear = linear_model.LinearRegression()

# Treinando o modelo com dados de treino e checando o score
linear.fit(x_treino, y_treino)
linear.score(x_treino, y_treino)

# Coletando os coeficientes
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

# Previsões
valores_previstos = linear.predict(x_teste)


# # Regressão Logística

# In[ ]:


# Import do módulo
from sklearn.linear_model import LogisticRegression

# Datasets de treino e de teste
x_treino = dataset_treino_variaveis_preditoras
y_treino = dataset_treino_variavel_prevista
x_teste = dataset_teste_variaveis_preditoras   

# Criando o obketo logistic regression 
modelo = LogisticRegression()

# Treinando o modelo com dados de treino e checando o score
modelo.fit(x_treino, y_treino)
modelo.score(x_treino, y_treino)

# Coletando os coeficientes
print('Coefficient: \n', modelo.coef_)
print('Intercept: \n', modelo.intercept_)

# Previsões
valores_previstos = modelo.predict(x_teste)


# # Árvores de Decisão

# In[ ]:


# Import do módulo
from sklearn import tree

# Datasets de treino e de teste
x_treino = dataset_treino_variaveis_preditoras
y_treino = dataset_treino_variavel_prevista
x_teste = dataset_teste_variaveis_preditoras   

# Criando o objeto tree para regressão
modelo = tree.DecisionTreeRegressor() 

# Criando o objeto tree para classificação
modelo = tree.DecisionTreeClassifier() 

# Treinando o modelo com dados de treino e checando o score
modelo.fit(x_treino, y_treino)
modelo.score(x_treino, y_treino)

# Previsões
valores_previstos = modelo.predict(x_teste)


# # Naive Bayes

# In[ ]:


# Import do módulo
from sklearn.naive_bayes import GaussianNB

# Datasets de treino e de teste
x_treino = dataset_treino_variaveis_preditoras
y_treino = dataset_treino_variavel_prevista
x_teste = dataset_teste_variaveis_preditoras   

# Criando o objeto GaussianNB
modelo = GaussianNB() 

# Treinando o modelo com dados de treino
modelo.fit(x_treino, y_treino)

# Previsões
valores_previstos = modelo.predict(x_teste)


# # Support Vector Machines

# In[ ]:


# Import do módulo
from sklearn import svm

# Datasets de treino e de teste
x_treino = dataset_treino_variaveis_preditoras
y_treino = dataset_treino_variavel_prevista
x_teste = dataset_teste_variaveis_preditoras   

# Criando o objeto de classificação SVM  
modelo = svm.svc() 

# Treinando o modelo com dados de treino e checando o score
modelo.fit(x_treino, y_treino)
modelo.score(x_treino, y_treino)

# Previsões
valores_previstos = modelo.predict(x_teste)


# # K-Nearest Neighbors

# In[ ]:


# Import do módulo
from sklearn.neighbors import KNeighborsClassifier

# Datasets de treino e de teste
x_treino = dataset_treino_variaveis_preditoras
y_treino = dataset_treino_variavel_prevista
x_teste = dataset_teste_variaveis_preditoras   

# Criando o objeto de classificação KNeighbors  
KNeighborsClassifier(n_neighbors = 6) # Valor default é 5

# Treinando o modelo com dados de treino 
modelo.fit(X, y)

# Previsões
valores_previstos = modelo.predict(x_teste)


# # K-Means

# In[ ]:


# Import do módulo
from sklearn.cluster import KMeans

# Datasets de treino e de teste
x_treino = dataset_treino_variaveis_preditoras
y_treino = dataset_treino_variavel_prevista
x_teste = dataset_teste_variaveis_preditoras   

# Criando o objeto KNeighbors 
k_means = KMeans(n_clusters = 3, random_state = 0)

# Treinando o modelo com dados de treino 
modelo.fit(x_treino)

# Previsões
valores_previstos = modelo.predict(x_teste)


# # Random Forest

# In[ ]:


# Import Library
from sklearn.ensemble import RandomForestClassifier

# Datasets de treino e de teste
x_treino = dataset_treino_variaveis_preditoras
y_treino = dataset_treino_variavel_prevista
x_teste = dataset_teste_variaveis_preditoras   

# Criando o objeto Random Forest 
model= RandomForestClassifier()

# # Treinando o modelo com dados de treino 
modelo.fit(x_treino, x_teste)

# Previsões
valores_previstos = modelo.predict(x_teste)


# # Redução de Dimensionalidade

# In[ ]:


# Import do módulo
from sklearn import decomposition

# Datasets de treino e de teste
x_treino = dataset_treino_variaveis_preditoras
y_treino = dataset_treino_variavel_prevista
x_teste = dataset_teste_variaveis_preditoras   

# Create objeto PCA  
pca= decomposition.PCA(n_components = k) 

# Para Factor analysis
fa= decomposition.FactorAnalysis()

# Reduzindo a dimensão do dataset de treino usando PCA
treino_reduzido = pca.fit_transform(treino)

# Reduzindo a dimensão do dataset de teste
teste_reduzido = pca.transform(teste)


# # Gradient Boosting & AdaBoost

# In[ ]:


# Import do módulo
from sklearn.ensemble import GradientBoostingClassifier

# Datasets de treino e de teste
x_treino = dataset_treino_variaveis_preditoras
y_treino = dataset_treino_variavel_prevista
x_teste = dataset_teste_variaveis_preditoras  

# Criando o objeto Gradient Boosting 
modelo = GradientBoostingClassifier(n_estimators = 100, learning_rate = 1.0, max_depth = 1, random_state = 0)

# Treinando o modelo com dados de treino 
modelo.fit(x_treino, y_treino)

# Previsões
valores_previstos = modelo.predict(x_teste)


# # Fim

# ### Obrigado - Data Science Academy - <a href=http://facebook.com/dsacademy>facebook.com/dsacademybr</a>
