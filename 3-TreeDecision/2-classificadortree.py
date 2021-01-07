'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
irisData = pd.read_csv('3-TreeDecision/iris_data.csv')
features = irisData[["SepalLength","SepalWidth","PetalLength","PetalWidth"]]
targetVariables = irisData.Class
featureTrain, featureTest, targetTrain, targetTest = train_test_split(features, 
                                                                      targetVariables, 
                                                                      test_size = .2)
#Árvore de Decisão
clf = DecisionTreeClassifier()
#Criação do Modelo
modelo = clf.fit(featureTrain, targetTrain)
#Testa as variaveis de predição
previsoes = modelo.predict(featureTest)
#Compara a previsão com o real valor
print (confusion_matrix(targetTest, previsoes))
#Compara a previsão com o real valor. Acuracia
print (accuracy_score(targetTest, previsoes))
'''

'''
#Random Forest Classifier I
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
digitos = load_digits()
#Deixando os dados na mesma escala
data = scale(digitos.data)
# Obtém número de observações(linhas) e número de atributos(colunas)
n_observ, n_features = data.shape
#Q
n_digits = len(np.unique(digitos.target))
#Atribuindo os Labels
labels = digitos.target
#Treinando com 10 estimadores
clf = RandomForestClassifier(n_estimators  = 10)
clf = clf.fit(data, labels)
print(clf.score(data, labels))
# Extraindo a importância
importances = clf.feature_importances_
indices = np.argsort(importances)
# Obtém os índices
ind=[]
for i in indices:
    ind.append(labels[i])
# Plot da Importância dos Atributos
plt.figure(1)
plt.title('Importância dos Atributos')
plt.barh(range(len(indices)), importances[indices], color = 'b', align = 'center')
plt.yticks(range(len(indices)),ind)
plt.xlabel('Importância Relativa')
plt.show()
'''

#Random Forest Classifier II    
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from treeinterpreter import treeinterpreter as ti
from sklearn.datasets import load_iris

iris = load_iris()
# Cria o classificador
rf = RandomForestClassifier(max_depth = 5)
# Obtém os índices a partir do comprimento da variável targetr
idx = list(range(len(iris.target)))
# Randomiza o ínidce
np.random.shuffle(idx)
# Cria o modelo
rf.fit(iris.data[idx][:100], iris.target[idx][:100])
# Obtém as instâncias (exemplos ou observações) e retorna as probabilidades
instance = iris.data[idx][100:101]
#Propabilidade de se classificar o Modelo com as 3 classe. Quando atinge o 100% de uma delas, mas não isso é overfiting
print(rf.predict_proba(instance))

prediction, bias, contributions = ti.predict(rf, instance)
print ("Previsões", prediction)
print ("Contribuição dos Atributos:")
for item, feature in zip(contributions[0], iris.feature_names):
    print (feature, item)