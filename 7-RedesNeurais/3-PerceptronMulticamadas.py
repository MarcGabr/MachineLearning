from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()
# Descrição completa do dataset
print(cancer['DESCR'])
print(cancer['data'].shape)
X = cancer['data']
y = cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Padronização
scaler = StandardScaler()
scaler.fit(X_train)
# Aplicando a padronização aos dados
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes = (30,30,30))
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))