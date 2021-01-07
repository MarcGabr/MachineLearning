#CLASSIFICADO
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier

X, y = make_hastie_10_2(random_state = 0)
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

clf = GradientBoostingClassifier(n_estimators = 100, learning_rate = 1.0, max_depth = 1, random_state = 0)

clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))     



#REGRESSOR
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor

X, y = make_friedman1(n_samples = 1200, random_state = 0, noise = 1.0)
X_train, X_test = X[:200], X[200:]
y_train, y_test = y[:200], y[200:]

est = GradientBoostingRegressor(n_estimators = 100, learning_rate = 0.1, max_depth = 1, random_state = 0, loss = 'ls')

est.fit(X_train, y_train)

print(mean_squared_error(y_test, est.predict(X_test)))   