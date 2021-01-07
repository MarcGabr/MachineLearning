#Classificado de Noticias
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# Definindo as categorias 
# (usando apenas 4 de um total de 20 disponível para que o processo de classificação seja mais rápido)
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
# Treinamento(Criou um subset com somente 4 categorias)
twenty_train = fetch_20newsgroups(subset = 'train', categories = categories, shuffle = True, random_state = 42)

# Tokenizing
count_vect = CountVectorizer() #Quebra as frases para palavras
X_train_counts = count_vect.fit_transform(twenty_train.data)
count_vect.vocabulary_.get(u'algorithm')
print(X_train_counts.shape)
# De ocorrências a frequências - Term Frequency times Inverse Document Frequency (Tfidf)
tf_transformer = TfidfTransformer(use_idf = False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

# Criando o modelo Multinomial
clf = MultinomialNB().fit(X_train_tf, twenty_train.target)
# Previsões
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

