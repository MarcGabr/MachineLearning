import pandas as pd
import numpy as np
import string 
import matplotlib.pyplot as plt
'''
#Array to Series
Serie1 = pd.Series(np.arange(26))
Icase = list(string.ascii_lowercase)
Ucase = list(string.ascii_uppercase)
Serie1.index = Icase
print(Serie1['f':'r'])

np.random.seed(784)
array1 = np.random.randint(1,20,26)
dados = pd.Series(array1)
print(dados)

#Valores NaN
array1 = np.array([1,2,3, np.nan])
Serie1 = pd.Series([1,2,3, np.nan])
#Concat 
Serie1 = pd.Series(np.arange(3))
Serie2 = pd.Series(np.arange(3))
Junto = pd.concat([Serie1,Serie2])
Junto.index = range(Junto.count())
print(Junto)
#Evitando NaN
print(Junto.reindex([0,2,11,12], fill_value = 0))
newJunto = Junto.reindex([0,2,11,12])
print(newJunto.ffill())
print(newJunto.bfill())
print(newJunto.fillna(12))

#Copiar
Serie1 = pd.Series([1,2,3, np.nan])
Serie2 = Serie1.copy()
#Mapeando
print(Serie1.map(lambda x: x ** 2))
print(Serie1.map({1:2,2:3,3:12}))

#Indexação
dict = {
    'Coluna1' : [1,2,3,4],'Coluna2' : [3,4,5,8],'Coluna3' : [6,7,8,9],'Coluna4' : [11,22,32,42],
}
df = pd.DataFrame(dict)
df.insert(0,'ID', [1,2,3,4])
df2 = pd.DataFrame(df.set_index('ID'))
print(df2)

Serie1 = pd.Series(np.arange(26))
Icase = list(string.ascii_lowercase)
Ucase = list(string.ascii_uppercase)
alfabeto = pd.DataFrame([Icase, Ucase, Serie1])
alfabeto = alfabeto.T
'''