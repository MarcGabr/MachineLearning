import pandas as pd
import numpy as np
'''
#Series
#Numero Random
np.random.seed(100)
Serie1 = pd.Series(np.random.rand(10))
print(Serie1)
#Calendario
import calendar as cal 
meses = [cal.month_name[i] for i in np.arange(1,13) ]
meses_Series1 = pd.Series(np.arange(1,13), index = meses)
print(meses)
print(meses_Series1)

#DataFrames
stock = {'B': pd.Series([111,222,333], index = ['Test1', 'Teste2', 'Teste3']),
'C': pd.Series([112,223,444], index = ['Test1', 'Teste2', 'Teste3']),
'A': pd.Series([11,22,33], index = ['Test1', 'Teste2', 'Teste3']),
}
datastock = pd.DataFrame(stock)
print(datastock)
#Ordenando
datastock = pd.DataFrame(stock, index = ['Test1','Teste2'], columns = ['A', 'C'])
print(datastock)

#Dicionarios
Qd = {'BR' : {'AAA': 11, 'BBB': 12, 'CCC': 1},
'USA' : {'AAA': 13, 'BBB': 12, 'CCC': 22},
'CHI' : {'AAA': 12, 'BBB': 14, 'DDD': 4}}
Qdframe = pd.DataFrame.from_dict(Qd)
print(Qdframe)
print(Qdframe[['CHI', 'USA']])
#Slicing
print(Qdframe[2:])
print(Qdframe.loc['AAA'])
print(Qdframe.loc['AAA', 'CHI'])
print(Qdframe.loc['AAA'] > 11)
#Delete
del Qdframe['CHI']
#ADD
Qdframe.insert(2, 'China', (12,11,0,0))
print(Qdframe)
#Descrever
print(Qdframe.describe())
'''

#Exemplo Ler e usar
cidadesD = pd.read_csv('cidades_digitais.csv', sep = ';', encoding = 'latin-1')
del cidadesD['IBGE']
#------ Agrupamento
cidadesUF = cidadesD.groupby(['UF', 'STATUS'])
#print(cidadesUF.groups)
ordenaUF = cidadesUF.size()
#print(ordenaUF.sort_values(ascending = False))

pd.set_option("max_rows", None)
#------ Mudar Indice
cidadesUF2 = cidadesD.set_index('STATUS')
print(cidadesUF2)

print('-----------------------------------------------')
cidadesUF2 = cidadesUF2.groupby(level ='STATUS')
print(cidadesUF2.head())