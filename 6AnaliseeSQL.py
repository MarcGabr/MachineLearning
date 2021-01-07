import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
df = pd.read_csv('taxis_bikes_nycity.csv')
#print(df.head(10))

#Transformar Object in Data
df = pd.read_csv('taxis_bikes_nycity.csv', parse_dates=['Data'])
#print(df['Data'].head())
df.set_index('Data', inplace = True)
df.plot(kind = 'area')
#plt.show()

#SQL JOIN
from datetime import datetime
start = datetime(2015,8,1)
end = datetime(2015,1,2)
start = df.index.min()
end = df.index.max()
df2 = pd.DataFrame(index= pd.date_range(start, end))
print(df.join(df2).head(10))
df.to_csv('dataframe_saved_v1.csv')
'''

#Split, Apply, Combine
df = pd.read_csv('dataframe_saved_v2.csv', parse_dates=['Data'], usecols = list(range(0,6)))
#print(df.Distancia[df.Distancia > 0].value_counts().index[0])

#Acessa Coluno Minutos preenche os valores Nan com 0, Soma acumulativa e divide por 60.
(df.Minutos.fillna(0).cumsum() / 60).plot()
#plt.show()

#Acessa a data e cria coluna o dia da semana com strftime
df['DiadaSemana'] = df.Data.map(lambda x: x.strftime('%A'))
df[df.Distancia > 0].DiadaSemana.value_counts().plot(kind = 'bar')
#plt.show()
'''
#Cortar uma coluna
df.drop('Segundos', axis=1)
#Cortar uma linha
df.drop(1, axis=0)
#Ver colunas Cortadas
df.pop('Segundos')

#Imprimir dados e suas informações
for dia in df.DiadaSemana.unique():
    print(dia)
    print(df[df.DiadaSemana == dia].head(3))
'''
dias = df.groupby('DiadaSemana')
print(dias.count())
print(dias.get_group('Friday'))