'''
#Mapeando lista para funções
def calc(num):
    return num ** 3
array1 = range(0,15)
listacubo = map(calc, array1)
print(list(listacubo))
print(list(map(float, array1[0:5])))

#Filtra Elementos 
def Par(num):
    return num % 2 == 0
array1 = range(0,15)
print(list(map(Par,array1)))
print(list(filter(Par,array1)))

#Redução de elementos
from functools import reduce
print(reduce((lambda x, y: x + y), [1,2,3,4]))
'''
#List Comprehentio(Filter e Map)
array1 = range(0,15)
print([item ** 2 for item in array1])
print([item ** 2 for item in array1 if item % 2 == 0])

#Lambda
array1 = range(0,15)
print(list(map(lambda num: num**2, array1)))
print(list(map(lambda a: a**2, filter(lambda a: a % 2 == 0, array1))))

a = [1]
for i in range(1,7):
    a.append(a[len(a)-1]* 3)
