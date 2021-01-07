import numpy as np
#'''
array1 = np.array([[10,20,30,40], [10,20,30,41]])
print(array1)

print(array1.shape) #Quantidade em cada Dimensão
print(array1.ndim) #Dimensão
print(array1.max()) # Maior elemento

#Criando com Range
array2 = np.arange(15)
print(array2)
array3 = np.arange(0,15,5)
print(array3)

#Dados Linspace
array4 = np.linspace(0, 3, 4)
print(array4)

#Criando Matrizes
#Com 1
array5 = np.ones((2, 3, 4))
print(array5)
#Diagonal com 1
array5 = np.eye(4)
print(array5)
#Diagonal com numeros
array5 = np.diag((2,1,4,6))
print(array5)

#ValoresRandomicos
array5 = np.random.rand(5)
print(array5)
#Vazia
array5 = np.empty((1,2))
print(array5)

#Duplicar a Matriz
array1 = np.array([[10,20,30,40], [10,20,30,41]])
array1 = np.tile(array1,2)
print(array1)
array1 = np.tile(array1,(2,2))
print(array1)

#Datatype
array1 = np.array([5,-6,2,3], dtype = 'float')
print(array1.dtype)

#Operações Aritimeticas
array1 = np.array([[10,20,30,40], [10,20,30,41]]) * 3
print(array1)
array2 = np.array([[10,20,30,40], [10,20,30,41]])
print(array1-array2)

#Slicing(Facada)
array1 = np.array([10,20,30,40])
#InverterArray
print(array1[::-1])

#Conrtando
array2 = np.array([[10,20,30], [10,20,30],[1,2,3]])
print(array2)
print(array2[:2, :2])
print(array2[:, 2:])
print(array2[1,:2])

#Transposição
array2 = np.array([[10,20,30], [10,20,30],[1,2,3]])
print(array2.T)

#Bi pra Uni
array2 = np.array([[10,20,30], [10,20,30],[1,2,3]])
print(array2.ravel())

#Uni para Bi
array5 = np.arange(1,16)
print(array5)
print(array5.reshape(3,5))

#Operadores Logicos
array3 = np.random.randint(1,10, size = (4,4))
print(array3)
#Resto
print(np.any((array3 % 7)==0))
#Maiorque
print(np.all(array3 < 11))

#'''

