#Map, Filter e Lambda igual o do 3
import numpy as np
array1 = np.arange(15)
'''
print(array1.mean())
print(array1.sum())
print(array1.min())
print(array1.max())
print(array1.std())

#MenosEficiente
print([x * x for x in array1])
#MaisEficiente
print(array1[array1 % 2 == 0])
#Anexando
array1 = np.array([x ** 2 for x in range(15)])

#Concatenar
array2 = np.arange(4)
array3 = np.concatenate((array1,array2))
print(array3)

#Join
array1 = np.ones((3,3))
array2 = np.zeros((3,3))
print(np.vstack((array1,array2)))
print(np.hstack((array1,array2)))
array1 = np.array([0,1,2])
array2 = np.array([3,4,5])
array3 = np.array([6,7,8])
print(np.column_stack((array1,array2,array3)))
print(np.row_stack((array1,array2,array3)))

#Split
array1 = np.arange(16).reshape((4,4))
print(array1)
[array2, array3] = np.hsplit(array1,2)
print(array2,array3)
[array2, array3] = np.vsplit(array1,2)
print(array2,array3)

#Matrizes
mat1 = np.matrix("1,2,3; 4,5,6")
mat1 = np.matrix([[1,2,3],[4,5,6]])
print(mat1)
print(mat1[:,0])
#Matriz Esparsa
import scipy.sparse
linhas = np.array([0,1,2])
colunas = np.array([1,2,4])
valores = np.array([10,20,30])
mat1 = scipy.sparse.coo_matrix((valores, (linhas, colunas)))
print(mat1)
print(mat1.todense())

#Multiplicação
a = np.array([[1,2],[4,5]])
print(a * a)
A = np.matrix(a)
print(A*A)
#Vetorização Usar para fazer um função receber um array
def calc(num):
    if num < 10:
        return num ** 3
    else:
        return num ** 2
array1 = np.random.randint(0,50,20)
print(array1)
v_calc = np.vectorize(calc)
print(v_calc(array1))
'''