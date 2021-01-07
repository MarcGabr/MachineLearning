import numpy as np
import math

# Função de ativação Sigmoid
def sigmoid(x):     
    return 1 /(1+(math.e**-x))

# Função derivada da função Sigmoid (para cálculo do gradiente)
def deriv_sigmoid(y):
    return y * (1.0 - y)   

alpha = 0.1   

# Gerando dados aleatórios para x e y
# X será o dataset com 3 features (3 atributos)
X = np.array([  [.35,.21,.33],
                [.2,.4,.3],
                [.4,.34,.5],
                [.18,.21,16] ])                

y = np.array([[0],
        [1],
        [1],
        [0]])

# Inicializando randomicamente os vetores de pesos (serão criadas 2 camadas ocultas)
np.random.seed(1)
theta0 = 2*np.random.random((3,4)) - 1
theta1 = 2*np.random.random((4,1)) - 1

print(theta0, theta1)
# Loop for para percorrer a rede neural
# O valor 205000 especifica a quantidade de rounds de treinamento
for iter in range(205000): 
    # Etapa 1: Feedforward 
    input_layer = X
    l1 = sigmoid(np.dot(input_layer, theta0))
    l2 = sigmoid(np.dot(l1,theta1))

    # Etapa 2: Calculando o erro 
    l2_error = y - l2
    
    if (iter% 1000) == 0:
        print ("Acurácia da Rede Neural: " + str(np.mean(1-(np.abs(l2_error)))))
        
    # Etapa 3: Calculando os gradientes de forma vetorizada 
    l2_delta = alpha * (l2_error * deriv_sigmoid(l2))
    l1_error = l2_delta.dot(theta1.T)
    l1_delta = alpha * (l1_error * deriv_sigmoid(l1))

    # Etapa 4 - Atualiza os vetores de pesos
    theta1 += l1.T.dot(l2_delta)
    theta0 += input_layer.T.dot(l1_delta)