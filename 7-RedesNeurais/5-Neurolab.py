import neurolab as nl
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Criando datasets de treino
x = np.linspace(-10,10, 60)
y = np.cos(x) * 0.9
size = len(x)
x_train = x.reshape(size,1)
y_train = y.reshape(size,1)

# Criando 4 redes com 0, 1, 2, e 3 camadas e randomizando a inicialização
d = [[1,1],[45,1],[45,45,1],[45,45,45,1]]

for i in range(4):
    net = nl.net.newff([[-10, 10]],d[i])
    train_net = nl.train.train_gd(net, x_train, y_train, epochs = 1000, show = 100)
    outp = net.sim(x_train)
    
    # Plot 
    plt.subplot(2, 1, 1)
    plt.plot(train_net)
    plt.xlabel('Epochs')
    plt.ylabel('Squared error')
    x2 = np.linspace(-10.0,10.0,150)
    y2 = net.sim(x2.reshape(x2.size,1)).reshape(x2.size)
    y3 = outp.reshape(size)
    plt.subplot(2, 1, 2)

    plt.suptitle([i ,'Camadas Ocultas'])
    plt.plot(x2, y2, '-',x , y, '.', x, y3, 'p')
    plt.legend(['Y previsto', 'Y observado'])
    plt.show()