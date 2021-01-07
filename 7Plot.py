import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
'''
#LINE
#Regressão Linear
x = np.array([113, 347, 199, 371, 549, 301, 419, 579])
y = np.array([1119, 1524, 2101, 2232, 2599, 3201, 3687, 4459]) 
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
plt.plot(x, y, 'ro', color = 'red')
plt.ylabel('Preço')
plt.xlabel('Tamanho da Casa')
    #axis(xmin,xmax, ymin, ymax)
plt.axis([0,600,0,5000])
plt.plot(x, x * slope + intercept, 'b')
plt.plot()
plt.show()
'''

#BAR
# Quantidade de vendas para o Produto A
valores_produto_A = [6,7,8,4,4]
# Quantidade de vendas para o Produto B
valores_produto_B = [3,12,3,4.1,6]
# Cria eixo x para produto A e produto B com uma separação de 0.25 entre as barras
x1 =  np.arange(len(valores_produto_A))
x2 = [x + 0.25 for x in x1]
# Plota as barras
plt.bar(x1, valores_produto_A, width=0.25, label = 'Produto A', color = 'b')
plt.bar(x2, valores_produto_B, width=0.25, label = 'Produto B', color = 'y')
meses = ['Agosto','Setembro','Outubro','Novembro','Dezembro']
plt.yticks(valores_produto_A + valores_produto_B,valores_produto_A + valores_produto_B)
plt.xticks([x + 0.25 for x in range(len(valores_produto_A))], meses)
plt.legend()
plt.title("Quantidade de Vendas")
plt.show()
