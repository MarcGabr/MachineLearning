import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import silhouette_score
from sklearn.metrics import homogeneity_completeness_v_measure

data = pd.read_csv('5-KMeans/household_power_consumption.txt', delimiter = ';', low_memory = False)
# Remove os registros com valores NA
power_consumption = data.iloc[0:, 2:9].dropna()
print(power_consumption.head())
# Obtém os atributos e separa em datasets de treino e de teste
pc_toarray = power_consumption.values
df_treino, df_teste = train_test_split(pc_toarray, train_size = .01)
# Aplica redução de dimensionalidade
hpc = PCA(n_components = 2).fit_transform(df_treino)
# Construção do modelo
k_means = KMeans()
k_means.fit(hpc)
# Obtém os valores mínimos e máximos e organiza o shape
x_min, x_max = hpc[:, 0].min() - 5, hpc[:, 0].max() - 1
y_min, y_max = hpc[:, 1].min(), hpc[:, 1].max() + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
Z = k_means.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# Plot das áreas dos clusters
plt.figure(1)
plt.clf()
plt.imshow(Z, 
           interpolation = 'nearest', 
           extent = (xx.min(), xx.max(), yy.min(), yy.max()), 
           cmap = plt.cm.Paired, 
           aspect = 'auto', 
           origin = 'lower')
# Plot dos centróides de cada cluster
plt.plot(hpc[:, 0], hpc[:, 1], 'k.', markersize = 4)
centroids = k_means.cluster_centers_
inert = k_means.inertia_
plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x', s = 169, linewidths = 3, color = 'r', zorder = 8)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

# Determinando um range de K
k_range = range(1,14)
# Aplicando o modelo K-Means a cada valor de K
k_means_var = [KMeans(n_clusters = k).fit(hpc) for k in k_range]
# Ajustando o centróide do cluster para cada modelo
centroids = [X.cluster_centers_ for X in k_means_var]
# Calculando a distância euclidiana de cada ponto de dado para o centróide
k_euclid = [cdist(hpc, cent, 'euclidean') for cent in centroids]
dist = [np.min(ke,axis=1) for ke in k_euclid]
# Soma dos quadrados das distâncias dentro do cluster
wcss = [sum(d**2) for d in dist]
# Soma total dos quadrados
tss = sum(pdist(hpc)**2)/hpc.shape[0]
# Soma dos quadrados entre clusters
bss = tss - wcss
# Curva de Elbow
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(k_range, bss/tss*100, 'b*-')
ax.set_ylim((0,100))
plt.grid(True)
plt.xlabel('Número de Clusters')
plt.ylabel('Percentual de Variância Explicada')
plt.title('Variância Explicada x Valor de K')

# Criando um novo modelo
k_means = KMeans(n_clusters = 7)
k_means.fit(hpc)
# Obtém os valores mínimos e máximos e organiza o shape
x_min, x_max = hpc[:, 0].min() - 5, hpc[:, 0].max() - 1
y_min, y_max = hpc[:, 1].min() + 1, hpc[:, 1].max() + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
Z = k_means.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# Plot das áreas dos clusters
plt.figure(1)
plt.clf()
plt.imshow(Z, 
           interpolation = 'nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Paired,
           aspect = 'auto', 
           origin = 'lower')
# Plot dos centróides
plt.plot(hpc[:, 0], hpc[:, 1], 'k.', markersize = 4)
centroids = k_means.cluster_centers_
inert = k_means.inertia_
plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x', s = 169, linewidths = 3, color = 'r', zorder = 8)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
# Silhouette Score
labels = k_means.labels_
silhouette_score(hpc, labels, metric = 'euclidean')
