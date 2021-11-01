# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 14:00:08 2021

@author: Administrator
"""

import os
os.getcwd()
os.chdir("C:/Users/Administrator/Desktop/PYTHON/K-Means Clustering")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("Mall_Customers.csv")
dataset.columns
x = dataset.iloc[:, [3, 4]].values

#use  Dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

#Training the Hierarchical Clustering model on the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)

#Visualising the clusters
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c="Red", label='Cluster1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c="Green", label='Cluster2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c="Blue", label='Cluster3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,c="Cyan", label='Cluster4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,c="Yellow", label='Cluster5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()