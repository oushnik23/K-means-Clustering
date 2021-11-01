# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 08:47:10 2021

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
#x=dataset[['CustomerID', 'Genre', 'Age', 'Annual Income (k$)']]
x = dataset.iloc[:, [3, 4]].values


dataset.isnull().any()
dataset.apply(lambda x:x.isnull().sum()/len(x)*100)

#Using the elbow method to find the optimal number of clusters
#WCSS is the matrix to be used to find an optimal number of cluster
from sklearn.cluster import KMeans
WCSS=[]
for i in range(1, 11):
    kmeans=KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    WCSS.append(kmeans.inertia_)
plt.plot(range(1,11),WCSS)
plt.title("elbow method")
plt.xlabel("Number of Cluster")
plt.ylabel("WCSS")

#Training the K-Means model on the dataset & create dependent variable
kmeans=KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans=kmeans.fit_predict(x)


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





