# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 11:05:59 2021

@author: Administrator
"""

import os
os.getcwd()
os.chdir("C:/Users/Administrator/Desktop/PYTHON/K-Means Clustering")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv("Mall_Customers.csv")
dataset.columns

#Changinf the column name
dataset.rename(columns={'Annual Income (k$)':'income','Spending Score (1-100)':'score'}, inplace=True)
dataset.shape

#Statistics
dataset.describe()

#Plot
sns.pairplot(dataset[['Age', 'income','score']])

from sklearn.cluster import KMeans
import sklearn.cluster as cluster
KMeans=cluster.KMeans(n_clusters=5)
KMeans=KMeans.fit(dataset[['income','score']])
KMeans.cluster_centers_
dataset['income_cluster']=KMeans.labels_
print(dataset)
dataset['income_cluster'].value_counts()
sns.scatterplot(x='income',y='score', hue='income_cluster',data=dataset)



KMeans=cluster.KMeans(n_clusters=2)
KMeans=KMeans.fit(dataset[['Age','score']])
KMeans.cluster_centers_
dataset['age_cluster']=KMeans.labels_
dataset['age_cluster'].value_counts()
print(dataset)

sns.scatterplot(x='Age',y='score',hue='age_cluster',data=dataset)

from sklearn.cluster import KMeans
WCSS=[]
for i in range(1,11):
    km=KMeans(n_clusters=i)
    km.fit(dataset[['income','score']])
    WCSS.append(km.inertia_)
plt.plot(range(1,11),WCSS)
plt.xlabel("no of cluster")
plt.ylabel("WCSS")
plt.legend()