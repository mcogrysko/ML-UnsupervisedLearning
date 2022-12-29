#Mike Ogrysko
#CS 737 Machine Learning
#Unsupervised learning using synthetic clustering dataset

#Rough feature ranges
#K-means clustering for anomolies
#DBSCAN clustering to find anomalies
#Decision tree classifier to model the species and visualize the model decision tree - clean data
#Comparison without cleaning data

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from IPython.display import Image
from subprocess import run, PIPE, call
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# locate and load the data file
dfOrg = pd.read_csv('./synthetic_clustering_dataset.csv')

# Sanity
print(f'#rows={len(dfOrg)} #columns={len(dfOrg.columns)}')
dfOrg.head()

#try plotting the features
plt.scatter(dfOrg.f1, dfOrg.f2)
plt.xlabel(r'f1', horizontalalignment='right', x=1.0)
plt.ylabel(r'f2', horizontalalignment='right', y=1.0)
plt.show()

#Finding rough feature ranges

print('f1 min: ', dfOrg.f1.min())
print('f1 max: ', dfOrg.f1.max())
print('f2 min: ', dfOrg.f2.min())
print('f2 max: ', dfOrg.f2.max())

#species 1
result_index_f2_sp1_top = dfOrg['f1'].sub(-.4).abs().idxmin()
result_index_f2_sp1_bottom = dfOrg['f2'].sub(-2).abs().idxmin()
result_index_f1_sp1_left = dfOrg['f2'].sub(.43).abs().idxmin()
result_index_f1_sp1_right = dfOrg['f1'].sub(-3).abs().idxmin()
#species 2
result_index_f2_sp2_top = dfOrg['f2'].sub(.47).abs().idxmin()
result_index_f2_sp2_bottom = dfOrg['f2'].sub(-1.6).abs().idxmin()
result_index_f1_sp2_left = dfOrg['f1'].sub(-.36).abs().idxmin()
result_index_f1_sp2_right = dfOrg['f1'].sub(2).abs().idxmin()
#species 3
result_index_f2_sp3_top = dfOrg['f2'].sub(3).abs().idxmin()
result_index_f2_sp3_bottom = dfOrg['f2'].sub(.5).abs().idxmin()
result_index_f1_sp3_left = dfOrg['f1'].sub(-.38).abs().idxmin()
result_index_f1_sp3_right = dfOrg['f1'].sub(1.75).abs().idxmin()
#print
print(dfOrg.iloc[result_index_f2_sp1_top])
print(dfOrg.iloc[result_index_f2_sp1_bottom])
print(dfOrg.iloc[result_index_f1_sp1_left])
print(dfOrg.iloc[result_index_f1_sp1_right])
print(dfOrg.iloc[result_index_f2_sp2_top])
print(dfOrg.iloc[result_index_f2_sp2_bottom])
print(dfOrg.iloc[result_index_f1_sp2_left])
print(dfOrg.iloc[result_index_f1_sp2_right])
print(dfOrg.iloc[result_index_f2_sp3_top])
print(dfOrg.iloc[result_index_f2_sp3_bottom])
print(dfOrg.iloc[result_index_f1_sp3_left])
print(dfOrg.iloc[result_index_f1_sp3_right])

#Try plotting the features
plt.scatter(dfOrg.f1, dfOrg.f2)
#Species 1
plt.scatter(dfOrg.f1[570],dfOrg.f2[570], color='pink')#top
plt.scatter(dfOrg.f1[16],dfOrg.f2[16], color='pink')#bottom
plt.scatter(dfOrg.f1[531],dfOrg.f2[531], color='pink')#left
plt.scatter(dfOrg.f1[146],dfOrg.f2[146], color='pink')#right
#Species 2
plt.scatter(dfOrg.f1[663],dfOrg.f2[663], color='orange')#right
plt.scatter(dfOrg.f1[684],dfOrg.f2[684], color='orange')#left
plt.scatter(dfOrg.f1[337],dfOrg.f2[337], color='orange')#bottom
plt.scatter(dfOrg.f1[82],dfOrg.f2[82], color='orange')#top
#Species 3
plt.scatter(dfOrg.f1[389],dfOrg.f2[389], color='red')#top
plt.scatter(dfOrg.f1[343],dfOrg.f2[343], color='red')#left
plt.scatter(dfOrg.f1[110],dfOrg.f2[110], color='red')#right
plt.scatter(dfOrg.f1[95],dfOrg.f2[95], color='red')#bottom
#Line
plt.plot([-3, 2], [.5, .5], '-', color = 'r')
plt.plot([-.4, -.4], [-2, 3], '-', color = 'r')
plt.xlabel(r'f1', horizontalalignment='right', x=1.0)
plt.ylabel(r'f2', horizontalalignment='right', y=1.0)
plt.show()

print('Species 1 has feature 1 in the range of [-2.274474,-0.422276]')
print('Species 1 has feature 2 in the range of [-1.823801,0.435205]')

print('Species 2 has feature 1 in the range of [-0.374479,1.870438]')
print('Species 2 has feature 2 in the range of [-1.624734,0.450459]')

print('Species 3 has feature 1 in the range of [-0.374479,1.870438]')
print('Species 3 has feature 2 in the range of [0.501611,2.245794]')

#K-means clustering to find anomalies

#get X
X = dfOrg.values

#kmeans
km = KMeans(n_clusters=3, init='random',n_init=10, max_iter=300, random_state=42)
clusters = km.fit_predict(X)

#identify centroids
centroids = km.cluster_centers_

#identify points
points = np.empty((0,len(X[0])),float)

#identify distances
distances = np.empty((0,len(X[0])),float)
for i, center in enumerate(centroids):
    distances = np.append(distances, cdist([center],X[clusters == i],'euclidean'))
    points = np.append(points, X[clusters == i], axis=0)

#set percentile
percentile = 95
outliers3 = points[np.where(distances > np.percentile(distances,percentile))]

#plot
plt.scatter(*zip(*X), c=clusters)
#centroids are aqua
plt.scatter(*zip(*centroids),marker='o',facecolor='None',edgecolor='aqua',s=70)
#outliers are red
plt.scatter(*zip(*outliers3),marker='o',facecolor='None',edgecolor='r',s=70)
plt.xlabel(r'f1', horizontalalignment='right', x=1.0)
plt.ylabel(r'f2', horizontalalignment='right', y=1.0)
plt.show()

#print centroids
print('Centroids:\n', centroids)

#print outliers
print('Outliers:\n', outliers3)

#DBSCAN clustering to find anomalies

#dbscan
dbs = DBSCAN(eps = .2, min_samples = 5)
dbs.fit(X)
species = dbs.labels_

#plot
plt.scatter(X[:,0],X[:,1], c = species)
outliers4 = X[dbs.labels_ == -1]
#outliers are red
plt.scatter(*zip(*outliers4),marker='o',facecolor='None',edgecolor='r',s=70)
plt.xlabel(r'f1', horizontalalignment='right', x=1.0)
plt.ylabel(r'f2', horizontalalignment='right', y=1.0)
plt.show()

#print outliers
print('Outliers:\n', outliers4)

#Decision tree classifier to model the species and visualize the model decision tree - clean data

#use dbscan outliers
print('# of DBSCAN outliers: ', len(outliers4))

dfOrg5 = dfOrg.copy()

dfOrg5.head()

#filter out outliers
dfOrg5 = dfOrg5[~dfOrg5.f1.isin(outliers4[:,0])]
dfOrg5.shape

#remove outliers (-1)
species5 = species[species != -1]
#get species count
print('Species count: ', len(species5))

#add species to df
dfOrg5['species'] = species5

#check scatter
plt.scatter(dfOrg5.f1,dfOrg5.f2, c = species5)
plt.xlabel(r'f1', horizontalalignment='right', x=1.0)
plt.ylabel(r'f2', horizontalalignment='right', y=1.0)
plt.show()

#set X,y
X = dfOrg5.drop(['species'], axis=1).values
y = dfOrg5['species'].values
X.shape,y.shape

dt = DecisionTreeClassifier()
model = dt.fit(X, y)

#display dt
plt.figure()
tree.plot_tree(model, filled=True)
plt.title("Decision tree with cleaned data")
plt.show()

#Comparison without cleaning data

dfOrg6 = dfOrg.copy()

dfOrg6['species'] = species
dfOrg6.shape

#set X,y
X = dfOrg6.drop(['species'], axis=1).values
y = dfOrg6['species'].values
X.shape,y.shape

dt6 = DecisionTreeClassifier()
model6 = dt6.fit(X, y)

#display dt
plt.figure(figsize=(24,12))
tree.plot_tree(model6, filled=True)
plt.title("Decision tree without cleaned data")
plt.show()






















