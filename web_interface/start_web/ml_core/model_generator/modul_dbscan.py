from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv("input/17_02_part1.csv")
#df = df.drop(['Time'], axis=1)
df_work = df[['Time','AxisCurrent_1']]

scaler = StandardScaler()
df_work = scaler.fit_transform(df_work)
#print(df_work)

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.5, min_samples=2).fit(df_work)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Cluster's size
counts = np.bincount(labels[labels>=0])
top_labels = np.argsort(-counts)
print("Clusters by size:")
print(counts[top_labels])

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# Array for anomalies
abnormal_data = df_work[0]

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = df_work[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = df_work[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

    # Write abnormal values
    if k == -1:
        abnormal_data = xy

print(scaler.inverse_transform(abnormal_data))
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.legend()
plt.show()
