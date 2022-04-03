# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 16:19:19 2022

@author: sheth07041990
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

path = "Wine.csv"

# Read in dataset
dataset = pd.read_csv(path)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
sc = StandardScaler()
scaled_data = sc.fit_transform(X)

# PCA
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

# Variance PCA
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals = 1)

# Scree Plot
labels = ["PC" + str(x) for x in range(1, len(per_var)+1)]
plt.bar(x = range(1, len(per_var)+1), height = per_var, tick_label = labels)
plt.ylabel("Percentage of Explained Variance")
plt.xlabel("Principal Component")
plt.title("Scree Plot")
plt.show()

# Plot PCA Data
pca_df = pd.DataFrame(pca_data, columns=labels)
plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title("My PCA Graph")
plt.xlabel("PC1 - {0}%".format(per_var[0]))
plt.ylabel("PC2 - {0}%".format(per_var[1]))
plt.show()

# Loading Scores
columns = dataset.columns[:-1]
loading_scores = pd.Series(pca.components_[0], index = columns)
sorted_loading_scores = loading_scores.abs().sort_values(ascending = False)
