# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 07:17:34 2022

@author: MS104124
"""

########## Correlation Matrix in Python using Pandas & Seaborn ###################

# Import Packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Read in Dataset
df = pd.read_csv("")

# Correlation Matrix
df.corr()

# Create Basic Heatmap or Correlation Matrix using Seaborn
sns.heatmap(df.corr())

# Increase the size of the heatmap (Default Colormap)
#vmin, vmax — set the range of values that serve as the basis for the colormap
#cmap — sets the specific colormap we want to use (check out the library of a wild range of color palettes here)
#center — takes a float to center the colormap; if no cmap specified, will change the colors in the default colormap; if set to True — it changes all the colors of the colormap to blues
#annot — when set to True, the correlation values become visible on the colored cells
#cbar — when set to False, the colorbar (that serves as a legend) disappears
plt.figure(figsize=(16, 6))
# Store heatmap object in a variable to easily access it when you want to include more features (such as title).
# Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap.set_title("Correlation Heatmap", fontdict={"fontsize":12}, pad=12)

# Diverging Color Palette
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title("Correlation Heatmap", fontdict={"fontsize":18}, pad=12);
# save heatmap as .png file
# dpi - sets the resolution of the saved image in dots/inches
# bbox_inches - when set to 'tight' - does not allow the labels to be cropped
plt.savefig("heatmap.png", dpi=300, bbox_inches='tight')

# Triangle Correlation Heatmap
plt.figure(figsize=(16, 6))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))
heatmap = sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title("Triangle Correlation Heatmap", fontdict={"fontsize":18}, pad=16)

# Correlation of Independent Variables with the Dependent Variable
plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(df.corr()[["Independent Variable"]].sort_values(by="Independent Variable", ascending=False), vmin=-1, vmax=1, annot=True, cmap="BrBG")
heatmap.set_title("Features Correlating with Sales Price", fontdict={"fontsize":18}, pad=16);

