import os, sys
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

'''
Clusters tiles from levels based on structure. Should work on any of the sets of levels. 
Doesn't seem to be very accurate in distinguishing between levels - only slightly better 
than outright guessing. This is likely due to the nature of the features used. The tiles 
within a given level don't share much in structure, and most of the levels (especially 
above ground ones) will have entirely empty tiles, which makes it difficult to distinguish 
between them based on structure alone.

In games where tile structure is more homogenous within a given level, "tile structure" could 
be used as a meaningful metric when designing or expanding on levels. In this case, it could 
also be used as a feature when attempting to classify levels (greater homogeneity = greater 
information gain).

Benchmark formatting was taken from here: 
http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
'''

# Options
base_path = './VGLC/Super Mario Bros/Processed'
tile_width = 10
tile_height = 10
verbose = False

def get_tiles(level, tile_width, tile_height):
      level_height = len(level)
      level_width = len(level[0])
      if tile_height > level_height or tile_width > level_width:
            print('Error: Tile size larger than level size.')
            sys.exit(0)

      level_tiles = []
      for i in range(level_height - tile_height + 1):
            for j in range(level_width - tile_width + 1):
                  level_tiles.append([[symbol for symbol in line[j:j + tile_width]] 
                                              for line in level[i:i + tile_height]])
      return level_tiles

def get_features(symbol):
      return int(symbol != '-')

def bench_k_means(estimator, name, data):
      t0 = time()
      estimator.fit(data)
      print('%9s  %4.2fs  %7i  %5.3f  %5.3f  %6.3f  %5.3f  %5.3f  %10.3f' 
            % (name, (time() - t0), estimator.inertia_,
            metrics.homogeneity_score(labels, estimator.labels_),
            metrics.completeness_score(labels, estimator.labels_),
            metrics.v_measure_score(labels, estimator.labels_),
            metrics.adjusted_rand_score(labels, estimator.labels_),
            metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
            metrics.silhouette_score(data, estimator.labels_, metric='euclidean')))
      return estimator

levels = []     # (level, line, symbol)
tiles = []      # (tile, line, symbol)
tiles_flat = [] # (tile, symbol)
labels = []     # (tile)

# Parsing levels from files
for file_name in os.listdir(base_path):
      with open(base_path + "/" + file_name, 'r') as file_text:
            levels.append([[symbol for symbol in line][:-1] for line in file_text.readlines()])
if verbose:
      print('\nNumber of levels: ' + str(len(levels)))
      print('First level representation:\n' + ''.join([symbol for line in levels[0] for symbol in (line + ['\n'])]))

# Parsing tiles from levels
for i in range(len(levels)):
      level_tiles = get_tiles(levels[i], tile_width, tile_height)
      tiles += level_tiles
      labels += [i for _ in range(len(level_tiles))]
if verbose:
      print('Number of tiles in first level: ' + str(sum([label == 0 for label in labels])))
      print('Tile representation:\n' + ''.join([symbol for line in tiles[int(len(tiles) / 2)] for symbol in (line + ['\n'])]))

# Converting symbols to numbers
for tile in tiles:
      features = [get_features(symbol) for line in tile for symbol in line]
      tiles_flat.append(features)

# Table formatting and k-means calculations
num_clusters = len(levels)

if not verbose:
      print()
print("Number of clusters: %d, number of samples: %d, number of features: %d"
      % (num_clusters, len(tiles_flat), len(tiles_flat[0])))
print(75 * '-')
print('%9s  %5s  %7s  %5s  %5s  %6s  %5s  %5s  %10s' 
      % ('init', 'time', 'inertia', 'homo', 'compl', 'v-meas', 'ARI', 'AMI', 'silhouette'))

bench_k_means(KMeans(init='k-means++', n_clusters=num_clusters, n_init=10), name="k-means++", data=tiles_flat)
bench_k_means(KMeans(init='random', n_clusters=num_clusters, n_init=10), name="random", data=tiles_flat)
# - In this case the seeding of the centers is deterministic, hence we run the kmeans algorithm only once
pca = PCA(n_components=num_clusters).fit(tiles_flat)
estimator = bench_k_means(KMeans(init=pca.components_, n_clusters=num_clusters, n_init=1), name="PCA-based", data=tiles_flat)

print(75 * '-')

# Write results to files
centroids = estimator.cluster_centers_.reshape((num_clusters, tile_height, tile_width)).tolist()
for centroid_index, centroid in enumerate(centroids):
      text = '\n'.join([' '.join(['X' if num >= 0.5 else '-' for num in line]) for line in centroid])
      for tile_index, tile in enumerate(tiles):
            if estimator.labels_[tile_index] == centroid_index:
                  text += '\n\n' + '\n'.join([' '.join(['X' if get_features(symbol) >= 0.5 else '-' for symbol in line]) for line in tile])
      results = open('./k_means_' + str(centroid_index) + '.txt', 'w+')
      results.write(text)