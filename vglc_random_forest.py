import os, sys
from time import time
import numpy as np

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

'''
Attempts to predict the level of Kid Icarus that a tile is from, using the counts of symbols in 
the tile. Adding trees doesn't seem to affect accuracy too much, but it decreases the variance 
greatly (e.g. when running the program multiple times). Increasing the tile height seems to improve 
accuracy, but with diminishing returns later on. This might be due to the amount of row structures 
that are reused between levels (such as empty rows, 3 Ts, 4 Ts, and all Ms).

The large difference in accuracy produced by varying tile sizes (~0.28 for single rows versus ~0.81 
for 16x16 tiles) shows that single columns may not be good enough to capture "player intent" in 
regards to the coop level editor.
'''

# Options
base_path = './VGLC/Kid Icarus/Processed'
tile_width = 16
tile_height = 1
test_size = 0.2
num_trees = 100
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

symbol_to_int = {'-' : 0, '#' : 1, 'D' : 2, 'H' : 3, 'M' : 4, 'T' : 5}
def get_features(tile):
      counts = [0] * len(symbol_to_int)
      for i in range(len(tile)):
            for j in range(len(tile[-1])):
                  counts[symbol_to_int[tile[i][j]]] += 1
      return counts

levels = []     # (level, line, symbol)
tiles = []      # (tile, line, symbol)
features = []   # (tile, features)
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
      features.append(get_features(tile))

# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=1267)

# Training model
classifier = RandomForestClassifier(n_estimators=num_trees)
classifier.fit(X_train, y_train)

# Testing model
if not verbose:
      print()
print("Classification report for random forest classifier:\n\n%s"
      % metrics.classification_report(y_test, classifier.predict(X_test)))