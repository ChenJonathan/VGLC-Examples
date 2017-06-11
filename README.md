# VGLC Examples
---

Some introductory-level machine learning projects that make use of the Video Game Level Corpus.

## Usage
---

The Video Game Level Corpus used for these examples can be found [here](https://github.com/TheVGLC/TheVGLC).

The entire corpus should be contained in the folder called "VGLC", and the Python files should be run in the same directory as the "VGLC" folder.

Three example projects are included - each within a separate Python file:

1. `vglc_k_means.py`: Splits a set of levels into tiles and attempts to cluster the tiles into levels based on tile structure (occupied vs. non-occupied spaces). A separate file is generated for each cluster, named `k_means_<cluster_num>.txt`, which contains the centroid as the first entry and all other tiles that fall within the cluster as subsequent entries. Occupied spaces are represented as " **X** ", while non-occupied spaces are represented as " **-** ".
2. `vglc_random_forest.py`: Attempts to predict the level of Kid Icarus that a tile is from using the numbers of different objects in the tile.
3. `vglc_autoencoder.py`: Trains an autoencoder using a set of levels to perform various tasks. Reconstruction mode encodes and decodes a single level before displaying the result. Recognition mode attempts to classify fixed-size windows of levels by the levels they came from based on the windows' encodings. Repair mode trains the autoencoder with noisy input before attempting to remove the noise from a given noisy level.

Each file has various options that can be set within the file.