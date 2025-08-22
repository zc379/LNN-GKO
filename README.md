# LNN-GKO
Long-Horizon Spatiotemporal Thermal Propagation for Metallic Additive Manufacturing: A Data-Driven Geometry-Aware Liquid Neural Algorithm

The algorithms and training scripts used in this work are provided in the LNN-GKO, RNN-GKO, and Vanilla GNN code directories. The result_plot directory contains the plotting scripts for reproducing the figures and results. To generate the plots, users need to select the appropriate model checkpoint (.pth file), dataset, and corresponding layer number.

The datasets used for algorithm training and plotting are available from the Dryad repository at doi:10.5061/dryad.b5mkkwhs1.
training_dataset.7z: Used for model training.
testing_dataset.7z: Used exclusively for evaluation on unseen structures and not included in training.
Please download the datasets from the Dryad link above before running the training and result plotting scripts.
Make sure to extract the dataset files into the data/ directory before executing the scripts
