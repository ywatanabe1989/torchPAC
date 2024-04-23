#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-23 15:33:05"
# Author: Yusuke Watanabe (ywata1989@gmail.com)

"""
This script generates parameter spaces in this project.
"""

# Imports
import sys

import matplotlib.pyplot as plt
import mngs


# Functions
def main():
    # Define parameter spaces
    PARAMS_GRID_BASE = {
        # Inputs shapes
        "batch_size": [1],
        "n_chs": [1],
        "n_segments": [1],
        "t_sec": [1],
        # Signal properties
        "fs": [512],
        "pha_n_bands": [10],
        "amp_n_bands": [10],
        # Calculation options
        "n_perm": [None],
        "chunk_size": [1],
        "fp16": [True],
        "no_grad": [True],
        "in_place": [True],
        "trainable": [False],
        "device": ["cuda"],
        "use_threads": [False],
        # Model switch
        "package": ["tensorpac", "mngs"],
    }
    PARAMS_GRID = PARAMS_GRID_BASE

    # PARAMS_GRID_FINAL = {
    #     # Inputs shapes
    #     "batch_size": [2**i for i in [3, 4, 5, 6]],
    #     "n_chs": [2**i for i in [3, 4, 5, 6]],
    #     "n_segments": [2**i for i in range(5)],
    #     "t_sec": [1, 2, 4, 8],
    #     # Signal properties
    #     "fs": [512],
    #     "pha_n_bands": [10, 30, 50, 70, 100],
    #     "amp_n_bands": [10, 30, 50, 70, 100],
    #     # Calculation options
    #     "n_perm": [None],
    #     "chunk_size": [1, 2, 4, 8],
    #     "fp16": [True, False],
    #     "in_place": [False, True],
    #     "trainable": [False, True],
    #     "device": ["cpu", "cuda"],
    #     "use_threads": [False, True],
    #     "no_grad": [True, False],
    #     # Model switch
    #     "package": ["tensorpac", "mngs"],
    # }

    print(PARAMS_GRID)
    print(f"{mngs.ml.utils.grid_search.count_grids(PARAMS_GRID):,}")

    # Saves the defined grid space
    mngs.io.save(PARAMS_GRID, "./config/PARAM_SPACES.yaml")

    # Loads
    params_grid = mngs.io.load("./config/PARAM_SPACES.yaml")
    assert params_grid == PARAMS_GRID

    # # Example of using the generator
    # for param_dict in mngs.ml.utils.grid_search.yield_grids(PARAMS_GRID):
    #     print(param_dict)


if __name__ == "__main__":
    # Start
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )

    main()

    # Close
    mngs.gen.close(CONFIG, verbose=False, notify=False)


# EOF

"""
/home/ywatanabe/proj/entrance/torchPAC/scripts/generate_param_spaces.py
"""
