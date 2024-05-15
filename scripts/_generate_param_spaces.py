#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-05-16 08:21:02 (ywatanabe)"
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
    # Base parameters
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
        "n_calc": [1],
        # Model switch
        "package": ["tensorpac", "mngs"],
    }

    # Ideal Parameters
    PARAMS_GRID_IDEAL = {
        # Inputs shapes
        "batch_size": [8, 16, 32, 64],
        "n_chs": [8, 16, 32, 64],
        "n_segments": [1, 2, 4, 8, 16],
        "t_sec": [1, 2, 4, 8],
        # Signal properties
        "fs": [512, 1024],
        "pha_n_bands": [10, 30, 50, 70, 100],
        "amp_n_bands": [10, 30, 50, 70, 100],
        # Calculation options
        "n_perm": [1, 2, 4, 8, 16],
        "chunk_size": [1, 2, 4, 8],
        "fp16": [True, False],
        "in_place": [False, True],
        "trainable": [False, True],
        "device": ["cpu", "cuda"],
        "use_threads": [False, True],
        "no_grad": [True, False],
        "n_calc": [1, 2, 4, 8],
        # Model switch
        "package": ["tensorpac", "mngs"],
    }

    # Construct experimental parameter spaces
    PARAMS_GRID = PARAMS_GRID_BASE
    UPDATE_KEYS = [
        "batch_size",
        "n_chs",
        "n_segments",
        "t_sec",
        "fs",
        "pha_n_bands",
        "amp_n_bands",
        "chunk_size",
        "fp16",
        "in_place",
        "trainable",
        "device",
        "use_threads",
        "no_grad",
        "n_calc",
    ]
    UPDATE_DICT = {k: PARAMS_GRID_IDEAL[k] for k in UPDATE_KEYS}
    PARAMS_GRID.update(UPDATE_DICT)

    # Print
    print(PARAMS_GRID)
    print(f"{mngs.ml.utils.grid_search.count_grids(PARAMS_GRID):,}")

    # Saves the defined grid space
    mngs.io.save(PARAMS_GRID, "./config/PARAM_SPACES.yaml")

    # Loads the saved yaml for confirmation
    params_grid = mngs.io.load("./config/PARAM_SPACES.yaml")
    assert params_grid == PARAMS_GRID


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
