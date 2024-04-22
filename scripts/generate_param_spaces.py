#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-21 22:53:49"
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
    PARAMS_GRID = {
        "batch_size": [2**i for i in [3, 4, 5, 6]],
        "n_chs": [2**i for i in [3, 4, 5, 6]],
        "seq_len": [2**i for i in range(8, 13)],
        "fs": [2**i for i in range(7, 10)],
        "n_segments": [2**i for i in range(5)],
        "n_bands_pha": [2**i for i in range(7)],
        "n_bands_amp": [2**i for i in range(7)],
        "precision": ["fp16", "fp32"],
        "device": ["cpu", "cuda"],
        "package": ["tensorpac", "mngs"],
    }
    print(PARAMS_GRID)
    print(f"{mngs.ml.utils.grid_search.count_grids(PARAMS_GRID):,}")

    # Saves the defined grid space
    mngs.io.save(PARAMS_GRID, "./config/PARAM_SPACES.yaml")

    # Loads
    params_grid = mngs.io.load("./config/PARAM_SPACES.yaml")
    assert params_grid == PARAMS_GRID

    # Example of using the generator
    for param_dict in mngs.ml.utils.grid_search.yield_grids(PARAMS_GRID):
        print(param_dict)


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
