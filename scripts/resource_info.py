#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-20 16:14:49"
# Author: Yusuke Watanabe (ywata1989@gmail.com)

"""
This script records the information of the current machine.
"""

# Imports
import sys

import matplotlib.pyplot as plt
import mngs

# Config
CONFIG = mngs.gen.load_configs()

# Functions
def main():
    resource_info = mngs.res.gather_info()
    mngs.io.save(resource_info, "resource_info.yaml")


if __name__ == "__main__":
    # Start
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)

    main()

    # Close
    mngs.gen.close(CONFIG)

# EOF

"""
/home/ywatanabe/proj/entrance/torchPAC/scripts/machine_spec.py &
"""
