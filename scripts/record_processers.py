#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-22 23:43:27"
# Author: Yusuke Watanabe (ywata1989@gmail.com)


"""
This script does XYZ.
"""


"""
Imports
"""
import sys

import matplotlib.pyplot as plt
import mngs

"""
Config
"""
# CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""


def main():
    mngs.res.rec_procs(limit_min=60, interval_s=1, reset=False)


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
