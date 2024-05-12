#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-23 19:24:00"
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
from mngs.general import timeout

"""
Config
"""
# CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""


def main(interval_s, reset):
    mngs.res.rec_procs(
        limit_min=60 * 24, interval_s=interval_s, reset=reset, verbose=False
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--interval_s",
        "-i",
        type=float,
        default=0.5,
        help="Interval in seconds.",
    )
    parser.add_argument(
        "--reset", "-r", action="store_true", default=False, help="Reset flag."
    )
    args = parser.parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main(args.interval_s, args.reset)
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
