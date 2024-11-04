#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-04 11:35:14 (ywatanabe)"
# File: ./torchPAC/scripts/resource_info.py

"""
Functionality:
    * Records and saves the current machine's resource information.
Input:
    * None (gathers information automatically)
Output:
    * YAML file containing resource information
Prerequisites:
    * mngs package
    * matplotlib
"""

"""Imports"""
import sys
import matplotlib.pyplot as plt
import mngs
from typing import Dict, Any

"""Config"""
CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""
def main() -> None:
    """
    Gathers and saves the current machine's resource information.

    Returns
    -------
    None
    """
    resource_info: Dict[str, Any] = mngs.resource.gather_info()
    mngs.io.save(resource_info, "resource_info.yaml")

if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)
    main()
    mngs.gen.close(CONFIG)

# EOF
