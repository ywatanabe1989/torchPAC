#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-22 00:57:35"
# Author: Yusuke Watanabe (ywata1989@gmail.com)

"""
This script does XYZ.
"""

import os
import subprocess
import sys

# Functions
import time

import matplotlib.pyplot as plt

# Imports
import mngs
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

# # Config
# CONFIG = mngs.gen.load_configs()


# Function to measure CPU and RAM usage
def start_monitoring():
    p = multiprocessing.Process(target=get_usage)
    p.start()
    return p


def stop_monitoring(process):
    process.terminate()


def _get_cpu_usage(process=os.getpid()):
    cpu_usage_perc = psutil.cpu_percent()
    ram_usage_gb = (
        psutil.virtual_memory().percent
        / 100
        * psutil.virtual_memory().total
        / (1024**3)
    )
    return cpu_usage_perc, ram_usage_gb


def _get_gpu_usage():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,nounits,noheader",
        ],
        capture_output=True,
        text=True,
    )
    gpu_usage_perc, _vram_usage_mib = result.stdout.strip().split(",")
    vram_usage_gb = float(_vram_usage_mib) / 1024
    return float(gpu_usage_perc), float(vram_usage_gb)


def profile_cpu_gpu():
    cpu_usage_perc, ram_usage_gb = _get_cpu_usage()
    gpu_usage_perc, vram_usage_gb = _get_gpu_usage()

    df = pd.DataFrame(
        pd.Series(
            {
                "CPU [%]": cpu_usage_perc,
                "RAM [GiB]": ram_usage_gb,
                "GPU [%]": gpu_usage_perc,
                "VRAM [GiB]": vram_usage_gb,
            }
        )
    ).round(1)
    return df


if __name__ == "__main__":
    # Start
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )

    # Example usage
    def fn(a, b):
        return a + b

    measure_performance(1, 2)
    results_tp = measure_performance(model_tp, xx)
    results_mngs = measure_performance(model_mngs, xx)

    print("Tensorpac Results:", results_tp)
    print("MNGS Results:", results_mngs)

    # Close
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF

"""
/ssh:ywatanabe@444:/home/ywatanabe/proj/entrance/torchPAC/scripts/profilers.py
"""
