#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-23 14:56:06"
# Author: Yusuke Watanabe (ywata1989@gmail.com)


"""
This script does XYZ.
"""


"""
Imports
"""
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Config
"""
# CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""
import plotly.express as px


def link_results():
    def find_closest_time(row):
        # Calculate the absolute time difference
        time_diff = df["time"].sub(row["time"]).abs()
        # Find the index of the minimum time difference
        min_index = time_diff.idxmin()
        # Return the row from df that corresponds to the minimum time difference
        return df.loc[min_index]

    # CPU/GPU usages
    proc_usages = mngs.io.load("/tmp/mngs/processer_usages.csv")

    # Calculation results
    LDIRS = glob("./scripts/main/2024*/")
    # Parameters
    params = pd.concat([mngs.io.load(ldir + "params.csv") for ldir in LDIRS])
    params["seq_len"] = params["t_sec"] * params["fs"]
    # Stats
    stats = pd.concat([mngs.io.load(ldir + "stats.csv") for ldir in LDIRS])
    assert len(params) == len(stats)
    df = pd.concat([stats, params], axis=1)

    # Convert the 'Time' and 'time' columns from string to datetime
    df["time"] = pd.to_datetime(df["time"])
    stats["time"] = pd.to_datetime(stats["time"])

    # Link the stats and proc_usages
    closest_proc_usages = stats.apply(find_closest_time, axis=1)

    df = pd.concat([df, closest_proc_usages], axis=1)
    return df


def main():
    df = link_results()

    columns = [
        "time",
        # "init_time_mean_sec",
        # "calc_time_mean_sec",
        "batch_size",
        "n_chs",
        "n_segments",
        "seq_len",
        "pha_n_bands",
        "amp_n_bands",
        "n_perm",
        "chunk_size",
        "fp16",
        "no_grad",
        "in_place",
        "trainable",
        "device",
        "use_threads",
        "package",
        "seq_len",
    ]

    fig = px.parallel_coordinates(
        df.reset_index(),
        dimensions=columns,  # Specify the order of axes if needed
        color="calc_time_mean_sec",  # Variable for coloring
        labels={
            col: col.replace("_", " ").title() for col in columns
        },  # Optional: Enhance readability of labels
        color_continuous_scale=px.colors.diverging.Tealrose,  # Color scheme
        color_continuous_midpoint=np.median(
            df["calc_time_mean_sec"]
        ),  # Midpoint for diverging color scale
    )
    fig.update_layout(
        # Increase plot size
        # autosize=False,
        width=1200,
        height=800,
        # # Rotate labels
        # margin=dict(l=20, r=20, t=20, b=20),
    )

    fig.update_xaxes(
        tickangle=45,  # Rotate axis labels
    )
    # fig = px.parallel_coordinates(
    #     df.reset_index(),
    #     color="calc_time_mean_sec",
    #     labels={col: col for col in columns},
    #     color_continuous_scale=px.colors.diverging.Tealrose,
    #     color_continuous_midpoint=2,
    # )

    mngs.io.save(fig, "parallel_plot.png")


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        fig_scale=5,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)
