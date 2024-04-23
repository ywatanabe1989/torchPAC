#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-23 18:30:14"
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

    # Convert columns that are categorical to the 'category' data type
    categorical_columns = [
        "fp16",  # categorical
        "no_grad",  # categorical
        "in_place",  # categorical
        "trainable",  # categorical
        "device",  # categorical
        "use_threads",  # categorical
        "package",  # categorical
    ]
    for col in categorical_columns:
        df[col] = df[col].astype("category")

    # All columns to plot
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
        "fp16",  # categorical
        "no_grad",  # categorical
        "in_place",  # categorical
        "trainable",  # categorical
        "device",  # categorical
        "use_threads",  # categorical
        "package",  # categorical
    ]

    # Plots
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


def drop_cols(df):
    cols_del = [
        "time",
        "init_time_mean_sec",
        "init_time_std_sec",
        "init_time_nn",
        "calc_time_std_sec",
        "calc_time_nn",
        0,
    ]

    for k in cols_del:
        del df[k]

    df = df[~df.n_calc.isna()]

    return df


def get_indi(df, tgt_col=None):
    cols = [
        "batch_size",
        "n_chs",
        "n_segments",
        "t_sec",
        "fs",
        "pha_n_bands",
        "amp_n_bands",
        "pha_n_bands",
        "amp_n_bands",
        "chunk_size",
        "seq_len",
        "n_calc",
    ]

    if tgt_col in cols:
        cols.remove(tgt_col)

    base_conditions = df[cols].min()
    indi = []
    for k, v in base_conditions.items():
        indi.append(df[k] == v)
    indi = np.array(indi).all(axis=0)
    return indi


def print_uniques(df):
    print()
    for k in df.columns:
        print(k, df[k].unique())


def plot(df, tgt_col):
    # Slices the df
    df = df[get_indi(df, tgt_col)]

    # Plots
    fig, ax = mngs.plt.subplots()
    yy = df["calc_time_mean_sec"]
    xx = df[tgt_col]
    pp = df["package"]

    if tgt_col == "seq_len":
        import ipdb

        ipdb.set_trace()

    for _pp in pp.unique():
        # Color
        cc = {
            "mngs": CC["blue"],
            "tensorpac": CC["red"],
        }[_pp]
        # Slices for packages
        _ii = np.array(pp == _pp)
        _xx = np.array(xx[_ii])
        _yy = np.array(yy[_ii])

        # Sort according to the x axis
        _ii = np.argsort(_xx)
        _xx = _xx[_ii]
        _yy = _yy[_ii]

        ax.plot(_xx, _yy, label=_pp, color=cc)
        ax.set_xyt(tgt_col, "Calculation time [s]", "PAC Calculation")
        ax.legend()
        # ax.set_yscale("log")
    return fig


def main():
    df = link_results()
    df = drop_cols(df)
    # print_uniques(df)

    for tgt_col in ["batch_size", "chunk_size", "seq_len", "n_calc"]:
        fig = plot(df, tgt_col)
        mngs.io.save(fig, tgt_col + ".png")


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        fig_scale=5,
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)
