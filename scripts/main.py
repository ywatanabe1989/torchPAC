#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-10 21:28:56 (ywatanabe)"
# Author: Yusuke Watanabe (ywata1989@gmail.com)
# ./scripts/main.py

"""
Functionality:
    * Performs PAC (Phase-Amplitude Coupling) calculation using different packages.
Input:
    * Parameters for signal generation and PAC calculation.
Output:
    * Calculation statistics and parameters saved as CSV files.
Prerequisites:
    * mngs, numpy, pandas, torch, matplotlib packages
    * Custom Handlers (MNGSHandler, TensorpacHandler)
"""

"""Imports"""
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import torch
from scripts.Handlers import MNGSHandler, TensorpacHandler

"""Functions & Classes"""
def init_model(params):
    try:
        params_h = params.copy()
        params_h["pha_min_hz"] = 2
        params_h["pha_max_hz"] = 20
        params_h["amp_min_hz"] = 80
        params_h["amp_max_hz"] = 160

        for k in [
            "batch_size",
            "n_chs",
            "n_segments",
            "t_sec",
            "package",
            "no_grad",
            "n_calc",
        ]:
            del params_h[k]

        if params["package"] == "mngs":
            model = MNGSHandler(**params_h)
        elif params["package"] == "tensorpac":
            model = TensorpacHandler(**params_h)

        return model

    except Exception as e:
        print(e)


def prepare_signal(params):
    try:
        # Demo Signal
        xx, tt, fs = mngs.dsp.demo_sig(
            sig_type="pac",
            batch_size=params["batch_size"],
            n_chs=params["n_chs"],
            n_segments=params["n_segments"],
            t_sec=params["t_sec"],
            fs=params["fs"],
        )

        if params["package"] == "tensorpac":
            xx = np.array(xx)
            if params["fp16"]:
                xx = np.array(xx).astype(np.float16)
            else:
                xx = np.array(xx).astype(np.float32)

        elif params["package"] == "mngs":
            if params["fp16"]:
                xx = torch.tensor(xx).half().to(params["device"])
            else:
                xx = torch.tensor(xx).float().to(params["device"])
        return xx
    except Exception as e:
        print(e)


def main(params):
    import matplotlib.pyplot as plt

    ts = mngs.gen.TimeStamper()

    params["ts"] = ts

    # Start
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    mngs.gen.print_block(params, c="yellow")

    # Model
    params["seq_len"] = int(params["t_sec"] * params["fs"])

    # Model Initialization
    model = init_model(params)

    # Input signal
    xx = prepare_signal(params)

    if (model is None) or (xx is None):
        return None

    # Calculation
    try:
        if params["no_grad"]:
            with torch.no_grad():
                for _ in range(params["n_calc"]):
                    model.ts(model.calc_start_str)
                    model.calc_pac(xx)
                    model.ts(model.calc_end_str)
        else:
            for _ in range(params["n_calc"]):
                model.ts(model.calc_start_str)
                model.calc_pac(xx)
                model.ts(model.calc_end_str)
    except Exception as e:
        print(e)
        return None

    # Stats
    mngs.gen.print_block(
        f"{params['package']}\n"
        f"{model.stats['calc_time_mean_sec'].iloc[0]} +/- {model.stats['calc_time_std_sec'].iloc[0]} sec",
        c="green" if params["package"] == "mngs" else "magenta",
    )
    mngs.io.save(model.stats, CONFIG["SDIR"] + "stats.csv")

    del params["ts"]
    mngs.io.save(
        pd.DataFrame(pd.Series(params)).T, CONFIG["SDIR"] + "params.csv"
    )

    # Close
    mngs.gen.close(CONFIG, verbose=False, notify=False)

    del model


if __name__ == "__main__":
    PARAM_SPACES = mngs.io.load("./config/PARAM_SPACES.yaml")
    for i_params, params in enumerate(
        mngs.ml.utils.grid_search.yield_grids(PARAM_SPACES, random=True)
    ):
        main(params)
        torch.cuda.empty_cache()

# EOF
