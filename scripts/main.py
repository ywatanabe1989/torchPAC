#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-18 15:32:36 (ywatanabe)"
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

import sys
import mngs
import torch
import pandas as pd
from utils.prepare_signal import prepare_signal
from utils.init_model import init_model
import multiprocessing

def main():
    PARAM_SPACES = mngs.io.load("./config/PARAM_SPACES.yaml")
    for params in mngs.gen.yield_grids(PARAM_SPACES, random=True):
        run_condition(params)
        torch.cuda.empty_cache()


def run_condition(params: dict) -> None:
    import matplotlib.pyplot as plt
    import sys
    import matplotlib.pyplot as plt

    ts = mngs.gen.TimeStamper()
    params["ts"] = ts

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt, verbose=False, agg=True)

    mngs.gen.print_block(params, c="yellow")

    params["seq_len"] = int(params["t_sec"] * params["fs"])
    model = init_model(params)
    signal = prepare_signal(params)

    if model is None or signal is None:
        return

    perform_pac_calculation(model, signal, params)
    save_results(model, params, CONFIG)

    mngs.gen.close(CONFIG, verbose=False, notify=False)
    del model

def perform_pac_calculation(model, signal, params):
    try:
        for _ in range(params["n_calc"]):
            model.ts(model.calc_start_str)
            if params["no_grad"]:
                with torch.no_grad():
                    model.calc_pac(signal)
            else:
                model.calc_pac(signal)
            model.ts(model.calc_end_str)
    except Exception as exception:
        print(f"Error in PAC calculation: {exception}")

def save_results(model, params, CONFIG):
    mngs.gen.print_block(
        f"{params['package']}\n"
        f"{model.stats['calc_time_mean_sec'].iloc[0]} +/- {model.stats['calc_time_std_sec'].iloc[0]} sec",
        c="green" if params["package"] == "mngs" else "magenta",
    )
    mngs.io.save(model.stats, CONFIG["SDIR"] + "stats.csv")

    params_copy = params.copy()
    del params_copy["ts"]
    mngs.io.save(pd.DataFrame(pd.Series(params_copy)).T, CONFIG["SDIR"] + "params.csv")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()

# EOF
