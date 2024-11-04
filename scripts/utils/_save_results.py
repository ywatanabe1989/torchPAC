#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-04 15:23:46 (ywatanabe)"
# File: ./torchPAC/scripts/utils/_save_results.py

import mngs
import pandas as pd


def save_results(model, params, CONFIG):
    mngs.str.printc(
        f"{params['package']}\n"
        f"{model.stats['calc_time_mean_sec'].iloc[0]} +/- {model.stats['calc_time_std_sec'].iloc[0]} sec "
        f"(n = {model.stats['calc_time_nn'].iloc[0]:,})",
        c="green" if params["package"] == "mngs" else "magenta",
    )
    mngs.io.save(model.stats, CONFIG["SDIR"] + "stats.csv")

    params_copy = params.copy()
    del params_copy["ts"]
    mngs.io.save(pd.DataFrame(pd.Series(params_copy)).T, CONFIG["SDIR"] + "params.csv")

# EOF
