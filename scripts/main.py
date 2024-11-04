#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-04 17:03:27 (ywatanabe)"
# File: ./torchPAC/scripts/main.py

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

import multiprocessing
from typing import Any, Dict

import mngs

from scripts.utils.init_model import init_model
from scripts.utils.perform_pac_calculation import perform_pac_calculation
from scripts.utils.prepare_signal import prepare_signal
from scripts.utils.save_results import save_results

CONFIG = mngs.io.load_configs()

def run_condition(params: Dict[str, Any]) -> None:
    """Executes PAC calculation for given parameters.

    Parameters
    ----------
    params : Dict[str, Any]
        Dictionary containing calculation parameters

    Returns
    -------
    None
    """
    import sys

    import matplotlib.pyplot as plt

    ts = mngs.gen.TimeStamper()
    params["ts"] = ts

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )

    mngs.str.printc(params, c="yellow")

    params["seq_len"] = int(params["t_sec"] * params["fs"])
    model = init_model(params)
    signal = prepare_signal(params)

    if model is None or signal is None:
        return

    perform_pac_calculation(model, signal, params)
    save_results(model, params, CONFIG)

    mngs.gen.close(CONFIG, verbose=False, notify=False)

    del model

    try:
        sys.stdout.close()
        sys.stderr.close()
    except Exception as e:
        print(e)




def main() -> None:
    """Main function to iterate through parameter spaces and run PAC calculations."""

    # mngs.resource.log_processor_usages(CONFIG.PATH.PROCESSOR_USAGE, background=True)

    param_space = mngs.io.load("./config/PARAM_SPACES.yaml")
    if CONFIG.IS_DEBUG:
        param_space = {k.replace("debug_", ""):v for k,v in param_space.items() if "debug_" in k.lower()}
    else:
        param_space = {k:v for k,v in param_space.items() if "debug_" not in k.lower()}

    for params in mngs.utils.yield_grids(
        param_space, random=True
    ):
        mngs.str.printc(params)
        run_condition(params)
        # torch.cuda.empty_cache()

    return 0

if __name__ == '__main__':
    # -----------------------------------
    # Initiatialization of mngs format
    # -----------------------------------
    # import sys

    # import matplotlib.pyplot as plt

    # # Configurations
    # CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
    #     sys,
    #     plt,
    #     verbose=False,
    #     agg=True,
    #     # sdir_suffix="",
    # )
    # -----------------------------------
    # Main
    # -----------------------------------

    # multiprocessing.set_start_method("spawn")
    exit_status = main()
    # exit_status = main()

    # -----------------------------------
    # Cleanup mngs format
    # -----------------------------------
    # mngs.gen.close(
    #     CONFIG,
    #     verbose=False,
    #     notify=False,
    #     message="",
    #     exit_status=exit_status,
    # )

# EOF
