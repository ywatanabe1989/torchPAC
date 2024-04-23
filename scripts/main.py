#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-23 12:46:03"
# Author: Yusuke Watanabe (ywata1989@gmail.com)

"""
This script does XYZ.
"""

import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import mngs
import numpy as np
import torch

# Imports
from scripts.Handlers import MNGSHandler, TensorpacHandler


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
            batch_size=params["batch_size"],
            n_chs=params["n_chs"],
            t_sec=params["t_sec"],
            n_segments=params["n_segments"],
            sig_type="pac",
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


def main(params, N_CALC=3):
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
    if params["no_grad"]:
        with torch.no_grad():
            for _ in range(N_CALC):
                model.ts(model.calc_start_str)
                model.calc_pac(xx)
                model.ts(model.calc_end_str)
    else:
        for _ in range(N_CALC):
            model.ts(model.calc_start_str)
            model.calc_pac(xx)
            model.ts(model.calc_end_str)

    # Stats
    mngs.gen.print_block(
        f"{params['package']}\n"
        f"{model.stats['calc_time_mean_sec'].iloc[0]} +/- {model.stats['calc_time_std_sec'].iloc[0]} sec",
        c="green" if params["package"] == "mngs" else "magenta",
    )
    mngs.io.save(model.stats, CONFIG["SDIR"] + "stats.csv")

    # Close
    mngs.gen.close(CONFIG, verbose=False, notify=False)

    del model


if __name__ == "__main__":
    PARAM_SPACES = mngs.io.load("./config/PARAM_SPACES.yaml")
    for i_params, params in enumerate(
        mngs.ml.utils.grid_search.yield_grids(PARAM_SPACES, random=True)
    ):
        main(params)


# koko

# def main():
#     # Parameters
#     BATCH_SIZE = 16
#     N_CHS = 16
#     N_SEGMENTS = 8
#     T_SEC = 4
#     device = "cuda"
#     fp16 = True
#     in_place = True
#     N_calc = 5
#     CHUNK_SIZE = 16
#     PHA_N_BANDS = 50
#     AMP_N_BANDS = 30

#     # Demo Signal
#     xx, tt, fs = mngs.dsp.demo_sig(
#         batch_size=BATCH_SIZE,
#         n_chs=N_CHS,
#         t_sec=T_SEC,
#         n_segments=N_SEGMENTS,
#         sig_type="pac",
#     )
#     # (8, 19, 20, 2048)
#     seq_len = xx.shape[-1]

#     # Time stampers
#     ts_tp = mngs.gen.TimeStamper()
#     ts_mngs = mngs.gen.TimeStamper()

#     ################################################################################
#     # PAC Calculation
#     ################################################################################
#     # mngs
#     model_mngs = MNGSHandler(
#         seq_len=seq_len,
#         fs=float(fs),
#         ts=ts_mngs,
#         device=device,
#         fp16=fp16,
#         pha_n_bands=PHA_N_BANDS,
#         amp_n_bands=AMP_N_BANDS,
#     )
#     _xx = torch.tensor(xx).to(device)

#     self = model_mngs
#     self.ts(self.calc_start_str)
#     with torch.no_grad():
#         for _ in range(N_calc):
#             model_mngs.calc_pac(_xx, chunk_size=CHUNK_SIZE)
#     self.ts(self.calc_end_str)

#     # tensorpac
#     model_tp = TensorpacHandler(
#         seq_len=seq_len,
#         fs=float(fs),
#         ts=ts_tp,
#         device=device,
#         fp16=fp16,
#         pha_n_bands=PHA_N_BANDS,
#         amp_n_bands=AMP_N_BANDS,
#     )
#     _xx = np.array(xx)
#     self = model_tp
#     self.ts(self.calc_start_str)
#     for _ in range(N_calc):
#         model_tp.calc_pac(_xx, chunk_size=CHUNK_SIZE)
#     self.ts(self.calc_end_str)
#     # koko

#     stats = pd.concat([model_tp.stats, model_mngs.stats])

#     assert np.allclose(model_tp.freqs_pha, model_mngs.freqs_pha)
#     mngs.gen.print_block(
#         "Error, fixme, np.allclose(model_tp.freqs_amp, model_mngs.freqs_amp)",
#         c="red",
#     )
#     # assert np.allclose(model_tp.freqs_amp, model_mngs.freqs_amp)  # fixme

#     ################################################################################
#     # Compare mngs with tensorpac
#     ################################################################################
#     tt = "Tensorpac"
#     mm = "mngs.dsp"

#     # ################################################################################
#     # # Input shape
#     # ################################################################################
#     # input_shapes = stats.input_shape
#     # assert input_shapes[mm] == input_shapes[tt]
#     # mngs.gen.print_block(f"Input shape: {input_shapes[mm]}")

#     # ################################################################################
#     # # Output shape
#     # ################################################################################
#     # output_shapes = stats.output_shape
#     # assert output_shapes[mm] == output_shapes[tt]
#     # mngs.gen.print_block(f"Output shape: {output_shapes[mm]}")

#     # ################################################################################
#     # # Precision of PAC values
#     # ################################################################################
#     # pac_values = stats.calculated_pac
#     # i_batch, i_ch = 0, 0
#     # indi = (i_batch, i_ch)

#     # fig, axes = mngs.plt.subplots(ncols=3)

#     # vmin = min(
#     #     pac_values[mm][indi].min(),
#     #     pac_values[tt][indi].min(),
#     #     (pac_values[mm][indi] - pac_values[tt][indi]).min(),
#     # )
#     # vmax = max(
#     #     pac_values[mm][indi].max(),
#     #     pac_values[tt][indi].max(),
#     #     (pac_values[mm][indi] - pac_values[tt][indi]).max(),
#     # )

#     # axes[0].imshow2d(pac_values[mm][indi], vmin=vmin, vmax=vmax, cbar=False)
#     # axes[1].imshow2d(pac_values[tt][indi], vmin=vmin, vmax=vmax, cbar=False)
#     # axes[2].imshow2d(
#     #     pac_values[mm][indi] - pac_values[tt][indi], vmin=vmin, vmax=vmax
#     # )

#     # plt.show()

#     ################################################################################
#     # Calculation Speed
#     ################################################################################
#     calc_times_mean_sec = stats.calc_time_mean_sec
#     time_ratio = (
#         calc_times_mean_sec["mngs.dsp"] / calc_times_mean_sec["Tensorpac"]
#     )
#     speed_ratio = 1 / time_ratio
#     mngs.gen.print_block(
#         f"Speed ratio (mngs.dsp / tensorpac): {speed_ratio:.3f}"
#     )


# # EOF
