#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-21 12:19:00"
# Author: Yusuke Watanabe (ywata1989@gmail.com)

"""
This script defines a handler for mngs package's PAC calculation, which is designed for fair comparison with that of the tensorpac package.
"""

import math

# Imports
import random
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import torch
import torch.nn as nn


# Functions
# batch version
class MNGSHandler(nn.Module):
    def __init__(
        self,
        seq_len,
        fs,
        pha_n_bands=50,
        amp_n_bands=50,
        n_perm=None,
        fp16=True,
        in_place=False,
        trainable=False,
    ):
        super().__init__()
        self.model = mngs.nn.PAC_dev(
            seq_len,
            fs,
            pha_n_bands=pha_n_bands,
            amp_n_bands=amp_n_bands,
            n_perm=n_perm,
            fp16=fp16,
            in_place=in_place,
            trainable=trainable,
        )
        self.fp16 = fp16
        self.dh = mngs.gen.DimHandler()

    def __call__(self, xx, batch_size=1):
        if xx.ndim == 3:
            n_batched, n_chs, seq_len = xx.shape
        elif xx.ndim == 4:
            n_batched, n_chs, n_segments, seq_len = xx.shape

        # Converts xx to the shape of (batch_size, n_chs=1, seq_len).
        xx = self.dh.fit(xx, keepdims=[-1]).unsqueeze(1)
        n_batches = math.ceil(len(xx) / batch_size)
        pac = torch.vstack(
            [
                self.model(
                    xx[i_batch * batch_size : (i_batch + 1) * batch_size]
                )
                for i_batch in range(n_batches)
            ]
        )
        pac = self.dh.unfit(pac)

        # Take means along the n_segments dimension
        if pac.ndim == 4:
            pac = pac.mean(dim=-3)

        return pac


if __name__ == "__main__":
    # Start
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    ts = mngs.gen.TimeStamper()

    # Parameters
    BATCH_SIZE = 32
    N_CHS = 19
    T_SEC = 4
    N_SEGMENTS = 2
    device = "cuda"
    fp16 = True
    in_place = True

    # Demo Signal
    xx, tt, fs = mngs.dsp.demo_sig(
        batch_size=BATCH_SIZE,
        n_chs=N_CHS,
        t_sec=T_SEC,
        n_segments=N_SEGMENTS,
        sig_type="tensorpac",
    )
    # (8, 19, 20, 2048)

    # Model Initialization
    ts("Model Initialization Starts")
    model = MNGSHandler(xx.shape[-1], fs, fp16=fp16, in_place=in_place)
    ts("Model Initialization Ends")

    # From the second time, cache boosts the calculation speed.
    calc_times = []
    for ii in range(10):
        # Signal
        xx, tt, fs = mngs.dsp.demo_sig(
            batch_size=BATCH_SIZE,
            n_chs=N_CHS,
            t_sec=T_SEC,
            n_segments=N_SEGMENTS,
            sig_type=random.choice(["tensorpac", "pac", "chirp"]),
        )

        # Noise
        xx = mngs.dsp.add_noise.gauss(xx)  # Ensures inputs are different.

        # device and dtype
        xx = torch.tensor(xx).to(device)
        if fp16:
            xx = xx.half()

        # Disables building calculation graphs for backpropagation
        xx.requires_grad = False

        # PAC calculation
        ts("PAC Calculation Starts")

        # Disables building calculation graphs for backpropagation
        with torch.no_grad():
            pac = model(xx, batch_size=16)
        ts("PAC Calculation Ends")

        # Print the results
        calc_time = ts.delta(-1, -2)
        calc_times.append(calc_time)
        mngs.gen.print_block(
            f"Trial {ii} | Calculation Time: {calc_time:.3f} sec from x ({xx.shape}) to pac {pac.shape} ",
            c="green",
        )

    mngs.gen.print_block(
        f"Summary | Calculation Time: {np.mean(calc_times):.3f} +/- {np.std(calc_times):.3f} sec (n = {len(calc_times)}) from x ({xx.shape}) to pac {pac.shape} ",
        c="green",
    )

    # Close
    mngs.gen.close(CONFIG, verbose=False, notify=False)


# EOF

"""
./scripts/MNGSHandler.py
"""
# ----------------------------------------
# Calculation Time: 0.061 sec from x (torch.Size([8, 19, 1, 2048])) to pac torch.Size([8, 19, 50, 30])
# ----------------------------------------

# ----------------------------------------
# Calculation Time: 0.143 sec from x (torch.Size([8, 19, 1, 2048])) to pac torch.Size([8, 19, 50, 30, 1])
# ----------------------------------------