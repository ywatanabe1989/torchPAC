#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-19 18:25:05"
# Author: Yusuke Watanabe (ywata1989@gmail.com)

"""
This script measures the PAC calculation time of the tensorpac package.
"""


import random

# Imports
import sys
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import mngs
import numpy as np
import tensorpac


# Functions
class TensorpacHandler:
    def __init__(self, n_bands_pha=50, n_bands_amp=50):
        res_dict = {
            10: "lres",
            30: "mres",
            50: "hres",
            70: "demon",
            100: "hulk",
        }

        self.model = tensorpac.Pac(
            f_pha=res_dict[n_bands_pha],
            f_amp=res_dict[n_bands_amp],
            dcomplex="wavelet",
        )
        self.model.idpac = (2, 0, 0)

    def __call__(self, xx, fs):
        batch_size, n_chs, _, _ = xx.shape
        pac_results = np.zeros(
            (batch_size, n_chs, len(self.freqs_pha), len(self.freqs_amp))
        )

        with ThreadPoolExecutor(
            max_workers=None
        ) as executor:  # Using system's max processors
            futures = []
            for i_batch in range(batch_size):
                for i_ch in range(n_chs):
                    futures.append(
                        executor.submit(self._calc_pac, xx, fs, i_batch, i_ch)
                    )

            for future, idx in zip(futures, np.ndindex(batch_size, n_chs)):
                pac_results[idx] = future.result()

        return pac_results

    def _calc_pac(self, xx, fs, i_batch=0, i_ch=0):
        """Calculate pac from two-dimensional signal (i_batch, i_ch)"""
        pha, amp = self._bandpass_and_hilbert(
            xx, fs, i_batch=i_batch, i_ch=i_ch
        )
        pac = self.modulation_index(pha, amp)
        return pac

    def _bandpass_and_hilbert(self, xx, fs, i_batch=0, i_ch=0):
        # 2d input is accepted.
        pha = self.model.filter(
            fs, xx[i_batch, i_ch], ftype="phase", n_jobs=-1
        )
        amp = self.model.filter(
            fs, xx[i_batch, i_ch], ftype="amplitude", n_jobs=-1
        )
        return pha, amp

    def modulation_index(self, pha, amp):
        xpac = self.model.fit(pha, amp, verbose=False)
        pac = xpac.mean(axis=-1)
        pac = pac.T  # (amp, pha) -> (pha, amp)
        return pac

    @property
    def freqs_amp(
        self,
    ):
        return self.model.f_amp.mean(axis=-1)

    @property
    def freqs_pha(
        self,
    ):
        return self.model.f_pha.mean(axis=-1)


if __name__ == "__main__":
    # Start
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    ts = mngs.gen.TimeStamper()

    # Parameters
    BATCH_SIZE = 4
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
    model = TensorpacHandler()
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
            sig_type=random.choice(["tensorpac", "pac"]),
        )

        # Noise
        xx = mngs.dsp.add_noise.gauss(xx)  # Ensures inputs are different.

        if fp16:
            xx = xx.astype(np.float16)

        # PAC calculation
        ts("PAC Calculation Starts")
        pac = model(xx, fs)
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
./scripts/TensorpacHandler.py
"""
