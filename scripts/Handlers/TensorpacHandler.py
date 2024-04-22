#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-22 22:11:18"
# Author: Yusuke Watanabe (ywata1989@gmail.com)


"""
This script defines MNGSHandler and TensorpacHandler, bsaed on BaseHandler.
"""

# Imports
import math
import sys
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional, Type, Union

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import tensorpac
import torch
from scripts.Handlers import BaseHandler

warnings.simplefilter("ignore", UserWarning)

# Functions
class TensorpacHandler(BaseHandler):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.model = self.init_model()

    def init_model(
        self,
    ):
        self.ts(self.init_start_str)
        res_dict = {
            10: "lres",
            30: "mres",
            50: "hres",
            70: "demon",
            100: "hulk",
        }
        model = tensorpac.Pac(
            f_pha=res_dict[self.pha_n_bands],
            f_amp=res_dict[self.pha_n_bands],
            dcomplex="wavelet",
        )
        model.idpac = (2, 0, 0)
        self.ts(self.init_end_str)
        return model

    def _calc_pac(self, xs):
        # Bandpass filtering and Hilbert Transformation
        pha = self.model.filter(self.fs, xs, ftype="phase", n_jobs=-1)
        amp = self.model.filter(self.fs, xs, ftype="amplitude", n_jobs=-1)

        # Modulation Index
        xpac = self.model.fit(pha, amp, n_jobs=-1, verbose=False)
        xpac = xpac.transpose(2, 1, 0)

        return xpac

    def calc_pac(self, xx, chunk_size=3):
        assert xx.ndim == 4

        xx = self.xx_dim_handler.fit(xx, keepdims=[-1])

        # Batch processing to limit RAM/VRAM usage
        n_chunks = math.ceil(len(xx) / chunk_size)

        if chunk_size == 1:
            # add the first dimension to be accepted
            pac = np.vstack(
                [
                    self._calc_pac(
                        xx[i_batch * chunk_size : (i_batch + 1) * chunk_size][
                            np.newaxis
                        ]
                    )
                    for i_batch in range(n_chunks)
                ]
            )
        else:
            pac = np.vstack(
                [
                    self._calc_pac(
                        xx[i_batch * chunk_size : (i_batch + 1) * chunk_size]
                    )
                    for i_batch in range(n_chunks)
                ]
            )

        pac = self.xx_dim_handler.unfit(pac)

        # Takes mean across n_segments
        pac = pac.mean(axis=-3)

        return pac

    def calc_pac_threds(self, xx, chunk_size=None, n_jobs=-1):
        xx = self.xx_dim_handler.fit(xx, keepdims=np.array([-2, -1]))

        batch_size_by_n_chs, n_segments, _ = xx.shape

        pac = []
        with ThreadPoolExecutor(
            max_workers=None
        ) as executor:  # Using system's max processors
            futures = []
            for i_batch in range(batch_size_by_n_chs):
                futures.append(
                    executor.submit(
                        self._calc_pac,
                        xx[i_batch],
                    )
                )
            for future in futures:
                pac.append(future.result())

        pac = np.array(pac)
        pac = self.xx_dim_handler.unfit(pac)

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

    def __str__(
        self,
    ):
        return "Tensorpac"
