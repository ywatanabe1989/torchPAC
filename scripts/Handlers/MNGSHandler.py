#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-22 22:10:20"
# Author: Yusuke Watanabe (ywata1989@gmail.com)


"""
This script defines MNGSHandler and TensorpacHandler, bsaed on BaseHandler.
"""

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
from mngs.general import torch_fn

# Imports
from scripts.Handlers import BaseHandler

warnings.simplefilter("ignore", UserWarning)

# Functions
class MNGSHandler(BaseHandler):
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
        # model = mngs.nn.PAC(
        #     self.seq_len,
        #     self.fs,
        #     pha_n_bands=self.pha_n_bands,
        #     amp_n_bands=self.amp_n_bands,
        #     # n_perm=self.n_perm,
        #     fp16=self.fp16,
        #     # in_place=self.in_place,
        #     # trainable=self.trainable,
        # )

        model = mngs.nn.PAC_dev(
            self.seq_len,
            self.fs,
            pha_n_bands=self.pha_n_bands,
            amp_n_bands=self.amp_n_bands,
            n_perm=self.n_perm,
            fp16=self.fp16,
            in_place=self.in_place,
            trainable=self.trainable,
        )
        self.ts(self.init_end_str)
        return model

    def calc_pac(self, xx, chunk_size=3):  # fixme
        # self.ts(self.calc_start_str)

        # self.input_shape = xx.shape

        assert xx.ndim == 4

        xx = self.xx_dim_handler.fit(xx, keepdims=[-2, -1])

        # Batch processing to limit RAM/VRAM usage
        n_chunks = math.ceil(len(xx) / chunk_size)
        if chunk_size == 1:
            # add the first dimension to be accepted
            pac = torch.vstack(
                [
                    self.model(
                        xx[
                            i_batch * chunk_size : (i_batch + 1) * chunk_size
                        ].unsqueeze(0)
                    )
                    for i_batch in range(n_chunks)
                ]
            )
        else:
            pac = torch.vstack(
                [
                    self.model(
                        xx[i_batch * chunk_size : (i_batch + 1) * chunk_size]
                    )
                    for i_batch in range(n_chunks)
                ]
            )

        pac = self.xx_dim_handler.unfit(pac)

        # # Takes mean across n_segments
        # pac = pac.mean(dim=-3)

        # # self.ts(self.calc_end_str)

        # self.output_shape = pac.shape

        return pac

    def __str__(
        self,
    ):
        return "mngs.dsp"

    @property
    def freqs_amp(self):
        return self.model.AMP_MIDS_HZ

    @property
    def freqs_pha(self):
        return self.model.PHA_MIDS_HZ
