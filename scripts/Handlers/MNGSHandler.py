#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-10 18:42:09 (ywatanabe)"
# Author: Yusuke Watanabe (ywata1989@gmail.com)

"""This script defines MNGSHandler."""

# Imports
import math

import mngs
import torch
from mngs.general import timeout
from scripts.Handlers import BaseHandler

TIMEOUT_SEC = int(10 * 60)

# Functions
class MNGSHandler(BaseHandler):
    def __init__(
        self,
        seq_len,
        fs,
        pha_n_bands,
        pha_min_hz,
        pha_max_hz,
        amp_n_bands,
        amp_min_hz,
        amp_max_hz,
        n_perm,
        chunk_size,
        fp16,
        in_place,
        trainable,
        device,
        use_threads,
        ts,
    ):
        # Maintains parameters as attributes by following the BaseHandler's requirements
        super().__init__(
            seq_len,
            fs,
            pha_n_bands,
            pha_min_hz,
            pha_max_hz,
            amp_n_bands,
            amp_min_hz,
            amp_max_hz,
            n_perm,
            chunk_size,
            fp16,
            in_place,
            trainable,
            device,
            use_threads,
            ts,
        )

        # Explicitly disables unneccessary variables for this class.
        # Since parameters are passed using the grid search method, the above parameters should be accepted.
        del self.use_threads

        self.ts(self.init_start_str)

        # Model Initialization
        self.model = self.init_model()

        self.ts(self.init_end_str)

    def init_model(
        self,
    ):
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

        model = mngs.nn.PAC(
            self.seq_len,
            self.fs,
            # Phase
            pha_start_hz=self.pha_min_hz,
            pha_end_hz=self.pha_max_hz,
            pha_n_bands=self.pha_n_bands,
            # Amplitude
            amp_n_bands=self.amp_n_bands,
            amp_start_hz=self.amp_min_hz,
            amp_end_hz=self.amp_max_hz,
            # Surrogation
            n_perm=self.n_perm,
            # Calculation
            fp16=self.fp16,
            in_place=self.in_place,
            trainable=self.trainable,
        )
        return model

    # @timeout(
    #     seconds=TIMEOUT_SEC,
    #     error_message=f"\nFunction call timed out after {TIMEOUT_SEC} seconds",
    # )
    def calc_pac(self, xx: torch.Tensor) -> torch.Tensor:
        """xx.shape: (batch_size, n_chs, n_segments, seq_len)"""

        # Ensures the input properties
        assert xx.ndim == 4
        assert xx.dtype == torch.float16 if self.fp16 else torch.float32

        xx = self.dim_handler.fit(xx, keepdims=[-2, -1])  # fixme

        # Batch processing to limit RAM/VRAM usage
        n_chunks = math.ceil(len(xx) / self.chunk_size)
        if self.chunk_size == 1:
            # add the first dimension to be accepted
            xpac = torch.vstack(
                [
                    self.model(
                        xx[
                            i_batch
                            * self.chunk_size : (i_batch + 1)
                            * self.chunk_size
                        ].unsqueeze(0)
                    ).unsqueeze(0)
                    for i_batch in range(n_chunks)
                ]
            )
        else:
            xpac = torch.vstack(
                [
                    self.model(
                        xx[
                            i_batch
                            * self.chunk_size : (i_batch + 1)
                            * self.chunk_size
                        ]
                    )
                    for i_batch in range(n_chunks)
                ]
            )

        xpac = self.dim_handler.unfit(xpac)  # fixme

        return xpac

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
