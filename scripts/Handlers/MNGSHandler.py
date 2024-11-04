#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-18 15:19:15 (ywatanabe)"
# Author: Yusuke Watanabe (ywata1989@gmail.com)

"""
This script defines MNGSHandler for phase-amplitude coupling calculations.
"""

# Imports
import math

import mngs
import torch
# from mngs.decorators import timeout
from scripts.Handlers import BaseHandler

# TIMEOUT_SEC = int(10 * 60)

# Functions
class MNGSHandler(BaseHandler):
    # def __init__(
    #     self,
    #     seq_len,
    #     fs,
    #     pha_n_bands,
    #     pha_min_hz,
    #     pha_max_hz,
    #     amp_n_bands,
    #     amp_min_hz,
    #     amp_max_hz,
    #     n_perm,
    #     chunk_size,
    #     fp16,
    #     in_place,
    #     trainable,
    #     device,
    #     use_threads,
    #     ts,
    # ):
    def __init__(
        self,
        seq_len: int,
        fs: float,
        pha_n_bands: int,
        pha_min_hz: float,
        pha_max_hz: float,
        amp_n_bands: int,
        amp_min_hz: float,
        amp_max_hz: float,
        n_perm: int,
        chunk_size: int,
        fp16: bool,
        in_place: bool,
        trainable: bool,
        device: str,
        use_threads: bool,
        ts: mngs.gen.TimeStamper,
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
        self.model = self.init_model()
        self.ts(self.init_end_str)

    # def init_model(
    #     self,
    # ):
    #     # model = mngs.nn.PAC(
    #     #     self.seq_len,
    #     #     self.fs,
    #     #     pha_n_bands=self.pha_n_bands,
    #     #     amp_n_bands=self.amp_n_bands,
    #     #     # n_perm=self.n_perm,
    #     #     fp16=self.fp16,
    #     #     # in_place=self.in_place,
    #     #     # trainable=self.trainable,
    #     # )

    #     model = mngs.nn.PAC(
    #         self.seq_len,
    #         self.fs,
    #         # Phase
    #         pha_start_hz=self.pha_min_hz,
    #         pha_end_hz=self.pha_max_hz,
    #         pha_n_bands=self.pha_n_bands,
    #         # Amplitude
    #         amp_n_bands=self.amp_n_bands,
    #         amp_start_hz=self.amp_min_hz,
    #         amp_end_hz=self.amp_max_hz,
    #         # Surrogation
    #         n_perm=self.n_perm,
    #         # Calculation
    #         fp16=self.fp16,
    #         in_place=self.in_place,
    #         trainable=self.trainable,
    #     )
    #     return model
    def init_model(self) -> mngs.nn.PAC:
        return mngs.nn.PAC(
            self.seq_len,
            self.fs,
            pha_start_hz=self.pha_min_hz,
            pha_end_hz=self.pha_max_hz,
            pha_n_bands=self.pha_n_bands,
            amp_n_bands=self.amp_n_bands,
            amp_start_hz=self.amp_min_hz,
            amp_end_hz=self.amp_max_hz,
            n_perm=self.n_perm,
            fp16=self.fp16,
            in_place=self.in_place,
            trainable=self.trainable,
        )

    # @timeout(
    #     seconds=TIMEOUT_SEC,
    #     error_message=f"\nFunction call timed out after {TIMEOUT_SEC} seconds",
    # )
    # def calc_pac(self, xx: torch.Tensor) -> torch.Tensor:
    #     """xx.shape: (batch_size, n_chs, n_segments, seq_len)"""

    #     # Ensures the input properties
    #     assert xx.ndim == 4
    #     assert xx.dtype == torch.float16 if self.fp16 else torch.float32

    #     xx = self.dim_handler.fit(xx, keepdims=[-2, -1])  # fixme

    #     # Batch processing to limit RAM/VRAM usage
    #     n_chunks = math.ceil(len(xx) / self.chunk_size)
    #     if self.chunk_size == 1:
    #         # add the first dimension to be accepted
    #         xpac = torch.vstack(
    #             [
    #                 self.model(
    #                     xx[
    #                         i_batch
    #                         * self.chunk_size : (i_batch + 1)
    #                         * self.chunk_size
    #                     ].unsqueeze(0)
    #                 ).unsqueeze(0)
    #                 for i_batch in range(n_chunks)
    #             ]
    #         )
    #     else:
    #         xpac = torch.vstack(
    #             [
    #                 self.model(
    #                     xx[
    #                         i_batch
    #                         * self.chunk_size : (i_batch + 1)
    #                         * self.chunk_size
    #                     ]
    #                 )
    #                 for i_batch in range(n_chunks)
    #             ]
    #         )

    #     xpac = self.dim_handler.unfit(xpac)  # fixme

    #     return xpac


    def calc_pac(self, xx: torch.Tensor) -> torch.Tensor:
        """
        Calculate phase-amplitude coupling.

        Parameters
        ----------
        xx : torch.Tensor
            Input tensor with shape (batch_size, n_chs, n_segments, seq_len).

        Returns
        -------
        torch.Tensor
            Calculated PAC values.
        """
        assert xx.ndim == 4
        assert xx.dtype == torch.float16 if self.fp16 else torch.float32

        xx = self.dim_handler.fit(xx, keepdims=[-2, -1])

        n_chunks = math.ceil(len(xx) / self.chunk_size)
        if self.chunk_size == 1:
            xpac = torch.vstack([
                self.model(xx[i_batch * self.chunk_size : (i_batch + 1) * self.chunk_size].unsqueeze(0)).unsqueeze(0)
                for i_batch in range(n_chunks)
            ])
        else:
            xpac = torch.vstack([
                self.model(xx[i_batch * self.chunk_size : (i_batch + 1) * self.chunk_size])
                for i_batch in range(n_chunks)
            ])

        return self.dim_handler.unfit(xpac)


    def __str__(self) -> str:
        return "mngs.dsp"

    @property
    def freqs_amp(self) -> torch.Tensor:
        return self.model.AMP_MIDS_HZ

    @property
    def freqs_pha(self) -> torch.Tensor:
        return self.model.PHA_MIDS_HZ

    # def __str__(
    #     self,
    # ):
    #     return "mngs.dsp"

    # @property
    # def freqs_amp(self):
    #     return self.model.AMP_MIDS_HZ

    # @property
    # def freqs_pha(self):
    #     return self.model.PHA_MIDS_HZ
