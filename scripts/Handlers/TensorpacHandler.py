#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-23 12:33:16"
# Author: Yusuke Watanabe (ywata1989@gmail.com)


"""
This script defines TensorpacHandler.
"""

# Imports
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorpac
from scripts.Handlers import BaseHandler


# Functions
class TensorpacHandler(BaseHandler):
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
        del self.in_place, self.trainable, self.device

        self.ts(self.init_start_str)
        # Model Initialization
        self.model = self.init_model()
        self.ts(self.init_end_str)

    def init_model(
        self,
    ):
        resolution_dict = {
            10: "lres",
            30: "mres",
            50: "hres",
            70: "demon",
            100: "hulk",
        }
        model = tensorpac.Pac(
            f_pha=resolution_dict[self.pha_n_bands],
            f_amp=resolution_dict[self.pha_n_bands],
            dcomplex="wavelet",
        )
        model.idpac = (2, 0, 0)  # PAC using Modulation Index
        return model

    def calc_pac(self, xx):
        if self.use_threads:
            return self._calc_pac_unfair_using_threads(xx)
        else:
            return self._calc_pac_fair_chunk(xx)

    def _calc_pac(self, xs):
        """xs.shape: (-1, seq_len)"""

        # Bandpass filtering and Hilbert Transformation
        pha = self.model.filter(self.fs, xs, ftype="phase", n_jobs=-1)
        amp = self.model.filter(self.fs, xs, ftype="amplitude", n_jobs=-1)

        # Modulation Index
        xpac = self.model.fit(pha, amp, n_jobs=-1, verbose=False)
        xpac = xpac.transpose(2, 1, 0)

        return xpac

    def _calc_pac_fair_chunk(self, xx: np.ndarray) -> np.ndarray:
        """
        xx.shape: (batch_size, n_chs, n_segments, seq_len).
        This is suitable for fair comparisons between the MNGS's PAC calculation.
        """

        # Ensures the input properties
        assert xx.ndim == 4
        assert xx.dtype == np.float16 if self.fp16 else np.float32

        xx = self.dim_handler.fit(xx, keepdims=[-1])  # fixme

        # Batch processing to limit RAM/VRAM usage
        n_chunks = math.ceil(len(xx) / self.chunk_size)
        if self.chunk_size == 1:
            # (Do not need to add the first dimension for consistency in numpy)
            xpac = np.vstack(
                [
                    self._calc_pac(
                        xx[
                            i_batch
                            * self.chunk_size : (i_batch + 1)
                            * self.chunk_size
                        ]
                    )
                    for i_batch in range(n_chunks)
                ]
            )
        else:
            xpac = np.vstack(
                [
                    self._calc_pac(
                        xx[
                            i_batch
                            * self.chunk_size : (i_batch + 1)
                            * self.chunk_size
                        ]
                    )
                    for i_batch in range(n_chunks)
                ]
            )

        xpac = self.dim_handler.unfit(xpac)

        return xpac

    def _calc_pac_unfair_using_threads(self, xx, n_jobs=-1):
        """
        xx.shape: (batch_size, n_chs, n_segments, seq_len)
        This is not suitable for fair comparisons between the MNGS's PAC calculation.
        """

        # Ensures the input properties
        assert xx.ndim == 4
        assert xx.dtype == np.float16 if self.fp16 else np.float32

        xx = self.dim_handler.fit(xx, keepdims=np.array([-2, -1]))

        # Processing using threads
        batch_size_by_n_chs, n_segments, _ = xx.shape
        xpac = []
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
                xpac.append(future.result())

        # Collect calculated xpacs
        xpac = np.array(xpac)
        xpac = self.dim_handler.unfit(xpac)

        return xpac

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
