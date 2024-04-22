#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-22 21:13:54"
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
from mngs.general import torch_fn

warnings.simplefilter("ignore", UserWarning)

# Functions
@dataclass
class BaseHandler(ABC):
    seq_len: int = None
    fs: float = None

    pha_min_hz: Union[int, float] = 2
    pha_max_hz: Union[int, float] = 20
    pha_n_bands: Union[int, float] = 50
    amp_min_hz: Union[int, float] = 2
    amp_max_hz: Union[int, float] = 20
    amp_n_bands: Union[int, float] = 50

    n_perm: int = None
    fp16: Optional[bool] = True
    in_place: Optional[bool] = True
    trainable: Optional[bool] = True

    device: Optional[str] = None
    # input_shape: Optional[List[int]] = field(default_factory=list)
    # output_shape: Optional[List[int]] = field(default_factory=list)
    model: Optional[Type] = None
    xx_dim_handler = mngs.gen.DimHandler()
    pha_dim_handler = mngs.gen.DimHandler()
    amp_dim_handler = mngs.gen.DimHandler()
    ts = None
    init_start_str = "Model Initialization Starts"
    init_end_str = "Model Initialization Ends"
    calc_start_str = "PAC Calculation Starts"
    calc_end_str = "PAC Calculation Ends"

    @abstractmethod
    def __init__(self, **kwargs):
        self.seq_len = kwargs.get("seq_len", None)
        self.fs = kwargs.get("fs", None)
        self.pha_min_hz = kwargs.get("pha_min_hz", 2)
        self.pha_max_hz = kwargs.get("pha_max_hz", 20)
        self.pha_n_bands = kwargs.get("pha_n_bands", 50)
        self.amp_n_bands = kwargs.get("amp_n_bands", 50)
        self.amp_min_hz = kwargs.get("amp_min_hz", 50)
        self.amp_max_hz = kwargs.get("amp_max_hz", 160)
        self.n_perm = kwargs.get("n_perm", None)
        self.fp16 = kwargs.get("fp16", None)
        self.in_place = kwargs.get("in_place", None)
        self.trainable = kwargs.get("trainable", None)
        self.device = kwargs.get("device", None)
        self.ts = kwargs.get("ts", None)

    @abstractmethod
    def init_model(self, **kwargs):
        pass

    @abstractmethod
    def __str__(
        self,
    ):
        pass

    @abstractmethod
    def calc_pac(self, xx):
        pass

    @property
    @abstractmethod
    def freqs_amp(self):
        pass

    @property
    @abstractmethod
    def freqs_pha(self):
        pass

    @property
    def stats(
        self,
    ):
        # Initialization metrics
        # indices
        indi_init_start = np.where(
            self.ts.record["comment"] == self.init_start_str
        )[0]
        indi_init_end = np.where(
            self.ts.record["comment"] == self.init_end_str
        )[0]
        # time difference
        init_start_times = np.array(
            self.ts.record.loc[indi_init_start].timestamp
        )
        init_end_times = np.array(self.ts.record.loc[indi_init_end].timestamp)
        init_delta_times = init_end_times - init_start_times
        init_delta_times_mm = init_delta_times.mean()
        init_delta_times_ss = init_delta_times.std()
        init_delta_times_nn = len(init_delta_times)

        # Calculation metrics
        # indices
        indi_calc_start = np.where(
            self.ts.record["comment"] == self.calc_start_str
        )[0]
        indi_calc_end = np.where(
            self.ts.record["comment"] == self.calc_end_str
        )[0]
        # time difference
        calc_start_times = np.array(
            self.ts.record.loc[indi_calc_start].timestamp
        )
        calc_end_times = np.array(self.ts.record.loc[indi_calc_end].timestamp)
        calc_delta_times = calc_end_times - calc_start_times
        calc_delta_times_mm = calc_delta_times.mean()
        calc_delta_times_ss = calc_delta_times.std()
        calc_delta_times_nn = len(calc_delta_times)

        dic = {
            # Initialization metrics
            "init_time_mean_sec": init_delta_times_mm,
            "init_time_std_sec": init_delta_times_ss,
            "init_time_nn": init_delta_times_nn,
            # Calculation metrics
            "calc_time_mean_sec": calc_delta_times_mm,
            "calc_time_std_sec": calc_delta_times_ss,
            "calc_time_nn": calc_delta_times_nn,
            # Shapes
            # "input_shape": [self.input_shape],
            # "output_shape": [self.output_shape],
            # "batch_size": self.input_shape[0],
            # "n_chs": self.input_shape[1],
            # # The calculated PAC
            # "calculated_pac": [self.pac],
        }

        df = pd.DataFrame(
            data=dic,
            index=[str(self)],
        ).round(3)

        return df


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

    # def calc_pac(self, xx, chunk_size=3):
    #     # self.ts(self.calc_start_str)

    #     # self.input_shape = xx.shape

    #     assert xx.ndim == 4

    #     xx = self.xx_dim_handler.fit(xx, keepdims=[-1])

    #     # Batch processing to limit RAM/VRAM usage
    #     n_chunks = math.ceil(len(xx) / chunk_size)

    #     if chunk_size == 1:
    #         # add the first dimension to be accepted
    #         pac = np.vstack(
    #             [
    #                 self._calc_pac(
    #                     xx[i_batch * chunk_size : (i_batch + 1) * chunk_size][
    #                         np.newaxis
    #                     ]
    #                 )
    #                 for i_batch in range(n_chunks)
    #             ]
    #         )
    #     else:
    #         pac = np.vstack(
    #             [
    #                 self._calc_pac(
    #                     xx[i_batch * chunk_size : (i_batch + 1) * chunk_size]
    #                 )
    #                 for i_batch in range(n_chunks)
    #             ]
    #         )

    #     pac = self.xx_dim_handler.unfit(pac)

    #     # Takes mean across n_segments
    #     pac = pac.mean(axis=-3)

    #     # self.output_shape = pac.shape

    #     return pac

    # unfair for the use of ThreadPoolexecutor
    def calc_pac(self, xx, chunk_size=None):
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


if __name__ == "__main__":
    # Start
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )

    # Parameters
    BATCH_SIZE = 16
    N_CHS = 16
    N_SEGMENTS = 8
    T_SEC = 4
    device = "cuda"
    fp16 = True
    in_place = True
    N_calc = 5
    CHUNK_SIZE = 16
    PHA_N_BANDS = 50
    AMP_N_BANDS = 30

    # Demo Signal
    xx, tt, fs = mngs.dsp.demo_sig(
        batch_size=BATCH_SIZE,
        n_chs=N_CHS,
        t_sec=T_SEC,
        n_segments=N_SEGMENTS,
        sig_type="pac",
    )
    # (8, 19, 20, 2048)
    seq_len = xx.shape[-1]

    # Time stampers
    ts_tp = mngs.gen.TimeStamper()
    ts_mngs = mngs.gen.TimeStamper()

    ################################################################################
    # PAC Calculation
    ################################################################################
    # mngs
    model_mngs = MNGSHandler(
        seq_len=seq_len,
        fs=float(fs),
        ts=ts_mngs,
        device=device,
        fp16=fp16,
        pha_n_bands=PHA_N_BANDS,
        amp_n_bands=AMP_N_BANDS,
    )
    _xx = torch.tensor(xx).to(device)

    self = model_mngs
    self.ts(self.calc_start_str)
    with torch.no_grad():
        for _ in range(N_calc):
            model_mngs.calc_pac(_xx, chunk_size=CHUNK_SIZE)
    self.ts(self.calc_end_str)

    # tensorpac
    model_tp = TensorpacHandler(
        seq_len=seq_len,
        fs=float(fs),
        ts=ts_tp,
        device=device,
        fp16=fp16,
        pha_n_bands=PHA_N_BANDS,
        amp_n_bands=AMP_N_BANDS,
    )
    _xx = np.array(xx)
    self = model_tp
    self.ts(self.calc_start_str)
    for _ in range(N_calc):
        model_tp.calc_pac(_xx, chunk_size=CHUNK_SIZE)
    self.ts(self.calc_end_str)
    # koko

    stats = pd.concat([model_tp.stats, model_mngs.stats])

    assert np.allclose(model_tp.freqs_pha, model_mngs.freqs_pha)
    mngs.gen.print_block(
        "Error, fixme, np.allclose(model_tp.freqs_amp, model_mngs.freqs_amp)",
        c="red",
    )
    # assert np.allclose(model_tp.freqs_amp, model_mngs.freqs_amp)  # fixme

    ################################################################################
    # Compare mngs with tensorpac
    ################################################################################
    tt = "Tensorpac"
    mm = "mngs.dsp"

    # ################################################################################
    # # Input shape
    # ################################################################################
    # input_shapes = stats.input_shape
    # assert input_shapes[mm] == input_shapes[tt]
    # mngs.gen.print_block(f"Input shape: {input_shapes[mm]}")

    # ################################################################################
    # # Output shape
    # ################################################################################
    # output_shapes = stats.output_shape
    # assert output_shapes[mm] == output_shapes[tt]
    # mngs.gen.print_block(f"Output shape: {output_shapes[mm]}")

    # ################################################################################
    # # Precision of PAC values
    # ################################################################################
    # pac_values = stats.calculated_pac
    # i_batch, i_ch = 0, 0
    # indi = (i_batch, i_ch)

    # fig, axes = mngs.plt.subplots(ncols=3)

    # vmin = min(
    #     pac_values[mm][indi].min(),
    #     pac_values[tt][indi].min(),
    #     (pac_values[mm][indi] - pac_values[tt][indi]).min(),
    # )
    # vmax = max(
    #     pac_values[mm][indi].max(),
    #     pac_values[tt][indi].max(),
    #     (pac_values[mm][indi] - pac_values[tt][indi]).max(),
    # )

    # axes[0].imshow2d(pac_values[mm][indi], vmin=vmin, vmax=vmax, cbar=False)
    # axes[1].imshow2d(pac_values[tt][indi], vmin=vmin, vmax=vmax, cbar=False)
    # axes[2].imshow2d(
    #     pac_values[mm][indi] - pac_values[tt][indi], vmin=vmin, vmax=vmax
    # )

    # plt.show()

    ################################################################################
    # Calculation Speed
    ################################################################################
    calc_times_mean_sec = stats.calc_time_mean_sec
    time_ratio = (
        calc_times_mean_sec["mngs.dsp"] / calc_times_mean_sec["Tensorpac"]
    )
    speed_ratio = 1 / time_ratio
    mngs.gen.print_block(
        f"Speed ratio (mngs.dsp / tensorpac): {speed_ratio:.3f}"
    )
    # ----------------------------------------
    # Speed ratio (mngs.dsp / tensorpac): 103.408
    # ----------------------------------------

# EOF
