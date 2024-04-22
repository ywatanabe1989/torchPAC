#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-22 22:06:26"
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
