from typing import Union
import numpy as np
import torch
import mngs
from scripts.Handlers import MNGSHandler, TensorpacHandler

def init_model(params: dict) -> Union[MNGSHandler, TensorpacHandler]:
    try:
        params_h = params.copy()
        params_h.update({
            "pha_min_hz": 2,
            "pha_max_hz": 20,
            "amp_min_hz": 80,
            "amp_max_hz": 160
        })

        for key in ["batch_size", "n_chs", "n_segments", "t_sec", "package", "no_grad", "n_calc"]:
            params_h.pop(key, None)

        if params["package"] == "mngs":
            return MNGSHandler(**params_h)
        elif params["package"] == "tensorpac":
            return TensorpacHandler(**params_h)
    except Exception as exception:
        print(exception)
        return None
