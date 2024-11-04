from typing import Union
import numpy as np
import torch
import mngs
from scripts.Handlers import MNGSHandler, TensorpacHandler

def prepare_signal(params: dict) -> Union[np.ndarray, torch.Tensor]:
    try:
        signal, _, _ = mngs.dsp.demo_sig(
            sig_type="pac",
            batch_size=params["batch_size"],
            n_chs=params["n_chs"],
            n_segments=params["n_segments"],
            t_sec=params["t_sec"],
            fs=params["fs"],
        )

        if params["package"] == "tensorpac":
            return np.array(signal, dtype=np.float16 if params["fp16"] else np.float32)
        elif params["package"] == "mngs":
            return torch.tensor(signal, dtype=torch.float16 if params["fp16"] else torch.float32).to(params["device"])
    except Exception as exception:
        print(exception)
        return None
