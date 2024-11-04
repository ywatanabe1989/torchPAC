#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-04 15:13:02 (ywatanabe)"
# File: ./torchPAC/scripts/utils/perform_pac_calculation.py

import torch


def perform_pac_calculation(model, signal, params):
    try:
        for _ in range(params["n_calc"]):
            model.ts(model.calc_start_str)
            if params["no_grad"]:
                with torch.no_grad():
                    model.calc_pac(signal)
            else:
                model.calc_pac(signal)
            model.ts(model.calc_end_str)
    except Exception as exception:
        print(f"Error in PAC calculation: {exception}")


# EOF
