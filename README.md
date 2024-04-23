# torchPAC: Fast and Learnable calculation of Phase-Amplitude Coupling

#### Calculation environment
```bash
$ python -m venv env && source ./env/bin/activate && python -m pip install -U pip && pip install -r requirements.txt
```

# Introduction

./scripts/monitor_processers.py

# Methods
<!-- - Implementation explanations
 !--   - [ ] Bandpass Filtering
 !--   - [ ] Hilbert Transformation
 !--   - [ ] Mutual Index -->
  
<!-- - Machine Specs
 !--   - [x] Rocky Linux v9.3
 !--       - [x] kernel version: 5.14.0-362.24.1.el9_3.x86_64
 !--   - [x] CPU: AMD Ryzen 9 7950X 16-Core Processor
 !--   - [x] GPU: NVIDIA GeForce RTX 4090
 !--   - [`./scripts/resource_info/resource_info.yaml`](./scripts/resource_info/resource_info.yaml) -->

<!-- - [ ] Data Preparation
 !--   - [x] Synthetic Signals
 !--     - mngs.dsp.demo_sig("pac")
 !--     - mngs.dsp.demo_sig("tensorpac")
 !--   - [ ] **Real Neuronal Signals** -->

# Results
- Parameters

- Precision of PAC values (MSE)
  - [ ] RMSE, MAE, correlation coefficients between the tensorpac package
  
- Computational Speed
  - [ ] Comparison with the tensorpac package and mngs modes

- CPU, GPU, RAM, and VRAM

  
- Training PAC Net
  - [ ] Real world data

# Discussion


### Evaluation Metrics

- **PAC Values**: 
- **Calculation Speed**: 
- **RAM Usage**: 

## Contact
Yusuke Watanabe (Yusuke.Watanabe@unimelb.edu.au)


## Fixme
```
########################################
## 2024Y-04M-23D-12h46m06s_1ao4
########################################


----------------------------------------
{'batch_size': 2, 'n_chs': 2, 'n_segments': 3, 't_sec': 2, 'fs': 512, 'pha_n_bands': 30, 'amp_n_bands': 70, 'n_perm': None, 'chunk_size': 2, 'fp16': False, 'no_grad': True, 'in_place': True, 'trainable': False, 'device': 'cpu', 'use_threads': True, 'package': 'mngs', 'ts': <mngs.gen._TimeStamper.TimeStamper object at 0x7ffb0492b2e0>}
----------------------------------------

/home/ywatanabe/proj/mngs/src/mngs/general/_converters.py:88: UserWarning: Converted from ndarray to Tensor (cpu). You might want to consider using Tensor (cpu) as input for faster computation.
  warnings.warn(

----------------------------------------
mngs
0.099 +/- 0.002 sec
----------------------------------------


Saved to: /home/ywatanabe/proj/entrance/torchPAC/scripts/main/RUNNING/2024Y-04M-23D-12h46m06s_1ao4/stats.csv


----------------------------------------
Congratulations! The script completed.

/home/ywatanabe/proj/entrance/torchPAC/scripts/main/2024Y-04M-23D-12h46m06s_1ao4/
----------------------------------------
```
