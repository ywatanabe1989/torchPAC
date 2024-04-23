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

# Experiment

``` bash
sudo nvidia-smi -pm 1
ss run_experiment
./scripts/run_experiment.sh
./scripts/summarize.py
```

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


## Note
Amdahl's law

## Fixme
no_grad?
