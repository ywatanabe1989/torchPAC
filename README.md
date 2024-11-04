<!-- ---
!-- title: README.md
!-- author: ywatanabe
!-- date: 2024-11-04 14:54:29
!-- --- -->


# TorchPAC: Fast and Learnable calculation of Phase-Amplitude Coupling

## Manuscript
[`manuscript.pdf`](./paper/manuscript/main/manuscript.pdf)

## Installation
See [installation guide](./docs/installation.md)

## Experiment
### Parameter Space
./config/PARAMETER_SPACES.yaml

### Run Experiments
```bash
sudo nvidia-smi -pm 1
ss torch_PAC_run_experiment 
./scripts/run_experiment.sh
```

## Results
### Parameters

### Precision of PAC values (MSE)
- [ ] RMSE, MAE, correlation coefficients between the tensorpac package
  
### Computational Speed
- [ ] Comparison with the tensorpac package and mngs modes

### CPU, GPU, RAM, and VRAM usage
  
### Training PAC Net
- [ ] Real world data

## Discussion

### Evaluation Metrics
- PAC Values
- Calculation Speed
- RAM Usage

## Contact
Yusuke Watanabe (Yusuke.Watanabe@unimelb.edu.au)
