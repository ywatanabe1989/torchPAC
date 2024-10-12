# torchPAC: Fast and Learnable calculation of Phase-Amplitude Coupling


#### Installation
``` bash
git clone git@github.com:ywatanabe1989/torchPAC.git && \
    cd torchPAC

PYTHON_VERSION=3.12 && \
    python -m venv .env-"$PYTHON_VERSION" && \
    ln -s .env-"$PYTHON_VERSION" .env && \
    source ./.env/bin/activate && \
    python -m pip install -U pip && \
    pip install -Ur requirements.txt
```


# Experiment

``` bash
sudo nvidia-smi -pm 1
ss torch_PAC_run_experiment 
./scripts/run_experiment.sh
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
