## Installation

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

