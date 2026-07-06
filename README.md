# lume-bmad
Tools for using Bmad in LUME

## Installation

This package expects `bmad` and `pytao` to be provided by conda.

```bash
conda create -n lume-bmad -c conda-forge python=3.11 bmad pytao
conda activate lume-bmad
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

Quick import check:

```bash
python -c "import pytao, lume_bmad; print('ok')"
```
