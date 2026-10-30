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

## Implementation example

The test fixtures in `tests/` provide a minimal Tao setup you can use directly:

- `tests/fodo.init`: Tao initialization file.
- `tests/fodo.bmad`: lattice file loaded by `fodo.init`.

Because `tests/fodo.init` references `fodo.bmad` with a relative path, keep those two files in the same directory.

```python
from pathlib import Path

from pytao import Tao

from lume_bmad.actions import EleScalarVariable
from lume_bmad.model import LUMEBmadModel

# Reuse the fixture Tao setup from tests/
tao_init = Path("tests") / "fodo.init"
tao = Tao(init_file=str(tao_init), noplot=True)

control_variables = [
    EleScalarVariable(
        name="qf:B1_GRADIENT",
        unit="1/m^2",
        element_name="qf",
        property_name="B1_GRADIENT",
    ),
    EleScalarVariable(
        name="qd:B1_GRADIENT",
        unit="1/m^2",
        element_name="qd",
        property_name="B1_GRADIENT",
    ),
]

model = LUMEBmadModel(
    tao=tao,
    action_variables=control_variables,
    dump_locations=["qf", "qd"],
)

# Read and set model variables
print(model.get(["qf:B1_GRADIENT", "qd:B1_GRADIENT"]))
model.set({"qf:B1_GRADIENT": 0.2})

# Enable beam tracking and read a dumped beam distribution
model.set({"track_type": "beam"})
qf_beam = model.get_value("qf_beam")
print(qf_beam.n_particle)
```

If you want to run this example from outside the repository, copy both fixture files together and pass the path to that copied `fodo.init`.

## Archiving and restoring

```python
from lume_bmad.archive import read_archive_variables
from lume_bmad.utils import tao_from_archive

model.archive("model.h5")
frame = read_archive_variables("model.h5")
print(frame.attrs["errors"])  # Failed variable names mapped to saved errors.
tao = tao_from_archive("model.h5")
```

To skip reading and storing current variable values while retaining names and
all descriptor metadata (including defaults):

```python
model.archive("model.h5", include_variable_values=False)
# Also supported: archive_model(model, "model.h5", include_variable_values=False)
```
