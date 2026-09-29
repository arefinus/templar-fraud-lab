# Reproducibility

## What this repository reproduces
Nothing from the source paper is reproduced here. Every number this repository produces
comes from the authored synthetic fixture (`examples/fixtures/v1/`, seed 42) and carries the
evidence label `demo`. Paper-reported values live only in `evidence/paper_reported/` with a
source locator and are never used as a test oracle.

## Exact commands
```
pip install -e . && pip install pytest      # or prefix commands with PYTHONPATH=src
python -m templar_fraud_lab make-fixture    # regenerates examples/fixtures/v1 (seed 42)
python -m templar_fraud_lab smoke           # under 30 s, validates outputs
python -m templar_fraud_lab demo            # full pipeline, writes examples/output/
python -m templar_fraud_lab train
python -m templar_fraud_lab evaluate
python -m templar_fraud_lab audit
python -m templar_fraud_lab figures
python -m pytest -q
```

## Seeds and determinism
Fixture seed 42. Model seeds are set in `configs/demo.yaml`; the paper's seeds
{11, 23, 37, 53, 79} are recorded in the `configs/paper_*.yaml` files as settings only.
CatBoost and torch on CPU are deterministic for a fixed seed and thread count; set
`OMP_NUM_THREADS=1` for bit-identical reruns.

## Run manifest
Every `smoke`, `demo` and `train` run writes `run_manifest.yaml` with mode, evidence
status, fixture hash, configuration hash, seed and sample counts. `git_commit` is null
until the publishing tooling fills it; it is never fabricated.

## Running on real data
Point `--data-root` at a directory holding BAF Base or IEEE-CIS in the layout described in
`DATA_CARD.md`. The adapter validates columns and refuses anything else. Results from such
a run are `reproduced` only if the run manifest, data hash and config hash are kept with
them. The paper's full protocol (TPE search, 50 trials, five seeds) is not executed by the
demo configs and would need the paper configs plus hours of CPU or a GPU.

## Environment
Python 3.10 locally; CI runs 3.10, 3.11 and 3.12 on Ubuntu with torch CPU wheels. The
paper's environment is recorded in `docs/method.md`, not reproduced here.
