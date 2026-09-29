# Contributing

Issues and pull requests are welcome. Please keep to the rules this repository is built on:

1. No dataset rows in the repository. Adapters read from `--data-root`; only the authored
   synthetic fixture under `examples/fixtures/` is committed.
2. No invented numbers. A metric appears in prose only if a script in this repository
   computed it on the fixture (and the text says so) or it is a paper-reported value in
   `evidence/paper_reported/` rendered with the label `paper-reported`.
3. Leakage discipline. Fit on training rows only, select on validation only, touch the
   test slice once. Add a test when you touch any fitting code.
4. No `eval`, `exec` or pickle of untrusted input. Predicates go through the allowlisted
   AST in `templar_fraud_lab/templates/predicate.py`.
5. Keep modules under about 300 lines and tests behavioural (no `assert True`).

Before opening a pull request run:

```
python -m pytest -q
python -m templar_fraud_lab smoke
```

Both must pass on Python 3.10, 3.11 and 3.12 (the CI matrix).
