"""Actionable data errors. Missing data is a hard failure, never a silent skip."""

from __future__ import annotations


class DatasetNotAvailable(FileNotFoundError):
    """Raised when a real dataset is not present under the data root."""

    def __init__(self, dataset: str, expected: str, source: str, terms: str) -> None:
        msg = (
            f"{dataset} was not found: expected {expected}. This repository ships no "
            f"dataset rows. Obtain the data from {source} (terms: {terms}) and point "
            "--data-root (or DATA_ROOT) at the directory containing it."
        )
        super().__init__(msg)
        self.dataset = dataset
        self.source = source


class SchemaMismatch(ValueError):
    """Raised when a file is present but does not have the expected columns."""

    def __init__(self, dataset: str, missing: list[str], present: list[str]) -> None:
        super().__init__(
            f"{dataset}: missing required columns {missing}. Present columns "
            f"(first 20): {present[:20]}"
        )
        self.missing = missing
