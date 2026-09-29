"""Adapter for the IEEE-CIS Fraud Detection competition data (Kaggle).

Reads `train_transaction.csv` and `train_identity.csv` from the data root, left-
merges identity onto transactions on `TransactionID`, and maps into the canonical
schema. Nothing from IEEE-CIS is shipped here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from templar_fraud_lab.data.errors import DatasetNotAvailable, SchemaMismatch
from templar_fraud_lab.schema import ENTITY_ID, LABEL, ROW_ID, TIMESTAMP, FeatureSchema

IEEE_SOURCE = "https://www.kaggle.com/c/ieee-fraud-detection (competition data)"
IEEE_TERMS = "Kaggle competition rules; download requires a Kaggle account"
TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"

IEEE_LOW_CARD = ("ProductCD", "card4", "card6", "M1", "M2", "M3", "M4", "M5", "M6", "M7",
                 "M8", "M9", "DeviceType")
IEEE_HIGH_CARD = ("P_emaildomain", "R_emaildomain", "addr1", "DeviceInfo", "id_30", "id_31")


def _card_key(frame: pd.DataFrame) -> pd.Series:
    """A card-like entity key built from the card1..card6 columns plus addr1.

    This is the common community construction; the paper does not state its
    exact card key (audit, ambiguity 11)."""
    parts = [frame[c].astype(str) for c in ("card1", "card2", "card3", "card5", "addr1")
             if c in frame.columns]
    key = parts[0]
    for p in parts[1:]:
        key = key + "|" + p
    return key


def load_ieee_cis(data_root: str | Path) -> tuple[pd.DataFrame, FeatureSchema]:
    root = Path(data_root)
    tx_path, id_path = root / TRANSACTION_FILE, root / IDENTITY_FILE
    if not tx_path.exists():
        raise DatasetNotAvailable("IEEE-CIS transactions", str(tx_path), IEEE_SOURCE, IEEE_TERMS)
    if not id_path.exists():
        raise DatasetNotAvailable("IEEE-CIS identity", str(id_path), IEEE_SOURCE, IEEE_TERMS)
    tx = pd.read_csv(tx_path)
    ident = pd.read_csv(id_path)
    for name, frame, required in (
        ("IEEE-CIS transactions", tx, ["TransactionID", "TransactionDT", "isFraud", "TransactionAmt"]),
        ("IEEE-CIS identity", ident, ["TransactionID"]),
    ):
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise SchemaMismatch(name, missing, list(frame.columns))
    merged = tx.merge(ident, on="TransactionID", how="left")
    merged[ROW_ID] = "IEEE-" + merged["TransactionID"].astype(str)
    merged[TIMESTAMP] = merged["TransactionDT"].astype(int)
    merged[LABEL] = merged["isFraud"].astype(int)
    merged[ENTITY_ID] = _card_key(merged)
    numeric = tuple(
        c for c in merged.columns
        if c not in {ROW_ID, TIMESTAMP, LABEL, ENTITY_ID, "TransactionID", "TransactionDT", "isFraud"}
        and c not in IEEE_LOW_CARD and c not in IEEE_HIGH_CARD
        and pd.api.types.is_numeric_dtype(merged[c])
    )
    schema = FeatureSchema(
        numeric=numeric,
        low_card=tuple(c for c in IEEE_LOW_CARD if c in merged.columns),
        high_card=tuple(c for c in IEEE_HIGH_CARD if c in merged.columns),
        target_encoded=tuple(c for c in IEEE_HIGH_CARD if c in merged.columns),
    )
    return merged, schema
