"""Adapter for the Bank Account Fraud (BAF) Base table.

BAF is a synthetic bank ACCOUNT-OPENING benchmark released by Jesus et al.
(NeurIPS 2022 Datasets and Benchmarks), not a table of observed payments. The
Base variant has a `month` column (0..7) and `fraud_bool` plus 30 raw features.
Nothing from BAF is shipped here; the file must be obtained by the user.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from templar_fraud_lab.data.errors import DatasetNotAvailable, SchemaMismatch
from templar_fraud_lab.schema import ENTITY_ID, LABEL, ROW_ID, TIMESTAMP, FeatureSchema

BAF_SOURCE = "https://github.com/feedzai/bank-account-fraud (Kaggle mirror: sgpjesus/bank-account-fraud-dataset-neurips-2022)"
BAF_TERMS = "CC BY-NC-SA 4.0, see the feedzai repository"
BAF_FILE = "Base.csv"

BAF_NUMERIC = (
    "income", "name_email_similarity", "prev_address_months_count",
    "current_address_months_count", "customer_age", "days_since_request",
    "intended_balcon_amount", "zip_count_4w", "velocity_6h", "velocity_24h",
    "velocity_4w", "bank_branch_count_8w", "date_of_birth_distinct_emails_4w",
    "credit_risk_score", "email_is_free", "phone_home_valid", "phone_mobile_valid",
    "bank_months_count", "has_other_cards", "proposed_credit_limit",
    "foreign_request", "session_length_in_minutes", "keep_alive_session",
    "device_distinct_emails_8w", "device_fraud_count",
)
BAF_LOW_CARD = ("payment_type", "employment_status", "housing_status", "source", "device_os")
BAF_RAW_FEATURES = BAF_NUMERIC + BAF_LOW_CARD  # 30 raw features

BAF_SCHEMA = FeatureSchema(
    numeric=BAF_NUMERIC + ("month",),
    low_card=BAF_LOW_CARD,
    high_card=(),
    target_encoded=(),
)


def load_baf(data_root: str | Path) -> tuple[pd.DataFrame, FeatureSchema]:
    """Load BAF Base into the canonical schema.

    BAF carries `month` only: there is no day-of-week, no card, device or IP
    identifier. The adapter therefore uses `month` as the chronological key and
    a synthetic-free, row-position entity id (no entity graph is possible). This
    is recorded in docs/paper-implementation-audit.md, ambiguity 2.
    """
    path = Path(data_root) / BAF_FILE
    if not path.exists():
        raise DatasetNotAvailable("BAF Base", str(path), BAF_SOURCE, BAF_TERMS)
    frame = pd.read_csv(path)
    required = ["month", "fraud_bool", *BAF_RAW_FEATURES]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise SchemaMismatch("BAF Base", missing, list(frame.columns))
    out = frame.copy()
    out[ROW_ID] = [f"BAF-{i:07d}" for i in range(len(out))]
    # month is the only chronological signal; stable order within a month by row.
    out[TIMESTAMP] = out["month"].astype(int) * 10_000_000 + out.index.to_numpy()
    out[LABEL] = out["fraud_bool"].astype(int)
    out[ENTITY_ID] = out[ROW_ID]
    return out, BAF_SCHEMA
