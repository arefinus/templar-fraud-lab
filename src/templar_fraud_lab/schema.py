"""Canonical column schema shared by the fixture and the dataset adapters.

Every adapter maps its source into this schema so that the rest of the pipeline
(splits, preprocessing, models, templates, verifier) is dataset-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MISSING_TOKEN = "__MISSING__"

ROW_ID = "row_id"
TIMESTAMP = "ts"
ENTITY_ID = "card_id"
LABEL = "is_fraud"


@dataclass(frozen=True)
class FeatureSchema:
    """Which columns play which role. Order is significant and is preserved by the
    preprocessor (tests assert the exact feature order)."""

    numeric: tuple[str, ...]
    low_card: tuple[str, ...]
    high_card: tuple[str, ...]
    target_encoded: tuple[str, ...] = field(default_factory=tuple)

    @property
    def all_features(self) -> tuple[str, ...]:
        return self.numeric + self.low_card + self.high_card

    def validate_frame(self, columns: list[str]) -> None:
        required = [ROW_ID, TIMESTAMP, LABEL, *self.all_features]
        missing = [c for c in required if c not in columns]
        if missing:
            raise ValueError(
                "frame is missing required columns "
                f"{missing}; present columns: {sorted(columns)}"
            )


FIXTURE_SCHEMA = FeatureSchema(
    numeric=(
        "amount",
        "account_age_days",
        "velocity_1h",
        "ip_risk_score",
        "country_match",
        "hour",
        "dow",
        "time_since_last_card_event",
    ),
    low_card=("merchant_category", "device_type"),
    high_card=("email_domain",),
    target_encoded=("email_domain",),
)

FIXTURE_NUMERIC_RAW = (
    "amount",
    "account_age_days",
    "velocity_1h",
    "ip_risk_score",
    "country_match",
)
