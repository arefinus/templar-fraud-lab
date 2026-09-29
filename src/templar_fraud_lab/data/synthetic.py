"""Authored synthetic fixture: fictional transactions with learnable fraud structure.

Fictional identifiers (`SYN-TX-000001`, `SYN-CARD-0001`). Not derived from any
person, company, account or dataset. Prevalence about 2%. The last 8% of the
timeline introduces a merchant category unseen earlier (`crypto_exchange`) and a
handful of very large amounts, so that unknown-category and out-of-range
handling are exercised on the test slice rather than assumed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from templar_fraud_lab.data.temporal import derive_temporal_features
from templar_fraud_lab.schema import ENTITY_ID, FIXTURE_SCHEMA, LABEL, ROW_ID, TIMESTAMP

FIXTURE_VERSION = "v1"
FIXTURE_FILE = "transactions.csv"
DEFAULT_ROWS = 6_000
DEFAULT_SEED = 42
SPAN_DAYS = 180
N_CARDS = 900
MERCHANT_CATEGORIES = ("grocery", "fuel", "electronics", "travel", "gaming",
                       "apparel", "restaurants", "digital_goods")
DRIFT_CATEGORY = "crypto_exchange"
DEVICE_TYPES = ("mobile_app", "web", "pos", "emulator")
EMAIL_DOMAINS = tuple(f"demo-mail-{i:02d}.example" for i in range(40))
INTERCEPT = -6.1  # tuned by hand for about 2% prevalence at seed 42


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_fixture(n_rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.integers(0, SPAN_DAYS * 86_400, size=n_rows))
    card_idx = rng.integers(0, N_CARDS, size=n_rows)
    amount = np.round(np.exp(rng.normal(3.6, 1.0, size=n_rows)), 2)
    account_age = rng.integers(1, 2_000, size=n_rows)
    device = rng.choice(len(DEVICE_TYPES), size=n_rows, p=[0.45, 0.30, 0.20, 0.05])
    merchant = rng.choice(len(MERCHANT_CATEGORIES), size=n_rows)
    country_match = (rng.random(n_rows) < 0.93).astype(int)
    velocity = rng.poisson(0.6, size=n_rows)
    ip_risk = np.clip(rng.beta(2, 8, size=n_rows) + 0.4 * (device == 3), 0, 1)
    email = rng.choice(len(EMAIL_DOMAINS), size=n_rows, p=_domain_weights(len(EMAIL_DOMAINS)))

    late = ts > (SPAN_DAYS * 86_400 * 0.92)
    drift_rows = late & (rng.random(n_rows) < 0.08)
    merchant_names = np.array(MERCHANT_CATEGORIES, dtype=object)[merchant].astype(object)
    merchant_names[drift_rows] = DRIFT_CATEGORY
    spike_rows = np.flatnonzero(late)[rng.random(late.sum()) < 0.01]
    amount[spike_rows] = np.round(amount[spike_rows] * 40, 2)

    logit = (
        INTERCEPT
        + 0.9 * (np.log1p(amount) - 3.6)
        + 1.6 * (country_match == 0)
        + 0.7 * velocity
        + 1.4 * (device == 3)
        + 2.2 * ip_risk
        + 0.8 * (account_age < 30)
        + 0.5 * np.isin(merchant, (2, 4, 7))
        + 1.0 * drift_rows
        + rng.normal(0, 0.5, size=n_rows)
    )
    label = (rng.random(n_rows) < _sigmoid(logit)).astype(int)

    frame = pd.DataFrame({
        ROW_ID: [f"SYN-TX-{i + 1:06d}" for i in range(n_rows)],
        "event_time": pd.to_datetime("2025-01-06") + pd.to_timedelta(ts, unit="s"),
        TIMESTAMP: ts.astype(np.int64),
        ENTITY_ID: [f"SYN-CARD-{c + 1:04d}" for c in card_idx],
        "amount": amount,
        "merchant_category": merchant_names,
        "device_type": np.array(DEVICE_TYPES, dtype=object)[device],
        "country_match": country_match,
        "account_age_days": account_age,
        "velocity_1h": velocity,
        "email_domain": np.array(EMAIL_DOMAINS, dtype=object)[email],
        "ip_risk_score": np.round(ip_risk, 4),
        LABEL: label,
    })
    frame.loc[rng.random(n_rows) < 0.05, "email_domain"] = np.nan
    frame.loc[rng.random(n_rows) < 0.10, "ip_risk_score"] = np.nan
    frame["event_time"] = frame["event_time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return frame


def _domain_weights(k: int) -> np.ndarray:
    w = 1.0 / np.arange(1, k + 1)
    return w / w.sum()


def write_fixture(out_dir: str | Path, n_rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = generate_fixture(n_rows=n_rows, seed=seed)
    path = out / FIXTURE_FILE
    frame.to_csv(path, index=False, lineterminator="\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = {
        "version": FIXTURE_VERSION,
        "generator": "tools/make_fixture.py",
        "seed": seed,
        "rows": int(len(frame)),
        "fraud_rows": int(frame[LABEL].sum()),
        "prevalence": round(float(frame[LABEL].mean()), 5),
        "sha256": digest,
        "columns": list(frame.columns),
    }
    (out / "fixture_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (out / "FIXTURE.md").write_text(_fixture_md(seed, n_rows), encoding="utf-8")
    return path


def _fixture_md(seed: int, n_rows: int) -> str:
    return (
        "# Fixture v1\n\n"
        "Authored synthetic demonstration data. Fictional identifiers. Not derived from any "
        "person, company, account or dataset. Generated by `tools/make_fixture.py` seed "
        f"`{seed}`, version v1.\n\n"
        f"Rows: {n_rows}. Columns: see `fixture_meta.json`. Label `is_fraud` is drawn from a "
        "hand-written logistic model over amount, country mismatch, velocity, device type, "
        "IP risk, account age and merchant category, with Gaussian noise, so the structure is "
        "learnable but not trivially separable. The final 8% of the timeline introduces a "
        "merchant category absent earlier and a few very large amounts to exercise unknown-"
        "category and out-of-range handling on the test slice.\n"
    )


def load_fixture(fixture_dir: str | Path) -> pd.DataFrame:
    path = Path(fixture_dir) / FIXTURE_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"fixture not found at {path}; run `python -m templar_fraud_lab make-fixture`"
        )
    frame = pd.read_csv(path)
    frame = frame.sort_values([TIMESTAMP, ROW_ID], kind="mergesort").reset_index(drop=True)
    frame = derive_temporal_features(frame)
    FIXTURE_SCHEMA.validate_frame(list(frame.columns))
    return frame
