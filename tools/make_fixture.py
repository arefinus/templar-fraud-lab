"""Regenerate the authored synthetic fixture (examples/fixtures/v1).

Usage: python tools/make_fixture.py [--rows 6000] [--seed 42] [--out examples/fixtures/v1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from templar_fraud_lab.data.synthetic import DEFAULT_ROWS, DEFAULT_SEED, write_fixture  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out", default="examples/fixtures/v1")
    a = p.parse_args()
    path = write_fixture(a.out, n_rows=a.rows, seed=a.seed)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
