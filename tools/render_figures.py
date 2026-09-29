"""Render original SVG architecture diagrams into docs/figures/.

The three drawings adapt the STRUCTURE of Figures 1 to 3 of the source article
(Discover Data 4:20, 2026, DOI 10.1007/s44248-026-00117-y, CC BY 4.0). They are
drawn from scratch here; no publisher artwork is copied. Hand-written SVG, no
dependencies.

Usage: python tools/render_figures.py [--out docs/figures]
"""

from __future__ import annotations

import argparse
from pathlib import Path

FONT = "font-family='Helvetica, Arial, sans-serif'"
INK, ACCENT, WARN, PALE, LIGHT = "#1F2A37", "#2E5A88", "#B4531F", "#EEF3F8", "#F7F7F5"
ATTRIBUTION = ("Original drawing adapted from the structure of the source article's figure "
               "(Discover Data 4:20, 2026, CC BY 4.0). Redrawn by tools/render_figures.py.")


def box(x: int, y: int, w: int, h: int, title: str, sub: str = "", fill: str = PALE, stroke: str = ACCENT) -> str:
    parts = [f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' fill='{fill}' stroke='{stroke}' stroke-width='1.5'/>",
             f"<text x='{x + w / 2}' y='{y + (h / 2 - 4 if sub else h / 2 + 4)}' text-anchor='middle' {FONT} font-size='12' font-weight='bold' fill='{INK}'>{title}</text>"]
    if sub:
        parts.append(f"<text x='{x + w / 2}' y='{y + h / 2 + 12}' text-anchor='middle' {FONT} font-size='10' fill='{INK}'>{sub}</text>")
    return "\n".join(parts)


def arrow(x1: int, y1: int, x2: int, y2: int, label: str = "", color: str = INK) -> str:
    s = f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-width='1.4' marker-end='url(#arrow)'/>"
    if label:
        s += f"<text x='{(x1 + x2) / 2}' y='{(y1 + y2) / 2 - 6}' text-anchor='middle' {FONT} font-size='9' fill='{color}'>{label}</text>"
    return s


def svg(width: int, height: int, title: str, body: str) -> str:
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>\n"
            f"<defs><marker id='arrow' markerWidth='8' markerHeight='8' refX='7' refY='4' orient='auto'>"
            f"<path d='M0,0 L8,4 L0,8 z' fill='{INK}'/></marker></defs>\n"
            f"<rect width='{width}' height='{height}' fill='white'/>\n"
            f"<text x='16' y='24' {FONT} font-size='14' font-weight='bold' fill='{INK}'>{title}</text>\n{body}\n"
            f"<text x='16' y='{height - 10}' {FONT} font-size='8' fill='#555'>{ATTRIBUTION}</text>\n</svg>\n")


def fig1_pipeline() -> str:
    y = 70
    b = [box(20, y, 120, 56, "Transactions", "chronological"),
         box(170, y, 130, 56, "Preprocess", "train-only fit"),
         box(330, y, 130, 56, "SAINT-style", "encoder z, p_z"),
         box(490, y, 130, 56, "CatBoost triage", "p_triage on [z, x]"),
         box(650, y, 120, 56, "Routing", "Eq. 5 or band"),
         box(800, 20, 130, 56, "Fast path", "no rationale", LIGHT, INK),
         box(800, 120, 130, 56, "Escalation", "templates + verifier", LIGHT, WARN),
         box(960, y, 130, 56, "Stacking", "available sources"),
         box(1120, y, 120, 56, "OT calibration", "fit on validation"),
         box(1270, y, 120, 56, "Decision", "fraud / legit / review")]
    a = [arrow(140, y + 28, 170, y + 28), arrow(300, y + 28, 330, y + 28), arrow(460, y + 28, 490, y + 28),
         arrow(620, y + 28, 650, y + 28), arrow(770, y + 20, 800, 48, "fast"), arrow(770, y + 36, 800, 148, "escalate", WARN),
         arrow(930, 48, 960, y + 20), arrow(930, 148, 960, y + 36), arrow(1090, y + 28, 1120, y + 28),
         arrow(1240, y + 28, 1270, y + 28)]
    note = (f"<text x='20' y='215' {FONT} font-size='10' fill='{INK}'>Splits: 70/15/15 chronological; validation drives early stopping, stacking, "
            f"calibration and the cost threshold; the test slice is touched once.</text>")
    return svg(1410, 240, "Figure 1 (adapted): routed, verifier-grounded, calibrated and cost-sensitive pipeline", "\n".join(b + a) + note)


def fig2_escalation() -> str:
    b = [box(30, 60, 150, 56, "Escalated row", "features + timestamp"),
         box(220, 60, 170, 56, "Template bank", "allowlisted predicates"),
         box(430, 60, 170, 56, "Fired templates", "structured rationales"),
         box(640, 60, 170, 56, "Verifier", "schema, support, range, consistency"),
         box(860, 20, 170, 56, "PASS", "p_escalation available", "#E8F3EA", "#2F7D4F"),
         box(860, 120, 170, 56, "FAIL", "review; no verdict", "#F9E9E3", WARN),
         box(1070, 20, 150, 56, "Stacking", "3 sources"),
         box(1070, 120, 150, 56, "Human review", "reason codes attached", LIGHT, INK)]
    a = [arrow(180, 88, 220, 88), arrow(390, 88, 430, 88), arrow(600, 88, 640, 88), arrow(810, 80, 860, 48, "all pass", "#2F7D4F"),
         arrow(810, 96, 860, 148, "any check fails", WARN), arrow(1030, 48, 1070, 48), arrow(1030, 148, 1070, 148)]
    notes = [f"<text x='30' y='215' {FONT} font-size='10' fill='{INK}'>Rationale = (template_id, feature_ids, observed_values, predicates, verification_status, reason_codes).</text>",
             f"<text x='30' y='232' {FONT} font-size='10' fill='{INK}'>Predicates are evaluated by a tiny interpreter over an allowlisted AST; the verifier has no parameters and re-evaluates every predicate.</text>"]
    return svg(1250, 260, "Figure 2 (adapted): escalation path with verifier pass/fail branch", "\n".join(b + a + notes))


def fig3_stacking() -> str:
    b = [box(30, 40, 150, 48, "p_triage", "always present"),
         box(30, 110, 150, 48, "p_escalation", "optional, verified only", LIGHT, WARN),
         box(30, 180, 150, 48, "p_z", "encoder head"),
         box(260, 100, 200, 68, "Bayesian stacking", "Dirichlet(1) prior, weights renormalised over available sources"),
         box(520, 100, 190, 68, "1-D OT calibration", "quantile matching, K knots, min support"),
         box(770, 100, 190, 68, "Cost threshold t*", "argmin C_FP FP + C_FN FN + C_R R"),
         box(1020, 40, 140, 48, "fraud", "p >= t*", "#F9E9E3", WARN),
         box(1020, 110, 140, 48, "legitimate", "p < t*", "#E8F3EA", "#2F7D4F"),
         box(1020, 180, 140, 48, "review", "verifier fail or high disagreement", LIGHT, INK)]
    a = [arrow(180, 64, 260, 120), arrow(180, 134, 260, 134, "mask", WARN), arrow(180, 204, 260, 148),
         arrow(460, 134, 520, 134, "p_stacked, u"), arrow(710, 134, 770, 134, "p_calibrated"),
         arrow(960, 124, 1020, 64), arrow(960, 134, 1020, 134), arrow(960, 144, 1020, 204)]
    notes = [f"<text x='30' y='262' {FONT} font-size='10' fill='{INK}'>Stacking weights and the calibration map are fitted on the validation split; the threshold grid is [0.001, 0.500] step 0.001 with lowest-threshold tie-break.</text>",
             f"<text x='30' y='279' {FONT} font-size='10' fill='{INK}'>Missing p_escalation is a mask, never a zero; the disagreement proxy is the availability-weighted spread of the sources.</text>"]
    return svg(1190, 300, "Figure 3 (adapted): stacking, calibration and cost-sensitive decision", "\n".join(b + a + notes))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="docs/figures")
    out = Path(p.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("fig1_pipeline.svg", fig1_pipeline), ("fig2_escalation.svg", fig2_escalation),
                     ("fig3_stacking_calibration_decision.svg", fig3_stacking)):
        (out / name).write_text(fn(), encoding="utf-8")
        print(f"wrote {out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
