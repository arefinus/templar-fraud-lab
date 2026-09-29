"""A small Monte Carlo tree search over conjunctive predicate trees.

Search space: conjunctions of at most `max_conjuncts` atoms, each atom being a
numeric comparison against a training-quantile constant, a categorical equality,
or a ratio of two numeric features against a constant (depth <= 3). Reward for a
completed predicate: forward-chaining validation AUC-PR of a logistic model over
the firings of (existing bank + candidate). UCB1 selection, one expansion per
simulation, random rollout, mean backup.

This is a compact, honest implementation for the fixture. It is not the paper's
search (K=64 templates, 50,000-row corpus, 100 simulations per expansion).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from templar_fraud_lab.splits.forward_chain import forward_chaining_folds
from templar_fraud_lab.templates.bank import BANK_DISCOVERED, Template, TemplateBank, firings_matrix
from templar_fraud_lab.templates.predicate import evaluate, validate

Atom = dict[str, Any]


@dataclass(frozen=True)
class MctsConfig:
    simulations: int = 24
    n_templates: int = 3
    max_conjuncts: int = 3
    exploration: float = 1.0
    n_folds: int = 3
    quantiles: tuple[float, ...] = (0.5, 0.8, 0.95)
    seed: int = 42


def candidate_atoms(train: pd.DataFrame, numeric: tuple[str, ...], categorical: tuple[str, ...],
                    quantiles: tuple[float, ...]) -> list[Atom]:
    atoms: list[Atom] = []
    for col in numeric:
        vals = pd.to_numeric(train[col], errors="coerce").dropna()
        for q in quantiles:
            thr = float(round(vals.quantile(q), 4))
            atoms.append({"op": ">", "args": [{"feat": col}, {"const": thr}]})
            atoms.append({"op": "<=", "args": [{"feat": col}, {"const": thr}]})
    for col in categorical:
        for level in sorted(train[col].dropna().astype(str).unique())[:8]:
            atoms.append({"op": "==", "args": [{"feat": col}, {"const": level}]})
    if len(numeric) >= 2:
        a, b = numeric[0], numeric[1]
        ratio = pd.to_numeric(train[a], errors="coerce") / pd.to_numeric(train[b], errors="coerce").replace(0, np.nan)
        thr = float(round(ratio.dropna().quantile(0.9), 4))
        atoms.append({"op": ">", "args": [{"op": "/", "args": [{"feat": a}, {"feat": b}]}, {"const": thr}]})
    for atom in atoms:
        validate(atom)
    return atoms


def _predicate(atoms: list[Atom]) -> dict[str, Any]:
    return atoms[0] if len(atoms) == 1 else {"op": "and", "args": list(atoms)}


class _Node:
    def __init__(self, atoms: tuple[int, ...], terminal: bool) -> None:
        self.atoms, self.terminal = atoms, terminal
        self.children: dict[int | None, _Node] = {}
        self.visits, self.value = 0, 0.0

    def ucb(self, parent_visits: int, c: float) -> float:
        if self.visits == 0:
            return math.inf
        return self.value / self.visits + c * math.sqrt(math.log(parent_visits) / self.visits)


class TemplateSearch:
    def __init__(self, train: pd.DataFrame, labels: np.ndarray, bank: TemplateBank,
                 atoms: list[Atom], cfg: MctsConfig) -> None:
        self.train, self.y, self.bank, self.atoms, self.cfg = train, labels, bank, atoms, cfg
        self.rows = train.to_dict(orient="records")
        self.base = firings_matrix(bank, train)
        self.folds = forward_chaining_folds(len(train), cfg.n_folds)
        self.rng = np.random.default_rng(cfg.seed)
        self.evaluations = 0

    def _firing(self, atoms: tuple[int, ...]) -> np.ndarray:
        pred = _predicate([self.atoms[i] for i in atoms])
        return np.array([1.0 if bool(evaluate(pred, r)) else 0.0 for r in self.rows], dtype=np.float32)

    def reward(self, atoms: tuple[int, ...]) -> float:
        if not atoms:
            return 0.0
        self.evaluations += 1
        x = np.hstack([self.base, self._firing(atoms)[:, None]])
        scores = []
        for fold in self.folds:
            ytr = self.y[fold.train_positions]
            if ytr.sum() == 0 or self.y[fold.val_positions].sum() == 0:
                continue
            clf = LogisticRegression(max_iter=200, C=1.0).fit(x[fold.train_positions], ytr)
            scores.append(average_precision_score(self.y[fold.val_positions], clf.predict_proba(x[fold.val_positions])[:, 1]))
        return float(np.mean(scores)) if scores else 0.0

    def _actions(self, atoms: tuple[int, ...]) -> list[int | None]:
        acts: list[int | None] = [None] if atoms else []
        if len(atoms) < self.cfg.max_conjuncts:
            acts.extend(i for i in range(len(self.atoms)) if i not in atoms)
        return acts

    def _rollout(self, atoms: tuple[int, ...]) -> tuple[int, ...]:
        while len(atoms) < self.cfg.max_conjuncts and self.rng.random() < 0.5:
            free = [i for i in range(len(self.atoms)) if i not in atoms]
            atoms = atoms + (int(self.rng.choice(free)),)
        return atoms or (int(self.rng.integers(len(self.atoms))),)

    def search_one(self) -> tuple[tuple[int, ...], float]:
        root = _Node((), terminal=False)
        best, best_r = (), -1.0
        for _ in range(self.cfg.simulations):
            node, path = root, [root]
            while not node.terminal:
                actions = self._actions(node.atoms)
                untried = [a for a in actions if a not in node.children]
                if untried:
                    a = untried[int(self.rng.integers(len(untried)))]
                    child = _Node(node.atoms if a is None else node.atoms + (a,), terminal=a is None)
                    node.children[a] = child
                    node, path = child, path + [child]
                    break
                a = max(actions, key=lambda k: node.children[k].ucb(node.visits, self.cfg.exploration))
                node, path = node.children[a], path + [node.children[a]]
            atoms = node.atoms if node.terminal else self._rollout(node.atoms)
            r = self.reward(atoms)
            if r > best_r:
                best, best_r = atoms, r
            for n in path:
                n.visits += 1
                n.value += r
        return best, best_r


def discover_templates(train: pd.DataFrame, labels: np.ndarray, bank: TemplateBank,
                       numeric: tuple[str, ...], categorical: tuple[str, ...],
                       cfg: MctsConfig) -> tuple[TemplateBank, list[dict[str, Any]]]:
    atoms = candidate_atoms(train, numeric, categorical, cfg.quantiles)
    discovered: list[Template] = []
    log: list[dict[str, Any]] = []
    current = bank
    for k in range(cfg.n_templates):
        search = TemplateSearch(train, labels, current, atoms, cfg)
        atoms_idx, reward = search.search_one()
        pred = _predicate([atoms[i] for i in atoms_idx])
        tpl = Template(f"M{k + 1:02d}", f"MCTS_RULE_{k + 1}", pred, origin=BANK_DISCOVERED,
                       description="discovered by tools MCTS on synthetic fixture; demo only")
        discovered.append(tpl)
        current = current.extended([tpl], BANK_DISCOVERED)
        log.append({"template_id": tpl.template_id, "reward_val_ap": reward,
                    "reward_evaluations": search.evaluations, "predicate": pred})
    return current, log
