"""Turning 4,349 free-text reports into supervision, and proving it beat the public tables.

Only 58 of 4,407 studies carry gold labels. Everything else is supervised by what
a reader — rules or an LLM — takes out of a report written in one of a dozen
languages. The best published table scores 0.893 agreement against those 58
studies; a regex baseline scores 0.814. That gap, not the backbone, is where the
remaining leaderboard headroom lives.

The subtle part is that "not mentioned" is a third state, not a negative. A
radiologist who does not mention the ACL is not asserting it is intact, and
mapping silence to zero teaches the model that absence of evidence is evidence
of absence. `calibrate` fits what each state is actually worth, per finding,
against the gold studies.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from .constants import TARGETS


class LabelState(str, Enum):
    """What the report says about one finding."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNMENTIONED = "not_mentioned"


STATES: tuple[LabelState, ...] = (
    LabelState.POSITIVE,
    LabelState.UNMENTIONED,
    LabelState.NEGATIVE,
)


@dataclass(frozen=True)
class Calibration:
    """Per-target map from report state to probability of the finding being present."""

    table: pd.DataFrame  # index: target, columns: state value, values: P(y=1 | state)

    def probability(self, target: str, state: LabelState) -> float:
        return float(self.table.loc[target, state.value])

    def apply(self, states: pd.DataFrame) -> np.ndarray:
        """Soft targets for a (study x target) frame of states."""
        out = np.zeros(states.shape, dtype=float)
        for j, target in enumerate(states.columns):
            for state in STATES:
                hit = states[target].to_numpy() == state.value
                out[hit, j] = self.probability(target, state)
        return out


def calibrate(
    states: pd.DataFrame,
    gold: pd.DataFrame,
    prior_strength: float = 10.0,
    floor: float = 0.02,
    margin: float = 0.01,
) -> Calibration:
    """Estimate P(finding | report state) per target from the gold studies.

    Shrunk hard toward corpus prevalence: with 58 gold studies a cell can hold two
    examples, and an unshrunk 0/2 becomes a confident zero. The ordering
    positive > unmentioned > negative is then enforced, because a calibration that
    inverts it is fitting noise, whatever the gold studies happen to show.

    `states` covers every study; `gold` holds the labelled subset, indexed the same.
    """
    targets = [t for t in TARGETS if t in states.columns]
    rows = {}
    for target in targets:
        prevalence = float(np.nanmean(gold[target].to_numpy(dtype=float)))
        state_col = states.loc[gold.index, target].to_numpy()
        y = gold[target].to_numpy(dtype=float)
        estimates = {}
        for state in STATES:
            hit = state_col == state.value
            n, k = float(hit.sum()), float(np.nansum(y[hit]))
            estimates[state.value] = (k + prior_strength * prevalence) / (n + prior_strength)
        rows[target] = _enforce_order(estimates, floor=floor, margin=margin)
    table = pd.DataFrame(rows).T[[s.value for s in STATES]]
    table.index.name = "target"
    return Calibration(table)


def _enforce_order(estimates: dict[str, float], floor: float, margin: float) -> dict[str, float]:
    """Clamp to positive > unmentioned > negative, keeping values inside [floor, 1-floor]."""
    pos = min(max(estimates[LabelState.POSITIVE.value], floor), 1 - floor)
    unm = min(max(estimates[LabelState.UNMENTIONED.value], floor), 1 - floor)
    neg = min(max(estimates[LabelState.NEGATIVE.value], floor), 1 - floor)
    unm = min(unm, pos - margin) if pos - margin > floor else floor
    neg = min(neg, unm - margin) if unm - margin > floor else floor
    return {
        LabelState.POSITIVE.value: pos,
        LabelState.UNMENTIONED.value: max(unm, floor),
        LabelState.NEGATIVE.value: max(neg, floor),
    }


def confidence_weights(
    states: pd.DataFrame, extractor_confidence: pd.DataFrame | None = None
) -> np.ndarray:
    """Per-cell loss weights.

    An explicit statement either way is worth more than silence, and the
    extractor's own confidence scales it further when available.
    """
    base = np.where(states.to_numpy() == LabelState.UNMENTIONED.value, 0.35, 1.0)
    if extractor_confidence is not None:
        base = base * np.clip(extractor_confidence.to_numpy(dtype=float), 0.0, 1.0)
    return base


def agreement_vs_gold(
    predicted: pd.DataFrame, gold: pd.DataFrame, threshold: float = 0.5
) -> pd.DataFrame:
    """Per-finding and macro agreement against the 58 gold studies.

    This is the number to compare with the public tables (LLM 0.893, regex 0.814).
    `predicted` may hold states or probabilities; states are mapped to 1 for
    positive and 0 otherwise, which is how a binary label table is read.
    """
    rows = []
    common = gold.index.intersection(predicted.index)
    for target in [t for t in TARGETS if t in gold.columns]:
        truth = gold.loc[common, target].to_numpy(dtype=float) > 0.5
        raw = predicted.loc[common, target]
        if pd.api.types.is_numeric_dtype(raw):
            pred = raw.to_numpy(dtype=float) > threshold
        else:
            pred = raw.to_numpy().astype(str) == LabelState.POSITIVE.value
        rows.append(
            {
                "target": target,
                "agreement": float((pred == truth).mean()),
                "n": int(len(common)),
                "n_pos_gold": int(truth.sum()),
                "n_pos_pred": int(pred.sum()),
            }
        )
    table = pd.DataFrame(rows)
    macro = pd.DataFrame(
        [{"target": "MACRO", "agreement": float(table["agreement"].mean()), "n": len(common)}]
    )
    return pd.concat([table.sort_values("agreement"), macro], ignore_index=True)


EXTRACTION_SYSTEM_PROMPT = (
    "You are a musculoskeletal radiologist reading a knee MRI report. "
    "The report may be in any language. For each finding you are asked about, decide whether the "
    "report states it is present, states it is absent, or does not address it at all. "
    "Not addressing a finding is its own answer: do not guess absent when the report is silent. "
    "Quote the span you based the decision on, in the report's original language."
)


def build_extraction_prompt(report: str, targets: tuple[str, ...] = TARGETS) -> str:
    """One prompt covering all twelve findings.

    Per-finding prompts are worth trying for the hard targets (Synovitis, PF OA),
    where a shared prompt tends to blur the distinction between effusion and
    synovial thickening.
    """
    listing = "\n".join(f"- {t}" for t in targets)
    return (
        f"Findings to decide:\n{listing}\n\n"
        f"Report:\n\"\"\"\n{report.strip()}\n\"\"\"\n\n"
        "Answer with one JSON object mapping each finding name to "
        '{"state": "positive"|"negative"|"not_mentioned", "confidence": 0.0-1.0, '
        '"evidence": "quoted span or empty string"}. Output JSON only.'
    )


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_extraction(raw: str, targets: tuple[str, ...] = TARGETS) -> dict[str, dict]:
    """Parse a model response into per-target states, tolerating fences and prose.

    Anything missing or unparseable becomes `not_mentioned` at zero confidence —
    an unreadable answer is not evidence of absence either.
    """
    default = {t: {"state": LabelState.UNMENTIONED.value, "confidence": 0.0, "evidence": ""} for t in targets}
    match = _JSON_BLOCK.search(raw or "")
    if not match:
        return default
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return default
    valid = {s.value for s in STATES}
    for target in targets:
        entry = payload.get(target)
        if not isinstance(entry, dict):
            continue
        state = str(entry.get("state", "")).strip().lower()
        if state not in valid:
            continue
        try:
            confidence = float(entry.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        default[target] = {
            "state": state,
            "confidence": min(max(confidence, 0.0), 1.0),
            "evidence": str(entry.get("evidence", ""))[:500],
        }
    return default


def disagreements(tables: dict[str, pd.DataFrame], targets: tuple[str, ...] = TARGETS) -> pd.DataFrame:
    """Studies where label tables disagree, most-contested first.

    Everyone shares the same three public tables, so their errors are everyone's
    errors. The rows this returns are the only cheap source of supervision nobody
    else is looking at — read fifty of them by hand.
    """
    names = list(tables)
    index = tables[names[0]].index
    rows = []
    for target in [t for t in targets if t in tables[names[0]].columns]:
        stacked = np.stack(
            [(tables[n].loc[index, target].to_numpy(dtype=float) > 0.5).astype(int) for n in names]
        )
        contested = stacked.std(axis=0) > 0
        for study in index[contested]:
            rows.append(
                {
                    "StudyInstanceUID": study,
                    "target": target,
                    **{n: int(tables[n].loc[study, target] > 0.5) for n in names},
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    counts = out.groupby("StudyInstanceUID").size().rename("n_contested")
    return out.join(counts, on="StudyInstanceUID").sort_values(
        ["n_contested", "StudyInstanceUID"], ascending=[False, True]
    )
