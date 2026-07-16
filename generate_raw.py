from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


ACTIONS = [
    "ACCEPT_REPORT",
    "RECOLLECT_SAMPLE",
    "CLARIFY_CUSTODY",
    "QUARANTINE_COLD_CHAIN",
    "RERUN_PCR",
    "DILUTE_EXTRACT",
    "CLEANUP_INHIBITORS",
    "REVIEW_BLANKS",
    "REJECT_CONTAMINATION",
    "REPEAT_TAXON_PANEL",
    "HOLD_REPORT",
    "ESCALATE_SUPERVISOR",
]

ISSUES = [
    "temp_excursion",
    "custody_gap",
    "blank_contamination",
    "inhibition",
    "replicate_discordance",
    "panel_mismatch",
    "preservation_failure",
    "weather_dilution",
]

REGIONS = ["coastal_marsh", "urban_stream", "peat_wetland", "alpine_lake", "farm_ditch"]
PANELS = ["amphibian_pathogen", "invasive_carp", "freshwater_mussel", "cyanobacteria_bloom", "salmonid_spawning"]
MATRICES = ["surface_water", "sediment_slurry", "biofilm_swab", "filter_membrane"]
CONTEXTS = ["baseflow", "storm_pulse", "post_rain", "thaw_window", "algal_bloom_watch"]
LAB_WINDOWS = ["same_day", "overnight_hold", "weekend_receipt", "field_freezer_delay", "split_batch"]
FIELD_TEAMS = ["delta", "fir", "heron", "lichen", "moraine", "sedge", "willow"]

ISSUE_WEIGHTS = {
    "temp_excursion": {"QUARANTINE_COLD_CHAIN": 3.2, "ESCALATE_SUPERVISOR": 1.1, "RECOLLECT_SAMPLE": 0.9, "HOLD_REPORT": 0.45},
    "custody_gap": {"CLARIFY_CUSTODY": 3.1, "ESCALATE_SUPERVISOR": 0.95, "RECOLLECT_SAMPLE": 0.75, "HOLD_REPORT": 0.45},
    "blank_contamination": {"REVIEW_BLANKS": 3.2, "REJECT_CONTAMINATION": 2.55, "ESCALATE_SUPERVISOR": 0.95, "HOLD_REPORT": 0.35},
    "inhibition": {"CLEANUP_INHIBITORS": 3.1, "DILUTE_EXTRACT": 2.45, "RERUN_PCR": 1.55},
    "replicate_discordance": {"RERUN_PCR": 3.0, "REPEAT_TAXON_PANEL": 1.25, "ESCALATE_SUPERVISOR": 0.85, "HOLD_REPORT": 0.25},
    "panel_mismatch": {"REPEAT_TAXON_PANEL": 3.0, "CLARIFY_CUSTODY": 1.85, "ESCALATE_SUPERVISOR": 0.95, "HOLD_REPORT": 0.25},
    "preservation_failure": {"RECOLLECT_SAMPLE": 3.1, "CLARIFY_CUSTODY": 1.05, "QUARANTINE_COLD_CHAIN": 0.9, "HOLD_REPORT": 0.45},
    "weather_dilution": {"RERUN_PCR": 2.25, "RECOLLECT_SAMPLE": 2.0, "ACCEPT_REPORT": 0.8, "DILUTE_EXTRACT": 0.45},
}

TRAIN_PHRASES = {
    "temp_excursion": [
        "cooler logger shows the sample warmed above the custody limit during transport",
        "ice pack was spent and the receipt temperature was outside the eDNA hold range",
        "temperature strip was red on arrival after a long courier leg",
    ],
    "custody_gap": [
        "chain-of-custody initials are missing between field transfer and lab receipt",
        "the bottle label and custody sheet disagree about the handoff time",
        "the split sample was moved to a second cooler without a signed transfer line",
    ],
    "blank_contamination": [
        "field blank amplified with the same target as the sample",
        "extraction blank has a late but repeatable target signal",
        "negative control was not clean for this plate",
    ],
    "inhibition": [
        "internal amplification control is delayed and the extract looks humic",
        "dilution curve indicates PCR inhibition in the sample extract",
        "matrix spike recovered poorly and the qPCR curve is suppressed",
    ],
    "replicate_discordance": [
        "only one of three qPCR replicates crossed threshold",
        "replicate wells disagree even though the positive control worked",
        "amplification is present in alternating replicates with no stable consensus",
    ],
    "panel_mismatch": [
        "the requested assay panel does not match the species named on the custody form",
        "sample was queued for the wrong taxon panel",
        "primer panel selected at accession conflicts with the field objective",
    ],
    "preservation_failure": [
        "filter envelope leaked preservative before accession",
        "ethanol line is below the membrane and the hold time is exceeded",
        "sample bottle arrived with a loose cap and visible preservative loss",
    ],
    "weather_dilution": [
        "collection occurred during a fast storm pulse with turbid runoff",
        "field team sampled immediately after heavy rain diluted the reach",
        "snowmelt surge changed conductivity and suspended sediment during collection",
    ],
}

HOLDOUT_PHRASES = {
    "temp_excursion": [
        "the logger trace flattened near room temperature before accession rather than staying in the blue band",
        "custody photos show mostly water in the ice sleeve, and the vial was not cold to the touch",
        "the transfer note says the tote sat by the loading door while the chill indicator faded",
    ],
    "custody_gap": [
        "there is a custody jump from the sampler's notebook to the night bench with no named carrier",
        "the barcode trail skips the river station handoff and resumes only after accession",
        "two labels point to the same bottle, but only one appears on the signed route sheet",
    ],
    "blank_contamination": [
        "the water-only traveler produced a small matching curve on the second read",
        "the no-sample lane is not silent and tracks the same amplicon family",
        "the control intended to stay blank shows a faint taxon trace after extraction",
    ],
    "inhibition": [
        "the spike control limps several cycles late in the brown extract",
        "the curve improves after dilution, suggesting the raw aliquot is suppressing polymerase",
        "humic color and a late internal standard point to matrix drag rather than absence",
    ],
    "replicate_discordance": [
        "one well rises cleanly while its companion wells stay at baseline",
        "the triplicate set splits into a late shoulder, a clean negative, and a weak positive",
        "replicate geometry is inconsistent despite an acceptable plate control",
    ],
    "panel_mismatch": [
        "the bench sheet names a bloom panel while the field request asks for a spawning marker",
        "accession routed the tube to a library that cannot answer the species note",
        "the custody comment and loaded primer set describe different surveillance goals",
    ],
    "preservation_failure": [
        "the membrane pouch is damp on the outside and the preservative volume is visibly low",
        "the cap stain suggests the bottle vented before it reached the accession bench",
        "the sample sat past the hold window after the preservative sachet separated from the filter",
    ],
    "weather_dilution": [
        "the reach was sampled while rainwater was still sheeting off the bank",
        "the crew note describes a fresh plume that likely pushed target copies below the usual range",
        "a thaw pulse made the bottle visibly silty and diluted relative to the baseline visit",
    ],
}


def _rng_choice(rng: np.random.Generator, values):
    return values[int(rng.integers(0, len(values)))]


def _stable_noise(*parts: object) -> float:
    raw = "::".join(str(part) for part in parts).encode()
    return (int(hashlib.sha256(raw).hexdigest()[:8], 16) % 1000) / 1000.0


def _context_weights(row: dict) -> dict[str, float]:
    weights = {action: 0.0 for action in ACTIONS}
    weights["ACCEPT_REPORT"] += 0.15
    if row["sample_matrix"] in {"sediment_slurry", "filter_membrane"}:
        weights["CLEANUP_INHIBITORS"] += 0.35
        weights["DILUTE_EXTRACT"] += 0.25
    if row["project_region"] in {"coastal_marsh", "peat_wetland"}:
        weights["DILUTE_EXTRACT"] += 0.2
        weights["REVIEW_BLANKS"] += 0.08
    if row["assay_panel"] in {"amphibian_pathogen", "salmonid_spawning"}:
        weights["RECOLLECT_SAMPLE"] += 0.25
        weights["RERUN_PCR"] += 0.08
    if row["collection_context"] in {"storm_pulse", "thaw_window"}:
        weights["RERUN_PCR"] += 0.25
        weights["RECOLLECT_SAMPLE"] += 0.2
    if row["lab_window"] in {"weekend_receipt", "field_freezer_delay"}:
        weights["QUARANTINE_COLD_CHAIN"] += 0.25
        weights["CLARIFY_CUSTODY"] += 0.15
    return weights


def _rank_actions(row: dict, primary: str, secondary: str | None, difficulty: str) -> tuple[str, dict[str, float]]:
    scores = {action: 0.0 for action in ACTIONS}
    hold_sensitive = {"temp_excursion", "custody_gap", "blank_contamination", "preservation_failure"}
    for action, value in _context_weights(row).items():
        scores[action] += value
    for action, value in ISSUE_WEIGHTS[primary].items():
        scores[action] += value
    if secondary:
        for action, value in ISSUE_WEIGHTS[secondary].items():
            scores[action] += 0.68 * value
    if primary in hold_sensitive:
        scores["HOLD_REPORT"] += 0.85
    if secondary in hold_sensitive:
        scores["HOLD_REPORT"] += 0.45
    if row["lab_window"] in {"weekend_receipt", "field_freezer_delay"}:
        scores["HOLD_REPORT"] += 0.25
    if difficulty == "hard":
        scores["ESCALATE_SUPERVISOR"] += 0.35
        if primary in hold_sensitive:
            scores["HOLD_REPORT"] += 0.25
    if difficulty == "easy" and primary not in {"blank_contamination", "preservation_failure"}:
        scores["ACCEPT_REPORT"] += 0.12

    ranked = sorted(
        ACTIONS,
        key=lambda action: (scores[action], _stable_noise(row["raw_case_id"], action)),
        reverse=True,
    )
    return "|".join(ranked[:3]), scores


def _compose_note(rng: np.random.Generator, row: dict, primary: str, secondary: str | None, split: str) -> str:
    phrases = HOLDOUT_PHRASES if split == "test" else TRAIN_PHRASES
    primary_phrase = _rng_choice(rng, phrases[primary])
    secondary_phrase = _rng_choice(rng, phrases[secondary]) if secondary else ""
    region_text = row["project_region"].replace("_", " ")
    panel_text = row["assay_panel"].replace("_", " ")
    matrix_text = row["sample_matrix"].replace("_", " ")
    context_text = row["collection_context"].replace("_", " ")
    lab_text = row["lab_window"].replace("_", " ")
    team = row["field_team"]

    openers = [
        f"Field team {team} logged a {matrix_text} eDNA sample from the {region_text} program for the {panel_text} panel.",
        f"The {region_text} chain packet describes a {matrix_text} bottle collected under {context_text} conditions for {panel_text}.",
        f"Accession note for team {team}: {panel_text} surveillance sample, {matrix_text}, received during the {lab_text} window.",
    ]
    closers = [
        "The supervisor wants the next three QA actions before results are released.",
        "Choose the ordered action manifest that best protects reporting quality.",
        "The lab needs a ranked remediation plan, not just a pass/fail disposition.",
    ]
    noise = [
        "A spare filter was photographed but not processed.",
        "The GPS point is readable and the jar count matches the cooler manifest.",
        "The plate map was reprinted after a label smudge, but well positions are legible.",
        "No external source lookup is available to adjudicate the record.",
        "The site visit was otherwise routine for this project.",
    ]
    parts = [_rng_choice(rng, openers), primary_phrase]
    if secondary:
        parts.append(secondary_phrase)
    if row["interference_axis"] == "negation":
        parts.append(f"The note also says no action is needed for a separate {rng.choice(ISSUES).replace('_', ' ')} concern.")
    elif row["interference_axis"] == "distractor":
        parts.append(_rng_choice(rng, noise))
    elif row["interference_axis"] == "cross_context":
        parts.append(f"Because the sample is tied to {context_text} and {lab_text}, the lab treats small quality signals conservatively.")
    parts.append(_rng_choice(rng, closers))
    return " ".join(parts)


def make_dataset(rows: int = 5200, seed: int = 2026071407) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    hard_issues = {"blank_contamination", "inhibition", "replicate_discordance", "panel_mismatch"}
    for idx in range(rows):
        split = "test" if idx >= int(rows * 0.72) else "train"
        if split == "test":
            difficulty = rng.choice(["medium", "hard"], p=[0.25, 0.75])
            primary = rng.choice(list(hard_issues | {"custody_gap", "preservation_failure"}))
        else:
            difficulty = rng.choice(["easy", "medium", "hard"], p=[0.34, 0.43, 0.23])
            primary = rng.choice(ISSUES)
        secondary = None
        if difficulty == "medium":
            secondary = rng.choice([issue for issue in ISSUES if issue != primary]) if rng.random() < 0.55 else None
        if difficulty == "hard":
            secondary = rng.choice([issue for issue in ISSUES if issue != primary])

        row = {
            "raw_case_id": f"edna_raw_{idx:05d}",
            "project_region": _rng_choice(rng, REGIONS),
            "assay_panel": _rng_choice(rng, PANELS),
            "sample_matrix": _rng_choice(rng, MATRICES),
            "collection_context": _rng_choice(rng, CONTEXTS),
            "lab_window": _rng_choice(rng, LAB_WINDOWS),
            "collection_month": int(rng.integers(1, 13)),
            "field_team": _rng_choice(rng, FIELD_TEAMS),
            "primary_issue": primary,
            "secondary_issue": secondary or "none",
            "difficulty_tier": difficulty,
            "interference_axis": rng.choice(["plain", "distractor", "negation", "cross_context"], p=[0.35, 0.25, 0.18, 0.22]),
            "split_hint": split,
        }
        row["custody_note"] = _compose_note(rng, row, primary, secondary, split)
        row["action_manifest"], action_scores = _rank_actions(row, primary, secondary, difficulty)
        row["score_trace_json"] = json.dumps({k: round(v, 3) for k, v in action_scores.items() if v > 0.0}, sort_keys=True)
        records.append(row)
    return pd.DataFrame(records)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    data = make_dataset()
    data.to_csv(out_dir / "data.csv", index=False)
    print(f"wrote {len(data)} rows to {out_dir / 'data.csv'}")


if __name__ == "__main__":
    main()
