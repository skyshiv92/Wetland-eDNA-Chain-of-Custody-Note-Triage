# Wetland eDNA Chain-of-Custody Note Triage Dataset

## Overview

This dataset supports an NLP benchmark for environmental DNA chain-of-custody triage. Each row represents an eDNA sample accession case where field notes, custody records, preservation observations, control behavior, and qPCR replicate descriptions must be translated into a ranked set of quality actions.

The records model realistic wetland monitoring workflows: field crews collect water, sediment, biofilm, or filter samples; laboratories accession the samples under different receipt windows; and QA staff decide whether to report, rerun, recollect, quarantine, or escalate the case. The included generator creates reproducible challenge rows from curated environmental sampling QA patterns so the raw dataset can be rebuilt and audited.

## Raw Upload File Structure

The raw dataset upload contains exactly these top-level files:

| File | Description |
|---|---|
| `data.csv` | Source rows used by the prepare script. Contains notes, public context columns, target action manifests, split hints, and private diagnostic grouping columns. |
| `generate_raw.py` | Deterministic Python script that creates `data.csv` from curated eDNA accession and QA patterns. |

## Raw Columns

| Column | Type | Description |
|---|---|---|
| `raw_case_id` | string | Internal source row identifier before preparation. This is replaced by an opaque public `case_id`. |
| `custody_note` | string | Free-text field/lab note describing accession, custody, blank/control, preservation, inhibition, replicate, or panel issues. |
| `project_region` | categorical | Monitoring region: `coastal_marsh`, `urban_stream`, `peat_wetland`, `alpine_lake`, or `farm_ditch`. |
| `assay_panel` | categorical | eDNA panel: `amphibian_pathogen`, `invasive_carp`, `freshwater_mussel`, `cyanobacteria_bloom`, or `salmonid_spawning`. |
| `sample_matrix` | categorical | Sample material: `surface_water`, `sediment_slurry`, `biofilm_swab`, or `filter_membrane`. |
| `collection_context` | categorical | Field timing context: `baseflow`, `storm_pulse`, `post_rain`, `thaw_window`, or `algal_bloom_watch`. |
| `lab_window` | categorical | Accession timing: `same_day`, `overnight_hold`, `weekend_receipt`, `field_freezer_delay`, or `split_batch`. |
| `collection_month` | integer | Collection month from 1 to 12. |
| `field_team` | categorical | Deidentified field crew code. |
| `primary_issue` | categorical | Main hidden QA issue family used for private robustness scoring. |
| `secondary_issue` | categorical | Secondary issue family for mixed cases, or `none`. |
| `difficulty_tier` | categorical | Hidden difficulty label: `easy`, `medium`, or `hard`. |
| `interference_axis` | categorical | Hidden note style such as plain wording, distractor text, negated distractors, or cross-context wording. |
| `split_hint` | categorical | Source split indicator used by `prepare.py` to create public train and test rows. |
| `action_manifest` | string | Target ranked action sequence, three distinct action codes separated by `|`. |
| `score_trace_json` | string | Audit trail of nonzero action scoring components used only to inspect the raw generation process. It is not included in public prepared data. |

## Target Actions

The target `action_manifest` uses three distinct action codes selected from this catalog:

| Action Code | Description |
|---|---|
| `ACCEPT_REPORT` | Release the result when no material QA concern dominates. |
| `RECOLLECT_SAMPLE` | Request a fresh field sample. |
| `CLARIFY_CUSTODY` | Resolve a missing, skipped, or conflicting custody transfer. |
| `QUARANTINE_COLD_CHAIN` | Hold a sample because temperature control is questionable. |
| `RERUN_PCR` | Repeat qPCR replicates from the available sample or extract. |
| `DILUTE_EXTRACT` | Dilute extract to reduce matrix suppression. |
| `CLEANUP_INHIBITORS` | Apply inhibitor cleanup before rerunning amplification. |
| `REVIEW_BLANKS` | Review blank/control behavior before interpreting the sample. |
| `REJECT_CONTAMINATION` | Treat the case as contaminated when blank or control evidence undermines it. |
| `REPEAT_TAXON_PANEL` | Rerun or redirect the sample on the correct assay panel. |
| `HOLD_REPORT` | Hold release pending QA review. |
| `ESCALATE_SUPERVISOR` | Send the case to senior review because multiple QA signals conflict. |

## Data Characteristics

- The free-text `custody_note` is the primary input signal.
- Public context columns are useful but incomplete; they do not directly reveal the correct action order.
- Test rows emphasize harder paraphrases and mixed issue cases.
- Hidden groups support worst-group scoring over issue family, difficulty, note style, region, panel, and sample matrix.
- Prepared public IDs are hashed from `raw_case_id` so source row order and raw identifiers are not exposed.

## Notes

The dataset is designed for CPU-only NLP modeling from the provided public training rows. Solvers should train a text model or sequence model from the public notes and context, then output ranked action manifests for the held-out cases. External datasets, internet lookup, private labels, and pretrained models are not required.
