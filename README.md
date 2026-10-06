# DevCARE-SE partial replication package

This camera-ready package supports the reported 730-PR, 14-repository feasibility study without publishing repository identity mappings, exact PR identifiers or timestamps, credentials, or unrestricted annotator workbooks.

## Included

- `data/devcare_deidentified_730.csv`: 730 eligible rows with repositories encoded as R01--R14, synthetic PR identifiers, order-preserving synthetic timestamps, the corrective-activity proxy, C1--C4, analyzer completeness fields, cohort, and locked split.
- `code/`: code for the primary expert, Logistic Regression, Random Forest, LightGBM, and hybrid temporal comparison plus figure generation.
- `results/`: primary summaries, uncertainty summaries, cohort/repository robustness, the aggregate expert mean weights, analyzer-error summaries, label-noise sensitivity, and aggregate external-audit metrics.
- `figures/`: the two paper figures.
- `MANIFEST.sha256`: integrity hashes for every released file.

## Citation

Archived release v1.0.0: https://doi.org/10.5281/zenodo.23183516

Please cite both the archived artifact and the accompanying accepted EAI FISAT 2026 paper using `CITATION.cff`. The paper citation will be completed with its proceedings DOI once assigned by the publisher.

## Reproduce the primary comparison

From this directory, install `requirements.txt`, then run:

```text
python code/run_hybrid_experiment.py --dataset data/devcare_deidentified_730.csv --output-dir reproduced/hybrid --provenance "De-identified camera-ready artifact"
```

The dataset preserves the original within-repository order but replaces exact timestamps. The historical target column is published as `outcome_proxy` to avoid implying verified defects or high-confidence ground truth.

## Scope limits

This is a partial replication package. It excludes repository identity keys, raw GitHub/API payloads, database exports, credentials, direct personal identifiers, private survey responses, individual expert allocations, and annotator workbooks. Only the normalized mean expert weights (0.32, 0.24, 0.18, 0.26) are released. Aggregate audit metrics disclose that the external reviewers used unlogged general-purpose AI assistance; the package does not present those judgments as unaided human ground truth.

## License

This repository uses a scoped license policy rather than applying one license to every artifact:

| Material | Paths | License |
|---|---|---|
| Source code and executable configuration | `code/`, `requirements.txt` | MIT License |
| De-identified data, aggregate results, figures, and documentation | `data/`, `results/`, `figures/`, `README.md`, `DATA_DICTIONARY.md`, `CITATION.cff` | Creative Commons Attribution 4.0 International (CC BY 4.0) |

See `LICENSE_MATRIX.md`, `LICENSE-CODE`, and `LICENSE-DATA.md` for the exact scope and attribution requirements. Excluded/private material is not licensed because it is not distributed.

