# Border Core Clustering

Reproducible implementation and experimental material for **Border Core**, a clustering framework combining density-guided initialization, protected interior assignments, selective frontier propagation, and contact-based cluster consolidation.

This repository provides an independently reconstructed implementation, **not the unavailable original source code**. The included synthetic experiments use researcher-generated data and must not be described as a third-party independent benchmark. Some experimental controls were developed after preliminary results were examined; their comparisons are exploratory.

## Structure

- `src/`: reconstructed Border Core implementation, factorial ablations, geometry and model-selection helpers.
- `scripts/merge_controls.py`: non-supervised cluster-merging controls and repeatable synthetic benchmark execution.
- `scripts/smoke_test.py`: checks that one regenerated case matches the archived output exactly.
- `results/`: full archived CSV/JSON outputs, including synthetic experiments, timing ablations, real-data exploratory HDBSCAN results, and merger comparisons.
- `requirements.txt`: recorded Python dependencies.
- `SHA256SUMS.txt`: cryptographic fingerprints of the deposited files (except this manifest itself).

## Installation

Python 3.11 or newer is recommended. Create a virtual environment and install the recorded versions:

```bash
python -m venv .venv
# Activate your virtual environment, then:
python -m pip install -r requirements.txt
```

## Reproduction

From the repository root, run a quick archived-result check:

```bash
python scripts/smoke_test.py
```

Recompute the 16 configurations across ten independently seeded synthetic replicates:

```bash
python scripts/merge_controls.py --repeats 10 --out results/recomputed_controls.csv
```

The core experimental output contains 16 synthetic configurations × 10 seeds × 3 base pipelines × 4 merger policies = **1,920 rows**. The automatic selection policies do not use external labels. Reported external metrics use labels only for evaluation, and oracle experiments are diagnostic rather than deployable unsupervised methods.

The computations are research prototypes. Reproducing measured wall-clock times requires controlling hardware, software versions, threads and JIT warm-up; the provided timings are descriptive, not a cross-platform speed claim.

## Scientific scope and limitations

- Four synthetic generator families do **not** establish external benchmark generalization.
- The contact-merger improvement is concentrated in specific geometries; it is not a general performance guarantee.
- The code base is a disclosed reconstruction; matching output on test cases does not establish identity with lost source code.
- Any journal manuscript and author disclosures should be confirmed by all contributors before submission.

## Citation and archive

The machine-readable `CITATION.cff` lists the manuscript's four authors. Cite the eventual permanent Zenodo DOI for the specific archived release once it has been assigned. GitHub repository creation alone does not assign a DOI.

## Licensing

No open-source license is granted by the presence of these files in a public repository. The authors should select and approve a license before third-party reuse. The availability of third-party benchmark data may be governed by separate terms.
