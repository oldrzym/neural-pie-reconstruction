# Neural PIE Reconstruction

Code, open data derivatives, and reproducibility materials for:

> Konstantin Kalinowski and Valentin Malykh. 2026. "Towards Neural PIE
> Reconstruction from Heterogeneous Cognate Data." *Computational Linguistics
> and Intellectual Technologies*, 24, 163-170.

- DOI: [10.29003/2075-7182-2026-24-163-170](https://doi.org/10.29003/2075-7182-2026-24-163-170)
- [Official paper PDF](https://dialogue-conf.org/wp-content/uploads/2026/06/KalinowskiKMalykhV.034.pdf)

## Scope

The repository contains the ByT5 training pipeline, preprocessing and analysis
scripts, phonetic-input experiments, and adapters for CognateTransformer, DPD,
and FeVeT. It also contains redistributable dataset variants assembled from
IE-CoR, Kaikki/Wiktionary, and EtymologyDB-derived material.

The public data is not an exact copy of the paper benchmark. Koebler-derived
rows are excluded because an open redistribution license could not be verified.
Starling rows were excluded from the paper's main branch and are not distributed
here. Consequently, the metrics reported in the paper must not be attributed to
the public `open_*` splits without retraining and reevaluation.

## Repository layout

```text
training/          ByT5 training, inference, and evaluation
phonetic_t5/       Epitran, eSpeak-NG, and uroman input preparation
dataset/scripts/   parsing, normalization, splitting, and validation
dataset/processed/ redistributable source-specific derivatives
dataset/splits/    public open-data variants with manifests
cogtran_pie/       CognateTransformer adapter
dpd/               DPD adapter; see the upstream-license warning
fevet_pie/         FeVeT adapter
results/           metrics reported in the published paper
paper/             publication metadata and links
```

## Quick start

Create an environment and install the baseline dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r training/requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Train the example ByT5 baseline on the open expanded split:

```bash
cd training
python train.py --config config.yaml
```

The public data format is:

```text
input:  [lat] canis [grc] kyon [san] sva
output: *kuon-
```

Each row also includes source provenance and a stable release identifier.

## Open data variants

- `open_base`: IE-CoR and Kaikki/Wiktionary.
- `open_expanded`: open base plus EtymologyDB-derived training rows.
- `open_validated_synthetic_expanded`: validated synthetic open base plus
  EtymologyDB; this is a public analogue, not the exact V11 paper artifact.

Every variant has a `manifest.json` containing split sizes, source counts, and
SHA-256 checksums. See [DATA_LICENSES.md](DATA_LICENSES.md) before redistribution.

To rebuild the open variants from an internal paper-data directory:

```bash
python dataset/scripts/build_open_release.py \
  --input-dir /path/to/internal/paper_split \
  --validated-train /path/to/train_wikt_validated_full.csv \
  --output-dir dataset/splits
```

## Synthetic-data scripts

Scripts that call the OpenAI API read credentials from `OPENAI_API_KEY`. Never
place API keys in source files:

```bash
export OPENAI_API_KEY="..."
export PIE_DATA_DIR="/path/to/pipeline/data"
```

## Specialized architectures

CognateTransformer and FeVeT are external projects and retain their upstream
licenses. The DPD repository currently has no declared software license; this
repository distributes only the independently written adapter and does not
vendor DPD source code. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Results

These are the results reported on the original paper benchmark. They must not
be attributed to the public `open_*` splits without retraining and evaluation.

| Configuration | Variant | EM | CharAcc | MED | MNED |
|---|---:|---:|---:|---:|---:|
| ByT5-Base, orthographic baseline | V1 | 0.136 | 0.449 | 3.02 | 0.365 |
| **ByT5-Base, conservative Epitran-only input** | **V10** | **0.216** | **0.490** | **2.84** | **0.344** |
| ByT5-Base, expanded orthographic data | V11 | 0.183 | 0.452 | 3.10 | 0.371 |
| CognateTransformer, LingPy alignment on | Paper test | 0.019 | 0.303 | 3.65 | 0.589 |
| CognateTransformer, LingPy alignment off | Paper test | 0.055 | 0.317 | 3.52 | 0.564 |

Selected data and model ablations:

| Configuration | Variant | EM | CharAcc | MED | MNED |
|---|---:|---:|---:|---:|---:|
| No language tags | V6 | 0.161 | 0.448 | 3.09 | 0.378 |
| No label smoothing | V1 | 0.144 | 0.446 | 3.02 | 0.364 |
| ByT5-Large | V1 | 0.000 | 0.158 | 7.09 | 0.767 |
| +10 synthetic cognates | V2 | 0.116 | 0.413 | 3.35 | 0.400 |
| +20 synthetic cognates | V3 | 0.144 | 0.440 | 3.24 | 0.381 |
| +20 + Wiktionary-guided validation | V5 | 0.108 | 0.400 | 3.43 | 0.410 |
| +20 + validation + EtymologyDB | V11 | 0.183 | 0.452 | 3.10 | 0.371 |

DPD and FeVeT use native metrics that are not directly comparable to the
sequence-level metrics above:

| Model | Setting | Native result |
|---|---|---:|
| DPD | Supervised, V1 | Accuracy 0.017 |
| DPD | Semi-supervised, V1 | Accuracy 0.069 |
| FeVeT | V7 | Phoneme edit distance 3.56; B-cubed F1 0.307 |

`results/paper_results.csv` contains the complete main and ablation tables in a
machine-readable form. Model checkpoints and prediction files are intentionally
not stored in Git.

## License and citation

Original code in this repository is licensed under Apache-2.0. Dataset rows are
covered by source-specific licenses described in `DATA_LICENSES.md`; the code
license does not apply to them.

Use `CITATION.cff` or the paper citation above when using this repository.
