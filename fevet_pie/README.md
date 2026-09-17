# FeVeT PIE

This folder contains a practical FeVeT-style pipeline for your PIE reconstruction split:

1) convert `input/output` CSV data to FeVeT-compatible structured data,
2) train a FeVeT model on that data,
3) evaluate and export predictions.

## Prerequisites

You need a local clone of the FeVeT repository with its `data/clts` folder:

- Repo: `https://github.com/TGH-2020/FeVeT`
- Default path expected by scripts: `tmp/FeVeT_repo`

Install dependencies in your environment:

```bash
pip install -r tmp/FeVeT_repo/requirements.txt
pip install pandas
```

## 1) Prepare data

Default source split:

- `dataset/splits/iecor_kaikki_koebler_normalized/train.csv`
- `dataset/splits/iecor_kaikki_koebler_normalized/val.csv`
- `dataset/splits/iecor_kaikki_koebler_normalized/test.csv`

Run:

```bash
python3 fevet_pie/prepare_data.py \
  --output-json fevet_pie/data/pie_iecor_prepared.json \
  --proto-lang protopie
```

If you want to use your phonetic split variant:

```bash
python3 fevet_pie/prepare_data.py \
  --train-csv dataset/splits/iecor_kaikki_koebler_normalized_phonetic_input_hybrid_safe/train.csv \
  --val-csv dataset/splits/iecor_kaikki_koebler_normalized_phonetic_input_hybrid_safe/val.csv \
  --test-csv dataset/splits/iecor_kaikki_koebler_normalized_phonetic_input_hybrid_safe/test.csv \
  --output-json fevet_pie/data/pie_iecor_phon_hybrid_safe_prepared.json \
  --proto-lang protopie
```

## 2) Train

```bash
python3 fevet_pie/train.py \
  --prepared-data fevet_pie/data/pie_iecor_prepared.json \
  --fevet-repo tmp/FeVeT_repo \
  --epochs 40 \
  --batch-size 32 \
  --eval-batch-size 32 \
  --learning-rate 5e-4 \
  --hidden-dim 128 \
  --dropout 0.15
```

Quick validation before a long run:

```bash
python3 fevet_pie/train.py \
  --prepared-data fevet_pie/data/pie_iecor_prepared.json \
  --fevet-repo tmp/FeVeT_repo \
  --dry-run
```

Outputs are written to:

- `fevet_pie/runs/run_YYYYmmdd_HHMMSS/`

Main artifacts:

- `model.pt`
- `vocab.pkl`
- `val_metrics.json`
- `test_metrics.json`
- `val_predictions.csv`
- `test_predictions.csv`
- `train_config.json`

## 3) Evaluate

```bash
python3 fevet_pie/evaluate.py \
  --checkpoint-dir fevet_pie/runs/<run_name> \
  --prepared-data fevet_pie/data/pie_iecor_prepared.json \
  --fevet-repo tmp/FeVeT_repo \
  --split test \
  --batch-size 32
```

Dry-run check:

```bash
python3 fevet_pie/evaluate.py \
  --checkpoint-dir fevet_pie/runs/<run_name> \
  --prepared-data fevet_pie/data/pie_iecor_prepared.json \
  --fevet-repo tmp/FeVeT_repo \
  --dry-run
```

Outputs:

- `<checkpoint-dir>/eval_test/metrics.json`
- `<checkpoint-dir>/eval_test/predictions.csv`

## Notes

- This is a direct adaptation for your PIE setup using FeVeT model components.
- Target language is fixed to `protopie` from your `output` column.
- Input forms are tokenized into grapheme-level symbols for FeVeT feature mapping.
