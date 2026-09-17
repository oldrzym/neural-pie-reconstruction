# Phonetic T5 Prep

This folder contains a helper script for creating a phoneticized copy of an existing PIE split.

## What it does

- Parses each `input` field as repeated chunks: `[lang] form`.
- Tries to transliterate each `form` to IPA using `epitran` for supported languages.
- Optional fallback: use `eSpeak` via `phonemizer` when `epitran` has no model.
- Optional fallback: use `uroman` when both `epitran` and `eSpeak` are unavailable.
- Leaves unsupported language tags unchanged.
- Writes new `train.csv`, `val.csv`, `test.csv` to a separate output folder.
- Adds `input_raw` column (backup of original input) by default.
- Exports:
  - `phoneticization_report.json`
  - `phoneticization_tag_stats.csv`
  - `phoneticization_run.log` (stage-by-stage logs)

## Usage

```bash
python3 prepare_phonetic_split.py \
  --source-dir /root/PIE-reconstruction/dataset/splits/iecor_kaikki_koebler_normalized \
  --output-dir /root/PIE-reconstruction/dataset/splits/iecor_kaikki_koebler_normalized_phonetic_input
```

With eSpeak fallback:

```bash
python3 prepare_phonetic_split.py \
  --source-dir /root/PIE-reconstruction/dataset/splits/iecor_kaikki_koebler_normalized \
  --output-dir /root/PIE-reconstruction/dataset/splits/iecor_kaikki_koebler_normalized_phonetic_input_hybrid \
  --use-espeak-fallback
```

With eSpeak + uroman fallback (best coverage for mixed scripts):

```bash
python3 prepare_phonetic_split.py \
  --source-dir /root/PIE-reconstruction/dataset/splits/iecor_kaikki_koebler_normalized \
  --output-dir /root/PIE-reconstruction/dataset/splits/iecor_kaikki_koebler_normalized_phonetic_input_hybrid_uroman \
  --use-espeak-fallback \
  --use-uroman-fallback
```

Safer eSpeak fallback (allowlist):

```bash
python3 prepare_phonetic_split.py \
  --source-dir /root/PIE-reconstruction/dataset/splits/iecor_kaikki_koebler_normalized \
  --output-dir /root/PIE-reconstruction/dataset/splits/iecor_kaikki_koebler_normalized_phonetic_input_hybrid_safe \
  --use-espeak-fallback \
  --espeak-allow-tags en,fr,de,es,pt,it,ca,nl,sv,da,nb,nn,is,ga,ro,pl,cs,ru,fa,hi,id,fi,et,lt,lv,tr,bg,sr,hr,sl,mk,sq,eu
```

## Requirements

```bash
apt-get install -y espeak-ng
python3 -m pip install epitran pycountry pandas phonemizer uroman
```

On Windows, run with UTF-8 mode for Epitran:

```bash
set PYTHONUTF8=1
python prepare_phonetic_split.py ...
```

## Train/eval launcher

Use `train_variant.py` to run `training/train.py` and `training/evaluate.py` on one of the prepared split variants.

Basic run (hybrid-safe + T5 v1.1 base):

```bash
python3 phonetic_t5/train_variant.py \
  --variant hybrid_safe \
  --model google/t5-v1_1-base \
  --run-name t5v11_base_hybrid_safe
```

Only show resolved paths/commands:

```bash
python3 phonetic_t5/train_variant.py \
  --variant hybrid_safe \
  --model google/t5-v1_1-base \
  --dry-run
```

Variants:
- `raw` -> `dataset/splits/iecor_kaikki_koebler_normalized`
- `phonetic` -> `dataset/splits/iecor_kaikki_koebler_normalized_phonetic_input`
- `hybrid_safe` -> `dataset/splits/iecor_kaikki_koebler_normalized_phonetic_input_hybrid_safe`
- `hybrid_full` -> `dataset/splits/iecor_kaikki_koebler_normalized_phonetic_input_hybrid_full`
- `hybrid_uroman` -> `dataset/splits/iecor_kaikki_koebler_normalized_phonetic_input_hybrid_uroman`

Common overrides:

```bash
python3 phonetic_t5/train_variant.py \
  --variant hybrid_safe \
  --model google/byt5-base \
  --run-name byt5_hybrid_safe \
  --num-epochs 20 \
  --batch-size 16 \
  --eval-batch-size 8 \
  --learning-rate 0.0003 \
  --warmup-steps 400 \
  --label-smoothing 0.0 \
  --early-stopping-patience 6 \
  --num-beams 4
```
