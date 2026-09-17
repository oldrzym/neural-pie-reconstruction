"""
Utilities for training/evaluating Cognate Transformer on PIE reconstruction data.
"""

from __future__ import annotations

import json
import re
from collections import Counter
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import lingpy

    LINGPY_AVAILABLE = True
except ImportError:
    lingpy = None
    LINGPY_AVAILABLE = False


DELIM = "|"
INPUT_PATTERN = re.compile(r"\[([^\]]+)\]\s*([^\[]+)")
logging.getLogger("lingpy").setLevel(logging.ERROR)


def get_character_tokenizer_compat_class():
    """
    Return a HF-compatible character tokenizer class.

    CognateTransformer ships its own tokenizer implementation that breaks on
    newer `transformers` because `get_vocab()` is not implemented.
    This local class is API-compatible for the needs of this project.
    """
    from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer

    class CharacterTokenizerCompat(PreTrainedTokenizer):
        def __init__(self, characters, model_max_length: int, delim: str = "", **kwargs):
            self.characters = list(characters)
            self.model_max_length = model_max_length
            self.delim = delim

            self._vocab_str_to_int = {
                "[CLS]": 0,
                "[SEP]": 1,
                "[BOS]": 2,
                "[MASK]": 3,
                "[PAD]": 4,
                "[RESERVED]": 5,
                "[UNK]": 6,
                **{ch: i + 7 for i, ch in enumerate(self.characters)},
            }
            self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

            bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
            eos_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
            sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
            cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
            pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
            unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)
            mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)

            super().__init__(
                bos_token=bos_token,
                eos_token=eos_token,
                sep_token=sep_token,
                cls_token=cls_token,
                pad_token=pad_token,
                mask_token=mask_token,
                unk_token=unk_token,
                add_prefix_space=False,
                model_max_length=model_max_length,
                **kwargs,
            )

        @property
        def vocab_size(self) -> int:
            return len(self._vocab_str_to_int)

        def get_vocab(self):
            return dict(self._vocab_str_to_int)

        def _tokenize(self, text: str) -> List[str]:
            if self.delim:
                return str(text).split(self.delim)
            return list(str(text))

        def _convert_token_to_id(self, token: str) -> int:
            return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

        def _convert_id_to_token(self, index: int) -> str:
            return self._vocab_int_to_str[index]

        def convert_tokens_to_string(self, tokens):
            return self.delim.join(tokens)

        def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
            sep = [self.sep_token_id]
            cls = [self.cls_token_id]
            out = cls + token_ids_0 + sep
            if token_ids_1 is not None:
                out += token_ids_1 + sep
            return out

        def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
            if already_has_special_tokens:
                return super().get_special_tokens_mask(
                    token_ids_0=token_ids_0,
                    token_ids_1=token_ids_1,
                    already_has_special_tokens=True,
                )
            out = [1] + ([0] * len(token_ids_0)) + [1]
            if token_ids_1 is not None:
                out += ([0] * len(token_ids_1)) + [1]
            return out

        def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
            sep = [self.sep_token_id]
            cls = [self.cls_token_id]
            out = len(cls + token_ids_0 + sep) * [0]
            if token_ids_1 is not None:
                out += len(token_ids_1 + sep) * [1]
            return out

        def get_config(self) -> Dict:
            return {
                "chars": self.characters,
                "model_max_length": self.model_max_length,
                "delim": self.delim,
            }

        @classmethod
        def from_config(cls, config: Dict):
            return cls(
                characters=config["chars"],
                model_max_length=config["model_max_length"],
                delim=config.get("delim", ""),
            )

        def save_pretrained(self, save_directory, **kwargs):
            save_dir = Path(save_directory)
            save_dir.mkdir(parents=True, exist_ok=True)
            cfg_file = save_dir / "tokenizer_config.json"
            with cfg_file.open("w", encoding="utf-8") as f:
                json.dump(self.get_config(), f, indent=2, ensure_ascii=False)

        @classmethod
        def from_pretrained(cls, save_directory, **kwargs):
            cfg_file = Path(save_directory) / "tokenizer_config.json"
            with cfg_file.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            return cls.from_config(cfg)

    return CharacterTokenizerCompat


def parse_input_forms(input_text: str) -> List[Tuple[str, str]]:
    """Parse '[lang] word [lang] word ...' into a list of (lang, word)."""
    matches = INPUT_PATTERN.findall(str(input_text))
    forms: List[Tuple[str, str]] = []
    for lang, word in matches:
        cleaned = word.strip()
        if cleaned:
            forms.append((lang.strip(), cleaned))
    return forms


def deduplicate_langs(forms: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Keep all cognate forms and disambiguate duplicate language tags.
    Example: lat, lat, grc -> lat#1, lat#2, grc
    """
    counts: Counter[str] = Counter(lang for lang, _ in forms)
    seen: Counter[str] = Counter()
    out: List[Tuple[str, str]] = []
    for lang, word in forms:
        seen[lang] += 1
        if counts[lang] == 1:
            out.append((lang, word))
        else:
            out.append((f"{lang}#{seen[lang]}", word))
    return out


def normalize_protoform(
    proto: str,
    *,
    take_first: bool = True,
    strip_leading_star: bool = True,
    strip_trailing_hyphen: bool = True,
) -> str:
    """Normalize target protoform into a single string sequence."""
    value = str(proto).strip()
    if take_first and "," in value:
        value = value.split(",")[0].strip()
    if strip_leading_star and value.startswith("*"):
        value = value[1:]
    if strip_trailing_hyphen:
        value = value.rstrip("-")
    return value.strip()


def tokenize_word(word: str) -> List[str]:
    """Character-level tokenization."""
    return list(str(word))


def _align_simple(forms: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Fallback alignment: pad tokenized strings to max length."""
    tokenized = [(lang, tokenize_word(word)) for lang, word in forms]
    max_len = max(len(tokens) for _, tokens in tokenized)
    out: Dict[str, List[str]] = {}
    for lang, tokens in tokenized:
        out[lang] = tokens + (["-"] * (max_len - len(tokens)))
    return out


def align_forms(
    forms: List[Tuple[str, str]],
    *,
    use_lingpy: bool = True,
) -> Dict[str, List[str]]:
    """Align daughter forms with lingpy MSA, with a simple fallback."""
    if len(forms) == 0:
        return {}

    if len(forms) == 1 or not use_lingpy or not LINGPY_AVAILABLE:
        return _align_simple(forms)

    langs = [lang for lang, _ in forms]
    seqs = [tokenize_word(word) for _, word in forms]

    try:
        msa = lingpy.Multiple(seqs)
        msa.prog_align()
        aligned = msa.alm_matrix
        out: Dict[str, List[str]] = {}
        for lang, algn in zip(langs, aligned):
            out[lang] = list(algn)
        return out
    except Exception:
        return _align_simple(forms)


def format_sequence(lang: str, tokens: List[str], delim: str = DELIM) -> str:
    return f"[{lang}]{delim}" + delim.join(tokens)


def build_sample(
    input_text: str,
    output_text: str,
    *,
    proto_lang: str = "PIE",
    use_lingpy: bool = True,
    min_unique_langs: int = 1,
    take_first_protoform: bool = True,
) -> Dict | None:
    """
    Build one sample in CogTran-like format:
      {
        "data": {"lat#1":"[lat#1]|c|a|n|i|s", ..., "PIE":"[PIE]|?|?|?..."},
        "solns":{"PIE":"[PIE]|k|w|o|n"},
      }
    """
    parsed = parse_input_forms(input_text)
    if not parsed:
        return None

    unique_langs = len(set(lang for lang, _ in parsed))
    if unique_langs < min_unique_langs:
        return None

    parsed = deduplicate_langs(parsed)
    aligned = align_forms(parsed, use_lingpy=use_lingpy)
    if not aligned:
        return None

    any_seq = next(iter(aligned.values()))
    seq_len = len(any_seq)
    data = {lang: format_sequence(lang, tokens) for lang, tokens in aligned.items()}
    data[proto_lang] = format_sequence(proto_lang, ["?"] * seq_len)

    proto = normalize_protoform(output_text, take_first=take_first_protoform)
    if not proto:
        return None
    proto_tokens = tokenize_word(proto)
    solns = {proto_lang: format_sequence(proto_lang, proto_tokens)}

    return {"data": data, "solns": solns, "target_text": proto, "input_text": str(input_text)}


def load_split(
    csv_path: str | Path,
    *,
    proto_lang: str = "PIE",
    use_lingpy: bool = True,
    min_unique_langs: int = 1,
    take_first_protoform: bool = True,
) -> Tuple[List[Dict], Dict[str, int]]:
    """Load CSV split and convert rows into samples."""
    df = pd.read_csv(csv_path)
    samples: List[Dict] = []
    skipped = 0
    for _, row in df.iterrows():
        sample = build_sample(
            row["input"],
            row["output"],
            proto_lang=proto_lang,
            use_lingpy=use_lingpy,
            min_unique_langs=min_unique_langs,
            take_first_protoform=take_first_protoform,
        )
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)

    stats = {
        "rows_total": int(len(df)),
        "rows_kept": int(len(samples)),
        "rows_skipped": int(skipped),
    }
    return samples, stats


def compute_shape_stats(samples: List[Dict]) -> Dict[str, int]:
    """Compute max/min dimensions for debugging and config sanity."""
    if not samples:
        return {
            "num_samples": 0,
            "max_alignments": 0,
            "max_alignment_length": 0,
        }
    max_rows = 0
    max_len = 0
    for sample in samples:
        rows = len(sample["data"])
        max_rows = max(max_rows, rows)
        for seq in sample["data"].values():
            seq_len = len(str(seq).split(DELIM)) - 1
            max_len = max(max_len, seq_len)
    return {
        "num_samples": int(len(samples)),
        "max_alignments": int(max_rows),
        "max_alignment_length": int(max_len),
    }


def ensure_msa_max_tokens_per_msa(model, min_tokens_per_msa: int) -> int:
    """
    Increase MSA max-tokens threshold on loaded model if needed.

    Some fair-esm versions recurse indefinitely in eval mode when
    `num_rows > max_tokens_per_msa`. We guard against that by ensuring the
    threshold is above observed row counts.
    """
    target = max(int(min_tokens_per_msa), 2)

    cfg = getattr(model, "config", None)
    if cfg is not None:
        cur = getattr(cfg, "max_position_embeddings_per_msa", None)
        if cur is not None and cur < target:
            cfg.max_position_embeddings_per_msa = target

    args = getattr(model, "args", None)
    if args is not None:
        cur = getattr(args, "max_tokens_per_msa", None)
        if cur is not None and cur < target:
            args.max_tokens_per_msa = target
        cur = getattr(args, "max_positions", None)
        if cur is not None and cur < target:
            args.max_positions = target

    msat = getattr(model, "msat", None)
    layers = getattr(msat, "layers", None)
    if layers is None:
        return target

    for layer in layers:
        for block_name in ("row_self_attention", "column_self_attention"):
            block = getattr(layer, block_name, None)
            if block is None:
                continue
            attn = getattr(block, "layer", block)
            cur = getattr(attn, "max_tokens_per_msa", None)
            if cur is not None and cur < target:
                attn.max_tokens_per_msa = target

    return target


def build_vocab(samples: List[Dict], delim: str = DELIM) -> List[str]:
    """Build tokenizer vocab from data+targets tokens in prepared samples."""
    vocab = set()
    for sample in samples:
        for side in ("data", "solns"):
            for seq in sample[side].values():
                for tok in str(seq).split(delim):
                    if tok:
                        vocab.add(tok)
    return sorted(vocab)


def tokenize_row(
    row: Dict,
    tokenizer,
    *,
    delim: str = DELIM,
    return_tensors=None,
) -> Dict:
    """
    Tokenize one prepared row into CogTran input format:
      input_ids: List[List[token_ids_per_alignment]]
      labels: token ids for target proto sequence
    """
    result = {"input_ids": [], "attention_mask": []}
    q_id = tokenizer.convert_tokens_to_ids("?")
    for _, aligned in row["data"].items():
        if aligned is None or len(aligned.split(delim)) <= 1:
            continue
        txt = aligned
        encoded = tokenizer(txt, return_tensors=return_tensors)
        input_ids = encoded["input_ids"]
        if q_id is not None and q_id != tokenizer.unk_token_id:
            if return_tensors == "pt":
                input_ids = input_ids.clone()
                input_ids[input_ids == q_id] = tokenizer.mask_token_id
            else:
                input_ids = [tokenizer.mask_token_id if t == q_id else t for t in input_ids]
        attention_mask = encoded.get("attention_mask")

        if attention_mask is None:
            if return_tensors == "pt":
                import torch

                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            else:
                attention_mask = [1] * len(input_ids)

        result["input_ids"].append(input_ids)
        result["attention_mask"].append(attention_mask)

    if result["input_ids"]:
        if return_tensors == "pt":
            import torch
            import torch.nn.functional as F

            max_len = max(int(x.shape[-1]) for x in result["input_ids"])
            for i in range(len(result["input_ids"])):
                seq_len = int(result["input_ids"][i].shape[-1])
                pad_len = max_len - seq_len
                if pad_len <= 0:
                    continue
                result["input_ids"][i] = F.pad(
                    result["input_ids"][i],
                    (0, pad_len),
                    value=int(tokenizer.pad_token_id),
                )
                result["attention_mask"][i] = F.pad(
                    result["attention_mask"][i],
                    (0, pad_len),
                    value=0,
                )
        else:
            max_len = max(len(x) for x in result["input_ids"])
            for i in range(len(result["input_ids"])):
                seq_len = len(result["input_ids"][i])
                pad_len = max_len - seq_len
                if pad_len <= 0:
                    continue
                result["input_ids"][i] = result["input_ids"][i] + [int(tokenizer.pad_token_id)] * pad_len
                result["attention_mask"][i] = result["attention_mask"][i] + [0] * pad_len

    if return_tensors == "pt":
        import torch

        for key in result:
            result[key] = torch.stack(result[key])

    if "solns" in row:
        # single proto target
        target_text = next(iter(row["solns"].values()))
        result["labels"] = tokenizer(target_text, return_tensors=return_tensors)["input_ids"]

    return result


def clean_decoded_text(decoded: str, *, delim: str = DELIM) -> str:
    """
    Remove special/lang/alignment tokens from decoded output and return plain string.
    """
    tokens = str(decoded).split(delim)
    out: List[str] = []
    for token in tokens:
        if token == "[SEP]":
            break
        if token in {"[CLS]", "[PAD]", "[MASK]", "-", ""}:
            continue
        if token.startswith("[") and token.endswith("]"):
            continue
        out.extend(token.split())
    return "".join(out)


def decode_predictions_and_labels(preds, labels, tokenizer) -> Tuple[List[str], List[str]]:
    if isinstance(preds, tuple):
        preds = preds[0]
    if preds.ndim == 3:
        pred_ids = np.argmax(preds, axis=-1)
    else:
        pred_ids = preds

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)
    pred_texts = [clean_decoded_text(x) for x in decoded_preds]
    label_texts = [clean_decoded_text(x) for x in decoded_labels]
    return pred_texts, label_texts


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            ins = previous_row[j + 1] + 1
            dele = current_row[j] + 1
            sub = previous_row[j] + (c1 != c2)
            current_row.append(min(ins, dele, sub))
        previous_row = current_row
    return previous_row[-1]


def compute_string_metrics(preds: List[str], labels: List[str]) -> Dict[str, float]:
    total = len(labels) if labels else 1
    exact = 0
    total_edit = 0.0
    total_norm_edit = 0.0
    total_chars = 0
    char_correct = 0
    for pred, gold in zip(preds, labels):
        pred = pred.strip()
        gold = gold.strip()
        if pred == gold:
            exact += 1
        dist = levenshtein_distance(pred, gold)
        total_edit += dist
        total_norm_edit += dist / max(len(pred), len(gold), 1)

        min_len = min(len(pred), len(gold))
        char_correct += sum(a == b for a, b in zip(pred[:min_len], gold[:min_len]))
        total_chars += max(len(pred), len(gold), 1)

    return {
        "exact_match_accuracy": exact / total,
        "character_accuracy": char_correct / total_chars if total_chars > 0 else 0.0,
        "mean_edit_distance": total_edit / total,
        "mean_normalized_edit_distance": total_norm_edit / total,
        "total_examples": int(len(labels)),
        "exact_matches": int(exact),
    }


def trainer_compute_metrics(eval_preds, tokenizer) -> Dict[str, float]:
    preds, labels = eval_preds
    pred_texts, label_texts = decode_predictions_and_labels(preds, labels, tokenizer)
    return compute_string_metrics(pred_texts, label_texts)
