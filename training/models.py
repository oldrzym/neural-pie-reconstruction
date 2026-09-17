"""
Model definitions for PIE reconstruction.

Supports pretrained T5/ByT5/mT5 models from HuggingFace.

For specialized architectures like DPD-BiReconstructor,
see: https://github.com/cmu-llab/dpd
"""

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


def load_pretrained_model(model_name: str):
    """
    Load pretrained T5/ByT5/mT5 model and tokenizer.

    Args:
        model_name: HuggingFace model name, e.g.:
            - "google/byt5-base" (recommended for Unicode/IPA)
            - "google/byt5-small"
            - "google/mt5-base"
            - "google/mt5-small"
            - "t5-base"
            - "t5-small"

    Returns:
        (model, tokenizer) tuple
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return model, tokenizer


# Model recommendations for PIE reconstruction:
#
# 1. ByT5 (byte-level) - best for Unicode/IPA characters
#    Works directly with bytes, no tokenization issues with diacritics
#
# 2. mT5 (multilingual) - good alternative
#    Pretrained on 101 languages, handles most scripts
#
# 3. DPD-BiReconstructor - specialized architecture
#    State-of-the-art for proto-language reconstruction
#    Requires separate installation: https://github.com/cmu-llab/dpd
