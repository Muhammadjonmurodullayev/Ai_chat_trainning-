"""
Train SentencePiece Unigram tokenizer on uz+en+code corpus.

Outputs:
  checkpoints/chat/chat_vocab.model   (binary SentencePiece model)
  checkpoints/chat/chat_vocab.vocab   (text vocabulary)

Usage:
  python chat/tokenizer_train.py --vocab_size 16000
"""

import os
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints" / "chat"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = DATA_DIR / "chat_corpus.txt"
MODEL_PREFIX = CKPT_DIR / "chat_vocab"

# Special tokens reserved at fixed IDs (must match app/chat/chat_template.py SPECIAL_TOKENS)
SPECIAL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab_size", type=int, default=16000)
    p.add_argument("--character_coverage", type=float, default=0.9999)
    p.add_argument("--model_type", choices=["unigram", "bpe"], default="unigram")
    p.add_argument("--extra_corpus", type=str, default="",
                   help="Comma-separated extra .txt files to mix in")
    args = p.parse_args()

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Run dataset_gen.py first; missing {CORPUS_PATH}")

    import sentencepiece as spm

    inputs = [str(CORPUS_PATH)]
    if args.extra_corpus:
        for p_ in args.extra_corpus.split(","):
            p_ = p_.strip()
            if p_ and os.path.exists(p_):
                inputs.append(p_)
    print(f"[tok] Training on: {inputs}")
    print(f"[tok] vocab_size={args.vocab_size} model_type={args.model_type}")

    spm.SentencePieceTrainer.train(
        input=",".join(inputs),
        model_prefix=str(MODEL_PREFIX),
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>",
        user_defined_symbols=",".join(SPECIAL_TOKENS),
        normalization_rule_name="nmt_nfkc",
        byte_fallback=True,
        split_digits=True,
        max_sentence_length=8192,
    )

    print(f"[tok] [OK] Saved {MODEL_PREFIX}.model and .vocab")
    sp = spm.SentencePieceProcessor()
    sp.Load(str(MODEL_PREFIX) + ".model")
    print(f"[tok] vocab_size = {sp.GetPieceSize()}")
    test_strings = [
        "Salom dunyo!",
        "Hello world!",
        "function add(a, b) { return a + b; }",
        "<|im_start|>user\nSalom<|im_end|>",
        "Yaxshimisiz?",
        "ўзбек тили",
    ]
    print("\n[tok] Sample encodings:")
    for s in test_strings:
        ids = sp.EncodeAsIds(s)
        print(f"  {s[:40]!r:42} -> {len(ids)} ids: {ids[:12]}...")


if __name__ == "__main__":
    main()
