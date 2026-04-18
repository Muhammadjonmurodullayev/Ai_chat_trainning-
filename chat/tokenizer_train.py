"""
Train SentencePiece Unigram tokenizer on uz+en+code corpus.

Outputs:
  checkpoints/chat/chat_vocab.model   (binary SentencePiece model)
  checkpoints/chat/chat_vocab.vocab   (text vocabulary)

Usage:
  python chat/tokenizer_train.py                 # safe default (2000)
  python chat/tokenizer_train.py --vocab_size 4000
"""

import os
import re
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints" / "chat"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = DATA_DIR / "chat_corpus.txt"
MODEL_PREFIX = CKPT_DIR / "chat_vocab"

SPECIAL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab_size", type=int, default=4000,
                   help="Target vocab size; auto-shrinks if corpus too small")
    p.add_argument("--character_coverage", type=float, default=0.9999)
    p.add_argument("--model_type", choices=["unigram", "bpe"], default="unigram")
    p.add_argument("--extra_corpus", type=str, default="")
    args = p.parse_args()

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Run dataset_gen.py first; missing {CORPUS_PATH}"
        )

    import sentencepiece as spm

    inputs = [str(CORPUS_PATH)]
    if args.extra_corpus:
        for p_ in args.extra_corpus.split(","):
            p_ = p_.strip()
            if p_ and os.path.exists(p_):
                inputs.append(p_)

    print(f"[tok] Training on: {inputs}")
    print(f"[tok] Target vocab_size={args.vocab_size}, model_type={args.model_type}")

    # ── BULLETPROOF: try requested size, shrink on failure ─────────
    candidates = [args.vocab_size, 2000, 1500, 1200, 1000, 800, 600, 500]
    seen = set()
    last_err = None

    for vs in candidates:
        if vs in seen:
            continue
        seen.add(vs)
        try:
            print(f"\n[tok] Attempt vocab_size={vs}  (hard_vocab_limit=False)")
            spm.SentencePieceTrainer.train(
                input=",".join(inputs),
                model_prefix=str(MODEL_PREFIX),
                vocab_size=vs,
                model_type=args.model_type,
                character_coverage=args.character_coverage,
                pad_id=0, unk_id=1, bos_id=2, eos_id=3,
                pad_piece="<pad>", unk_piece="<unk>",
                bos_piece="<s>", eos_piece="</s>",
                user_defined_symbols=SPECIAL_TOKENS,
                normalization_rule_name="nmt_nfkc",
                byte_fallback=True,
                split_digits=True,
                max_sentence_length=8192,
                hard_vocab_limit=False,   # ⭐ KEY FIX: allow smaller vocab
            )
            sp = spm.SentencePieceProcessor()
            sp.Load(str(MODEL_PREFIX) + ".model")
            actual = sp.GetPieceSize()
            print(f"[tok] [OK] SUCCESS - saved {MODEL_PREFIX}.model")
            print(f"[tok]    Requested={vs}, actual vocab_size={actual}")

            for tok in SPECIAL_TOKENS + ["<pad>", "<unk>", "<s>", "</s>"]:
                pid = sp.PieceToId(tok)
                print(f"[tok]    {tok!r:18} -> id={pid}")

            print("\n[tok] Sample encodings:")
            samples = [
                "Salom dunyo!",
                "Hello world!",
                "function add(a, b) { return a + b; }",
                "<|im_start|>user\nSalom<|im_end|>",
                "Yaxshimisiz? Men yaxshi, rahmat.",
            ]
            for s in samples:
                ids = sp.EncodeAsIds(s)
                print(f"  {s[:38]!r:40} -> {len(ids)} ids")
            return
        except RuntimeError as e:
            last_err = e
            msg = str(e)
            print(f"[tok] [WARN] vocab_size={vs} failed: {msg.splitlines()[-1][:120]}")
            m = re.search(r"<=\s*(\d+)", msg)
            if m:
                hint = int(m.group(1))
                if hint not in seen and hint > 100:
                    candidates.append(hint)
                    print(f"[tok]    SP suggests max={hint}, will try")
            continue

    raise RuntimeError(
        f"All vocab_size attempts failed. Last error: {last_err}"
    )


if __name__ == "__main__":
    main()
