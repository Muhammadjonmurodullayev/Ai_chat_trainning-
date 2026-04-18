"""
Generate chat dataset — uz + en mixed.

Outputs:
  data/chat_corpus.txt        — raw text for SentencePiece training
  data/chat_train.jsonl       — training conversations
  data/chat_val.jsonl         — held-out (~5%)

Each line:
  {"messages": [{"role":"system|user|assistant","content":"..."}, ...]}

Uses seed_uz_extended.all_pairs() (6700+ unique pairs) as primary source.
"""

from __future__ import annotations
import os
import json
import random
import argparse
from pathlib import Path

# Local import (same package)
from seed_uz_extended import all_pairs as load_extended_pairs

random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SEED_PATH = ROOT / "chat" / "seed_uz_en.jsonl"


SYSTEM_PROMPTS = [
    "Siz AI Coding Platform yordamchisiz. O'zbek va ingliz tillarida tabiiy va do'stona suhbat qiling.",
    "You are the AI Coding Platform assistant. Chat naturally in Uzbek and English. Be concise.",
    "Siz foydalanuvchiga kod yozishda va savollarga javob berishda yordam beruvchi AI siz.",
    "You are a helpful AI assistant for coders. Reply in the user's language.",
    "Siz o'zbek tilini yaxshi biluvchi AI yordamchisiz. Aniq va qisqa javob bering.",
    "You are a friendly AI tutor. Explain things clearly and simply.",
]


# ── Single-turn (each pair → 1 conversation, repeated N times w/ different system) ──

def build_single_turn(pairs, repeat=2):
    """Each (user, assistant) pair becomes a single-turn conversation."""
    out = []
    for _ in range(repeat):
        for u, a in pairs:
            sys = random.choice(SYSTEM_PROMPTS)
            out.append({
                "messages": [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": a},
                ]
            })
    return out


# ── Multi-turn: realistic chains with continuation hints ──

CONTINUATIONS_UZ = [
    "Yana boshqa nima yordam beray?",
    "Boshqa savol bormi?",
    "Yana bilmoqchi bo'lgan narsa bormi?",
    "Davom etamizmi?",
]

CONTINUATIONS_EN = [
    "Anything else I can help with?",
    "Any other questions?",
    "Want me to explain more?",
    "Should we continue?",
]

FOLLOWUPS_UZ = [
    ("Rahmat", "Arzimaydi! Yana savolingiz bo'lsa, bemalol so'rang."),
    ("Tushundim", "Yaxshi! Yana yordam kerak bo'lsa ayting."),
    ("Yana misol bering", "Albatta, qaysi mavzu bo'yicha misol kerak?"),
    ("Tushuntirib bering", "Aniqroq qaysi qismini tushuntirishimni xohlaysiz?"),
    ("Bo'pti", "Mayli! Boshqa nima qilamiz?"),
    ("Aniqroq aytib bering", "Albatta. Qaysi joyi noaniq qoldi?"),
    ("Boshqa misol", "Yaxshi, mana boshqa misol kerak bo'lsa ayting."),
]

FOLLOWUPS_EN = [
    ("Thanks", "You're welcome! Let me know if you have more questions."),
    ("Got it", "Great! Anything else you'd like to know?"),
    ("Give another example", "Sure, on which topic would you like an example?"),
    ("Explain more", "Of course. Which part should I clarify?"),
    ("OK", "Alright, what's next?"),
    ("Tell me more", "Happy to. What aspect interests you most?"),
]


def build_multi_turn(pairs, n=3000):
    """Generate multi-turn dialogues (2-4 user turns) by chaining pairs."""
    out = []
    for _ in range(n):
        turns = random.randint(2, 4)
        msgs = [{"role": "system", "content": random.choice(SYSTEM_PROMPTS)}]
        # Detect language preference from first pair
        for i in range(turns):
            u, a = random.choice(pairs)
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
        out.append({"messages": msgs})
    return out


def build_followup_multi_turn(pairs, n=2000):
    """First turn is a real question, second is a followup like 'thanks'/'got it'."""
    fup_uz = FOLLOWUPS_UZ
    fup_en = FOLLOWUPS_EN
    out = []
    for _ in range(n):
        u1, a1 = random.choice(pairs)
        # crude language detection
        lang_en = any(w in u1.lower() for w in [" what ", "how ", "why ", "the ", "is ", " a "]) or u1[:2].lower() in {"hi", "he", "th", "wh"}
        fup_pool = fup_en if lang_en else fup_uz
        u2, a2 = random.choice(fup_pool)
        out.append({
            "messages": [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)},
                {"role": "user", "content": u1},
                {"role": "assistant", "content": a1},
                {"role": "user", "content": u2},
                {"role": "assistant", "content": a2},
            ]
        })
    return out


def load_seed_jsonl():
    """Original hand-written multi-turn seed file."""
    if not SEED_PATH.exists():
        return []
    out = []
    with SEED_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_corpus(rows, path):
    """Flat text corpus for SentencePiece training (one sentence per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            for m in r["messages"]:
                # Split content into individual lines for SP
                for line in m["content"].splitlines():
                    line = line.strip()
                    if line:
                        f.write(line + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeat_single", type=int, default=2,
                   help="How many times each unique pair appears as single-turn")
    p.add_argument("--multi_turn", type=int, default=4000,
                   help="Number of random multi-turn dialogues")
    p.add_argument("--followup_turn", type=int, default=2500,
                   help="Number of question + followup dialogues")
    p.add_argument("--val_ratio", type=float, default=0.05)
    args = p.parse_args()

    print("[gen] Loading extended pair set...")
    pairs = load_extended_pairs()
    print(f"[gen] Loaded {len(pairs)} unique (user, assistant) pairs")

    seed = load_seed_jsonl()
    print(f"[gen] Loaded {len(seed)} hand-written seed conversations")

    print("[gen] Building single-turn conversations...")
    single = build_single_turn(pairs, args.repeat_single)
    print(f"        single-turn: {len(single)}")

    print("[gen] Building multi-turn conversations...")
    multi = build_multi_turn(pairs, args.multi_turn)
    print(f"        multi-turn: {len(multi)}")

    print("[gen] Building followup conversations...")
    fup = build_followup_multi_turn(pairs, args.followup_turn)
    print(f"        followup: {len(fup)}")

    all_rows = seed + single + multi + fup
    random.shuffle(all_rows)
    print(f"[gen] Total conversations: {len(all_rows):,}")

    n_val = max(1, int(len(all_rows) * args.val_ratio))
    val = all_rows[:n_val]
    train = all_rows[n_val:]

    write_jsonl(train, DATA_DIR / "chat_train.jsonl")
    write_jsonl(val, DATA_DIR / "chat_val.jsonl")
    write_corpus(all_rows, DATA_DIR / "chat_corpus.txt")

    train_mb = (DATA_DIR / "chat_train.jsonl").stat().st_size / (1024 * 1024)
    corpus_mb = (DATA_DIR / "chat_corpus.txt").stat().st_size / (1024 * 1024)
    print(f"[gen] Wrote chat_train.jsonl ({len(train):,} convs, {train_mb:.1f} MB)")
    print(f"[gen] Wrote chat_val.jsonl ({len(val):,} convs)")
    print(f"[gen] Wrote chat_corpus.txt ({corpus_mb:.1f} MB) for SentencePiece")


if __name__ == "__main__":
    main()
