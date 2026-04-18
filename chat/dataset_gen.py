"""
Generate chat dataset — uz + en mixed.

Outputs:
  data/chat_corpus.txt        — raw text for SentencePiece training
  data/chat_train.jsonl       — list of conversations (~10–30k)
  data/chat_val.jsonl         — held-out (~5%)

Each line in chat_*.jsonl:
  {"messages": [{"role":"system|user|assistant","content":"..."}, ...]}

Sources combined:
  1. Hand-written seed (chat/seed_uz_en.jsonl)
  2. Synthetic templates (greetings, smalltalk, Q&A, refusals)
  3. Optional: Alpaca-cleaned subset translated/adapted
"""

from __future__ import annotations
import os
import json
import random
import argparse
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SEED_PATH = ROOT / "chat" / "seed_uz_en.jsonl"


# ─── Synthetic templates ────────────────────────────────

GREETINGS_UZ = [
    ("Salom", "Salom! Sizga qanday yordam bera olaman?"),
    ("Assalomu alaykum", "Vaalaykum assalom! Bugun nima bilan band ekansiz?"),
    ("salom", "Salom! Yaxshimisiz?"),
    ("Hayr", "Xayr, omadli kun bo'lsin!"),
    ("Rahmat", "Arzimaydi! Yana savolingiz bo'lsa, bemalol so'rang."),
    ("Ismingiz nima?", "Men AI Coding Platform yordamchisiman. Sizga kod yozish yoki savollarda yordam bera olaman."),
    ("Sen kimsan?", "Men sun'iy intellekt yordamchiman, kod yozish va suhbatlashish uchun moslashtirilganman."),
    ("Nima qila olasan?", "Men o'zbek va ingliz tillarida suhbatlashaman, JavaScript/TypeScript kodi yozaman, savollarga javob beraman va tushuntirishlar beraman."),
    ("Yaxshimisan?", "Rahmat, men yaxshiman! Siz qandaysiz?"),
    ("Yordam ber", "Albatta, qanday yordam kerak? Kod yozishmi, tushuntirishmi yoki boshqa narsami?"),
    ("Bugun havo qanday?", "Kechirasiz, men real vaqtdagi havo ma'lumotlariga kira olmayman. Lekin boshqa savol bo'lsa, javob bera olaman."),
    ("Soat necha?", "Kechirasiz, men real vaqtni bilmayman. Tizim soatiga qarang."),
    ("Sen robotmisan?", "Ha, men sun'iy intellekt asosidagi yordamchiman."),
    ("Qaysi tillarda gaplasha olasan?", "Hozir asosan o'zbek va ingliz tillarida gaplashaman. Kelajakda boshqa tillar ham qo'shiladi."),
    ("Kim seni yaratdi?", "Meni AI Coding Platform jamoasi mahalliy ravishda noldan o'qitgan."),
    ("Tushundim", "Yaxshi! Yana savolingiz bormi?"),
    ("Xo'p", "Bo'pti! Boshqa nima bilan yordam bera olaman?"),
]

GREETINGS_EN = [
    ("Hi", "Hi! How can I help you today?"),
    ("Hello", "Hello! What can I do for you?"),
    ("Hey", "Hey there! How can I assist?"),
    ("Good morning", "Good morning! How can I help?"),
    ("Good evening", "Good evening! What's on your mind?"),
    ("Thanks", "You're welcome! Let me know if you need anything else."),
    ("Thank you", "You're welcome!"),
    ("Bye", "Goodbye! Have a great day."),
    ("What's your name?", "I'm the AI Coding Platform assistant. I can help with code and conversation."),
    ("Who are you?", "I'm an AI assistant trained from scratch to help with coding and chat."),
    ("What can you do?", "I can chat in Uzbek and English, write JavaScript/TypeScript code, answer questions, and explain concepts."),
    ("How are you?", "I'm doing well, thanks! How about you?"),
    ("Help me", "Sure! What do you need help with — coding, explanation, or something else?"),
    ("Are you a robot?", "Yes, I'm an AI assistant."),
    ("Who made you?", "I was trained from scratch by the AI Coding Platform team."),
    ("What languages do you speak?", "Right now I primarily speak Uzbek and English. More languages will be added later."),
    ("Got it", "Great! Anything else you'd like to know?"),
    ("OK", "Alright! What's next?"),
]

SMALLTALK_UZ = [
    ("Charchadim", "Bir oz dam oling! Kichik tanaffus juda foydali bo'ladi."),
    ("Kayfiyatim yomon", "Buni eshitganim uchun afsusdaman. Kayfiyatingizni ko'tarish uchun nima qila olaman?"),
    ("Bo'shman", "Yaxshi! Balki yangi narsa o'rgansak yoki kichik loyiha qilsak?"),
    ("Charchatdi", "Nima sizni charchatdi? Gaplashib ko'ring, yengilroq bo'ladi."),
    ("Quvonchliman", "Buni eshitganimdan xursandman! Yaxshi yangiliklar bormi?"),
    ("Nimani maslahat berasan?", "Aniqroq nima haqida maslahat kerak? Kod, dizayn, yoki o'rganish bo'yicha?"),
    ("Charchadik", "Tushunaman. Kichik tanaffus oling, keyin davom etaylik."),
]

SMALLTALK_EN = [
    ("I'm tired", "Take a short break — even five minutes helps a lot."),
    ("I feel sad", "I'm sorry to hear that. Want to talk about it, or shall we focus on something else?"),
    ("I'm bored", "Want to learn something new or build a small project together?"),
    ("I'm happy", "That's great to hear! What's making you happy?"),
    ("What do you suggest?", "Could you be more specific — coding, learning, or something else?"),
]

QA_UZ = [
    ("JavaScript nima?",
     "JavaScript — bu veb saytlarda interaktivlik qo'shish uchun ishlatiladigan dasturlash tili. "
     "Brauzerda, serverda (Node.js) va mobil ilovalarda ishlatiladi."),
    ("React nima?",
     "React — bu Facebook tomonidan yaratilgan UI kutubxona. Komponentlar asosida foydalanuvchi interfeysini quradi."),
    ("Funksiya nima?",
     "Funksiya — bu ma'lum bir vazifani bajaradigan kod bo'lagi. Uni bir necha marta chaqirish mumkin."),
    ("Massiv (array) nima?",
     "Massiv — bu bir nechta qiymatlarni tartiblangan ro'yxatda saqlash uchun ishlatiladigan ma'lumot turi."),
    ("Obyekt (object) nima?",
     "Obyekt — bu kalit-qiymat juftliklaridan iborat ma'lumot tuzilmasi. Misol: {ism: 'Ali', yosh: 25}."),
    ("Async nima?",
     "Async — bu asinxron operatsiyalar uchun ishlatiladi. await bilan birga Promise natijasini kutamiz."),
    ("API nima?",
     "API (Application Programming Interface) — bu dasturlar bir-biri bilan bog'lanish uchun ishlatadigan qoidalar to'plami."),
    ("Database nima?",
     "Database — bu ma'lumotlarni tartibli saqlash va boshqarish uchun ishlatiladigan tizim. Masalan: PostgreSQL, MongoDB."),
    ("Git nima?",
     "Git — bu kod o'zgarishlarini kuzatib boruvchi versiya nazorat tizimi. Jamoa bilan ishlash uchun zarur."),
    ("HTML va CSS farqi nima?",
     "HTML — sahifa tarkibini (matn, rasm, tugma) yaratadi. CSS esa shu elementlarning ko'rinishini (rang, o'lcham, joylashuv) belgilaydi."),
]

QA_EN = [
    ("What is JavaScript?",
     "JavaScript is a programming language used to add interactivity to web pages. "
     "It runs in browsers, on servers (Node.js), and in mobile apps."),
    ("What is React?",
     "React is a UI library created by Facebook. It builds user interfaces using reusable components."),
    ("What is a function?",
     "A function is a reusable block of code that performs a specific task. You can call it multiple times."),
    ("What is an array?",
     "An array is a data type that stores multiple values in an ordered list."),
    ("What is an object?",
     "An object is a data structure made of key-value pairs. Example: {name: 'Ali', age: 25}."),
    ("What is async/await?",
     "async/await is JavaScript syntax for working with Promises. await pauses execution until the Promise resolves."),
    ("What is an API?",
     "An API (Application Programming Interface) is a set of rules that lets programs communicate with each other."),
    ("What is a database?",
     "A database is a system for organizing and storing data. Examples: PostgreSQL, MongoDB."),
    ("What is Git?",
     "Git is a version control system that tracks changes in code. It's essential for team collaboration."),
    ("Difference between HTML and CSS?",
     "HTML defines the structure of a page (text, images, buttons). CSS defines how those elements look (color, size, layout)."),
]

REFUSALS_UZ = [
    ("Parolimni o'g'irlab ber", "Kechirasiz, men bunga yordam bera olmayman. Bu xavfsizlik qoidalariga zid."),
    ("Virus yozib ber", "Kechirasiz, zararli dastur yaratishda yordam bera olmayman."),
    ("Boshqa odamning hisobiga kir", "Kechirasiz, bunday so'rovga javob bera olmayman."),
]

REFUSALS_EN = [
    ("Steal someone's password", "Sorry, I can't help with that — it goes against safety rules."),
    ("Write malware", "Sorry, I won't help create malicious software."),
    ("Hack into an account", "Sorry, I can't assist with that request."),
]

CODE_QA = [
    ("Salom dunyo dasturini yoz",
     "```javascript\nconsole.log('Salom, dunyo!');\n```\nBu eng oddiy JavaScript dasturi."),
    ("Write hello world",
     "```javascript\nconsole.log('Hello, world!');\n```\nThat's the classic Hello World."),
    ("Ikki sonni qo'shadigan funksiya",
     "```javascript\nfunction add(a, b) {\n  return a + b;\n}\n```\nIshlatish: `add(2, 3)` → `5`."),
    ("Function to add two numbers",
     "```javascript\nfunction add(a, b) {\n  return a + b;\n}\n```\nUsage: `add(2, 3)` returns `5`."),
    ("Array elementlarini yig'indisi",
     "```javascript\nconst sum = arr => arr.reduce((a, b) => a + b, 0);\nsum([1, 2, 3]); // 6\n```"),
    ("Sum of array elements",
     "```javascript\nconst sum = arr => arr.reduce((a, b) => a + b, 0);\nsum([1, 2, 3]); // 6\n```"),
]


SYSTEM_PROMPTS = [
    "Siz AI Coding Platform yordamchisiz. O'zbek va ingliz tillarida tabiiy va do'stona suhbat qiling.",
    "You are the AI Coding Platform assistant. Chat naturally in Uzbek and English. Be concise.",
    "Siz foydalanuvchiga kod yozishda va savollarga javob berishda yordam beruvchi AI siz.",
    "You are a helpful AI assistant for coders. Reply in the user's language.",
]


def build_single_turn(pairs, n_repeat=2):
    out = []
    for _ in range(n_repeat):
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


def build_multi_turn(n=500):
    """Generate multi-turn dialogues by chaining 2-4 turns."""
    all_pairs = (
        GREETINGS_UZ + GREETINGS_EN +
        SMALLTALK_UZ + SMALLTALK_EN +
        QA_UZ + QA_EN + CODE_QA
    )
    out = []
    for _ in range(n):
        turns = random.randint(2, 4)
        msgs = [{"role": "system", "content": random.choice(SYSTEM_PROMPTS)}]
        for _ in range(turns):
            u, a = random.choice(all_pairs)
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
        out.append({"messages": msgs})
    return out


def load_seed():
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
    """Flat text corpus for SentencePiece training."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            for m in r["messages"]:
                f.write(m["content"].replace("\n", " ") + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--multi_turn", type=int, default=2000)
    p.add_argument("--repeat_single", type=int, default=4)
    p.add_argument("--val_ratio", type=float, default=0.05)
    args = p.parse_args()

    seed = load_seed()
    print(f"[gen] Loaded {len(seed)} seed conversations from {SEED_PATH}")

    single = (
        build_single_turn(GREETINGS_UZ, args.repeat_single) +
        build_single_turn(GREETINGS_EN, args.repeat_single) +
        build_single_turn(SMALLTALK_UZ, 2) +
        build_single_turn(SMALLTALK_EN, 2) +
        build_single_turn(QA_UZ, 3) +
        build_single_turn(QA_EN, 3) +
        build_single_turn(REFUSALS_UZ, 2) +
        build_single_turn(REFUSALS_EN, 2) +
        build_single_turn(CODE_QA, 3)
    )
    multi = build_multi_turn(args.multi_turn)

    all_rows = seed + single + multi
    random.shuffle(all_rows)
    print(f"[gen] Total conversations: {len(all_rows)}")
    print(f"      seed={len(seed)} single={len(single)} multi={len(multi)}")

    n_val = max(1, int(len(all_rows) * args.val_ratio))
    val = all_rows[:n_val]
    train = all_rows[n_val:]

    write_jsonl(train, DATA_DIR / "chat_train.jsonl")
    write_jsonl(val, DATA_DIR / "chat_val.jsonl")
    write_corpus(all_rows, DATA_DIR / "chat_corpus.txt")

    print(f"[gen] Wrote chat_train.jsonl ({len(train)})")
    print(f"[gen] Wrote chat_val.jsonl ({len(val)})")
    print(f"[gen] Wrote chat_corpus.txt for SentencePiece")


if __name__ == "__main__":
    main()
