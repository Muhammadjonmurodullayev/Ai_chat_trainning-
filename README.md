# 💬 AI Chat Training (Uzbek + English)

Conversational AI training pipeline for the **Ai_baza01** platform.
Trains a compact transformer (~25M params) to chat naturally in Uzbek and English.

> This repo is **chat-only**. Code-generation training lives in a separate repo.

---

## 🎯 What this trains

- **Architecture**: MiniTransformer v2 (RoPE + SwiGLU + RMSNorm)
  - `embed_dim=384`, `num_layers=8`, `num_heads=8`, `ff_dim=1536`, `max_seq_len=1024`
  - ~25M parameters (fits comfortably on a free Colab T4)
- **Tokenizer**: SentencePiece Unigram, **16k vocab**, multilingual
  - Latin, Cyrillic, Uzbek diacritics (ʻʼ), digits, code symbols
- **Format**: ChatML
  ```
  <|im_start|>system
  You are a helpful assistant.<|im_end|>
  <|im_start|>user
  Salom!<|im_end|>
  <|im_start|>assistant
  Salom! Sizga qanday yordam bera olaman?<|im_end|>
  ```
- **Loss masking**: only assistant tokens contribute to loss (instruction tuning)

---

## 🚀 Quick start — Google Colab (recommended)

1. Open [`notebooks/train_chat_colab.ipynb`](notebooks/train_chat_colab.ipynb) in Colab
2. **Runtime → Change runtime type → GPU (T4)**
3. **Run all cells**
4. After ~1–2 hours, download from `checkpoints/`:
   - `chat_best.pt` — best validation checkpoint
   - `chat_last.pt` — final checkpoint
   - `chat_vocab.model` — SentencePiece tokenizer
   - `chat_vocab.vocab` — vocabulary list

---

## 🖥 Quick start — local

```bash
# 1. Install
pip install -r requirements.txt

# 2. Generate dataset (uz+en greetings, Q&A, refusals, code-Q&A, multi-turn)
python chat/dataset_gen.py

# 3. Train tokenizer (SentencePiece, 16k)
python chat/tokenizer_train.py --vocab_size 16000

# 4. Train model
python chat/train_chat.py --config configs/chat_gpu.json
```

Outputs land in `checkpoints/`.

---

## 📂 Repo layout

```
Ai_chat_training/
├── chat/
│   ├── dataset_gen.py       # Synthetic uz+en chat dataset generator
│   ├── seed_uz_en.jsonl     # ~30 hand-written multi-turn conversations
│   ├── tokenizer_train.py   # SentencePiece training
│   ├── chat_dataset.py      # PyTorch Dataset with loss masking
│   └── train_chat.py        # Main training loop (AMP + OOM-safe)
├── model/
│   ├── transformer.py       # MiniTransformer (RoPE + SwiGLU)
│   ├── attention.py
│   ├── embeddings.py
│   ├── feedforward.py
│   └── rope.py
├── configs/
│   └── chat_gpu.json        # Training hyperparameters
├── notebooks/
│   └── train_chat_colab.ipynb
├── checkpoints/             # Output: trained models
├── data/                    # Generated dataset (gitignored)
├── requirements.txt
└── README.md
```

---

## ⚙️ Training config (`configs/chat_gpu.json`)

| Param | Value |
|------|------|
| epochs | 30 |
| batch_size | 4 |
| grad_accum | 4 |
| effective_batch | 16 |
| learning_rate | 3e-4 |
| warmup_ratio | 0.05 |
| label_smoothing | 0.05 |
| weight_decay | 0.01 |
| AMP | enabled (fp16 on T4) |
| save_every_steps | 1000 |
| early_stopping_patience | 5 |

Adjust to your GPU. On A100 you can comfortably push `batch_size=16`.

---

## 📥 After training — deploy to Ai_baza01

Place these three files into the platform:

```
ai-coding-platform/services/model-service/checkpoints/chat/
    ├── chat_best.pt
    ├── chat_last.pt
    └── chat_vocab.model
```

Then either restart `model-service` or hot-reload:

```bash
curl -X POST http://localhost:8000/api/chat/reload
```

Test:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Salom! Ahvoling qalay?"}]}'
```

---

## 📊 Expected metrics (T4, ~1.5h)

| Metric | Target |
|------|------|
| train loss | < 1.5 |
| val loss | < 2.0 |
| perplexity | < 7.5 |
| short-greeting quality | natural uz/en replies |

---

## 🧠 Adding more data

Drop more JSONL into `chat/seed_uz_en.jsonl`. Format:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Then re-run `python chat/dataset_gen.py`.

---

## 🪪 License

MIT
