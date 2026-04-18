# 💬 AI Chat Training (Uzbek + English)

Compact transformer (~25M params) for natural conversation in **uz + en**.
Trains on Google Colab T4 GPU in ~1.5–2 hours.

---

## 🚀 Colab — 7 qadam (faqat ko'chiring va bosing)

### 1. Colab ochish + GPU yoqish
👉 https://colab.research.google.com → `New notebook`

`Runtime` → `Change runtime type` → **`T4 GPU`** → **Save**
`Runtime` → **`Connect`**

### 2. GPU tekshirish
```bash
!nvidia-smi
```
`Tesla T4` ko'rinishi kerak.

### 3. Repo'ni clone qilish
```bash
%cd /content
!rm -rf Ai_chat_trainning-
!git clone https://github.com/Muhammadjonmurodullayev/Ai_chat_trainning-.git
%cd Ai_chat_trainning-
!git log --oneline -1
```

### 4. Kutubxonalar
```bash
!pip install -q -r requirements.txt
```

### 5. Dataset yaratish
```bash
!python chat/dataset_gen.py
```

### 6. Tokenizer (avtomatik adaptive)
```bash
!python chat/tokenizer_train.py
```
> Default `vocab_size=2000`, lekin corpus kichik bo'lsa avtomatik kichraytiradi (bulletproof retry).

### 7. 🔥 TRAINING
```bash
!python chat/train_chat.py --config configs/chat_gpu.json
```
~1.5–2 soat T4'da, 30 epoch.

### 8. Yuklab olish
```python
import shutil
from google.colab import files
shutil.make_archive("chat_results", "zip", ".", "checkpoints")
files.download("chat_results.zip")
```

---

## 📥 Lokalda deploy

`chat_results.zip`'ni oching va 4 ta faylni ko'chiring:

```
ai-coding-platform/services/model-service/checkpoints/chat/
    ├── chat_best.pt
    ├── chat_last.pt
    ├── chat_vocab.model
    └── chat_vocab.vocab
```

Keyin model-service'ni qayta ishga tushiring (yoki `POST /api/chat/reload`).

---

## 🧠 Texnik ma'lumot

| Element | Qiymat |
|---|---|
| Architecture | MiniTransformer (RoPE + SwiGLU) |
| Params | ~25M (`embed=384`, `layers=8`, `heads=8`) |
| Vocab | SentencePiece Unigram, adaptive (1374–4000) |
| Format | ChatML (`<\|im_start\|>` / `<\|im_end\|>`) |
| Loss | Masked — only assistant tokens count |
| Optimizer | AdamW + cosine LR + warmup |
| AMP | fp16 on T4/V100, bf16 on A100 |

---

## 🪪 License

MIT
