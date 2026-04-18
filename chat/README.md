# Chat training subsystem (uz + en)

Parallel to the code-only training script (`train.py`).

## Files
- `tokenizer_train.py` — train SentencePiece on uz+en+code corpus
- `dataset_gen.py` — generate synthetic chat data (uz+en greetings, Q&A, refusals)
- `seed_uz_en.jsonl` — hand-written seed conversations
- `chat_dataset.py` — PyTorch Dataset with loss masking (only assistant tokens contribute to loss)
- `train_chat.py` — main training loop (Colab-ready, AMP, OOM-safe)

## Run order on Colab
```bash
pip install -r requirements.txt sentencepiece datasets
# 1) Build chat corpus + seed
python chat/dataset_gen.py
# 2) Train tokenizer (~2 min)
python chat/tokenizer_train.py --vocab_size 16000
# 3) Train model (~30-90 min on T4)
python chat/train_chat.py --config configs/chat_gpu.json
```

Outputs:
- `checkpoints/chat/chat_vocab.model` (SentencePiece)
- `checkpoints/chat/chat_best.pt`
- `checkpoints/chat/chat_last.pt`

Place those into `ai-coding-platform/services/model-service/checkpoints/chat/`.
