"""
═══════════════════════════════════════════════════════
 TRAIN_CHAT.PY — Chat-tuned MiniTransformer training
═══════════════════════════════════════════════════════

Trains a small chat model with:
  - SentencePiece tokenizer (16k vocab, uz+en)
  - ChatML format with <|im_start|>/<|im_end|>
  - Loss masked to assistant tokens only (instruction tuning)
  - Mixed precision on GPU (T4/V100/A100)
  - OOM-safe loop, checkpoint resume

Run on Colab:
  pip install sentencepiece tqdm
  python chat/dataset_gen.py
  python chat/tokenizer_train.py --vocab_size 16000
  python chat/train_chat.py --config configs/chat_gpu.json
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
import gc
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.transformer import MiniTransformer, TransformerConfig
from chat.chat_dataset import ChatDataset, chat_collate, LOSS_IGNORE


# ─── Device ───────────────────────────────────────

def detect_device():
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            cap = torch.cuda.get_device_capability(0)
            print(f"  GPU: {name} ({mem:.1f} GB, compute {cap[0]}.{cap[1]})")
            print(f"  CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")
            return torch.device("cuda"), True, cap[0] >= 7
    except Exception as e:
        print(f"  CUDA detection failed: {e}")
    print(f"  CPU mode  PyTorch: {torch.__version__}")
    return torch.device("cpu"), False, False


# ─── LR scheduler ──────────────────────────────────

def cosine_lr(step, total_steps, warmup_steps, base_lr, min_lr_ratio=0.1):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cos)


# ─── Checkpoint ───────────────────────────────────

def save_checkpoint(path, model, optimizer, scaler, epoch, step, val_loss, train_loss, cfg, vocab_size):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "model_config": cfg,
        "config": cfg,         # alias
        "vocab_size": vocab_size,
        "epoch": epoch,
        "global_step": step,
        "val_loss": val_loss,
        "best_val_loss": val_loss,
        "train_loss": train_loss,
    }, path)


# ─── Train ────────────────────────────────────────

def train(config_path: str, resume: str | None = None):
    print("\n" + "=" * 60)
    print("  CHAT TRAINING — MiniTransformer (instruction tuning)")
    print("=" * 60)
    print(f"  Config: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device, is_gpu, supports_amp = detect_device()
    use_amp = is_gpu and supports_amp and cfg["training"].get("mixed_precision", True)

    # --- Tokenizer ---
    import sentencepiece as spm
    tok_path = ROOT / cfg["data"]["tokenizer_path"]
    if not tok_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {tok_path}\n"
            f"Run: python chat/tokenizer_train.py"
        )
    sp = spm.SentencePieceProcessor()
    sp.Load(str(tok_path))
    vocab_size = sp.GetPieceSize()
    print(f"  Tokenizer: {tok_path.name} (vocab={vocab_size})")

    pad_id = 0
    im_start = sp.PieceToId("<|im_start|>")
    im_end = sp.PieceToId("<|im_end|>")
    print(f"  Special: pad={pad_id} im_start={im_start} im_end={im_end}")

    # --- Datasets ---
    train_ds = ChatDataset(
        ROOT / cfg["data"]["train_path"],
        sp, max_seq_len=cfg["model"]["max_seq_len"],
        im_start_id=im_start, im_end_id=im_end, pad_id=pad_id,
    )
    val_ds = ChatDataset(
        ROOT / cfg["data"]["val_path"],
        sp, max_seq_len=cfg["model"]["max_seq_len"],
        im_start_id=im_start, im_end_id=im_end, pad_id=pad_id,
    )

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=0,
        collate_fn=lambda b: chat_collate(b, pad_id=pad_id),
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=0,
        collate_fn=lambda b: chat_collate(b, pad_id=pad_id),
    )
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # --- Model ---
    model_cfg_dict = {
        "vocab_size": vocab_size,
        "embed_dim": cfg["model"]["embed_dim"],
        "num_heads": cfg["model"]["num_heads"],
        "num_layers": cfg["model"]["num_layers"],
        "ff_dim": cfg["model"]["ff_dim"],
        "max_seq_len": cfg["model"]["max_seq_len"],
        "dropout": cfg["model"]["dropout"],
        "padding_idx": pad_id,
    }
    model_cfg = TransformerConfig(**model_cfg_dict)
    model = MiniTransformer(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,} ({n_params/1e6:.2f}M)")

    # --- Optimizer + scaler ---
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        betas=(0.9, 0.95),
    )
    if use_amp:
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None
    print(f"  AMP enabled: {use_amp}")

    criterion = nn.CrossEntropyLoss(
        ignore_index=LOSS_IGNORE,
        label_smoothing=cfg["training"].get("label_smoothing", 0.0),
    )

    # --- Schedule ---
    epochs = cfg["training"]["epochs"]
    accum = cfg["training"].get("accumulation_steps", 1)
    total_steps = (len(train_loader) // accum + 1) * epochs
    warmup_steps = max(1, int(total_steps * cfg["training"].get("warmup_ratio", 0.05)))
    base_lr = cfg["training"]["learning_rate"]
    grad_clip = cfg["training"].get("grad_clip", 1.0)
    print(f"  Total steps: {total_steps}  Warmup: {warmup_steps}")

    # --- Resume ---
    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        if ck.get("optimizer_state_dict"):
            optim.load_state_dict(ck["optimizer_state_dict"])
        if scaler is not None and ck.get("scaler_state_dict"):
            scaler.load_state_dict(ck["scaler_state_dict"])
        start_epoch = ck.get("epoch", 0) + 1
        global_step = ck.get("global_step", 0)
        best_val = ck.get("best_val_loss", float("inf"))
        print(f"  Resumed from {resume} (epoch {start_epoch}, best_val={best_val:.4f})")

    # --- Output paths ---
    save_dir = ROOT / cfg["checkpoint"]["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    BEST = save_dir / "chat_best.pt"
    LAST = save_dir / "chat_last.pt"
    save_every_steps = cfg["checkpoint"].get("save_every_n_steps", 0)
    save_every_epochs = cfg["checkpoint"].get("save_every_n_epochs", 0)
    patience = cfg["training"].get("early_stopping_patience", 0)
    bad_epochs = 0

    history = []
    t0 = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        ep_loss = 0.0
        ep_tokens = 0
        optim.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for it, (input_ids, labels, attn_mask) in enumerate(pbar):
            try:
                input_ids = input_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Shift for causal LM
                inp = input_ids[:, :-1].contiguous()
                tgt = labels[:, 1:].contiguous()

                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = model(inp)
                        loss = criterion(logits.reshape(-1, logits.size(-1)),
                                         tgt.reshape(-1)) / accum
                    scaler.scale(loss).backward()
                else:
                    logits = model(inp)
                    loss = criterion(logits.reshape(-1, logits.size(-1)),
                                     tgt.reshape(-1)) / accum
                    loss.backward()

                if (it + 1) % accum == 0 or (it + 1) == len(train_loader):
                    # LR update
                    lr = cosine_lr(global_step, total_steps, warmup_steps, base_lr)
                    for pg in optim.param_groups:
                        pg["lr"] = lr

                    if use_amp:
                        scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        scaler.step(optim)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        optim.step()
                    optim.zero_grad(set_to_none=True)
                    global_step += 1

                with torch.no_grad():
                    n_tokens = (tgt != LOSS_IGNORE).sum().item()
                    ep_loss += loss.item() * accum * max(1, n_tokens)
                    ep_tokens += n_tokens

                pbar.set_postfix(loss=f"{loss.item()*accum:.3f}", lr=f"{optim.param_groups[0]['lr']:.2e}")

                # Step-based save
                if save_every_steps and global_step > 0 and global_step % save_every_steps == 0:
                    save_checkpoint(LAST, model, optim, scaler, epoch, global_step,
                                    val_loss=best_val, train_loss=loss.item()*accum,
                                    cfg=model_cfg_dict, vocab_size=vocab_size)
            except torch.cuda.OutOfMemoryError:
                print("\n  [OOM] reducing memory, skipping batch...")
                optim.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                continue

        avg_train = ep_loss / max(1, ep_tokens)

        # ─ Validation ─
        model.eval()
        val_loss = 0.0
        val_tokens = 0
        with torch.no_grad():
            for input_ids, labels, attn_mask in val_loader:
                input_ids = input_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                inp = input_ids[:, :-1].contiguous()
                tgt = labels[:, 1:].contiguous()
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = model(inp)
                        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                else:
                    logits = model(inp)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                n = (tgt != LOSS_IGNORE).sum().item()
                val_loss += loss.item() * max(1, n)
                val_tokens += n
        avg_val = val_loss / max(1, val_tokens)

        elapsed = time.time() - t0
        print(f"  [Epoch {epoch+1}] train={avg_train:.4f} val={avg_val:.4f} "
              f"lr={optim.param_groups[0]['lr']:.2e} elapsed={elapsed:.0f}s")
        history.append({
            "epoch": epoch + 1, "train_loss": round(avg_train, 4),
            "val_loss": round(avg_val, 4),
            "lr": optim.param_groups[0]["lr"],
        })

        # Save last + maybe best
        save_checkpoint(LAST, model, optim, scaler, epoch, global_step,
                        val_loss=avg_val, train_loss=avg_train,
                        cfg=model_cfg_dict, vocab_size=vocab_size)
        if avg_val < best_val:
            best_val = avg_val
            save_checkpoint(BEST, model, optim, scaler, epoch, global_step,
                            val_loss=avg_val, train_loss=avg_train,
                            cfg=model_cfg_dict, vocab_size=vocab_size)
            print(f"  [BEST] saved (val={avg_val:.4f})")
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience and bad_epochs >= patience:
                print(f"  [EARLY STOP] no improvement for {patience} epochs")
                break

        if save_every_epochs and (epoch + 1) % save_every_epochs == 0:
            ep_path = save_dir / f"chat_epoch_{epoch+1}.pt"
            save_checkpoint(ep_path, model, optim, scaler, epoch, global_step,
                            val_loss=avg_val, train_loss=avg_train,
                            cfg=model_cfg_dict, vocab_size=vocab_size)

    duration = time.time() - t0
    log_path = ROOT / "logs" / f"chat_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "completed_at": datetime.now().isoformat(),
            "duration_sec": round(duration, 1),
            "best_val_loss": round(best_val, 4),
            "total_epochs": len(history),
            "config": cfg, "model_params": n_params,
            "device": str(device), "mixed_precision": use_amp,
            "history": history,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Done in {duration:.1f}s. Best val_loss={best_val:.4f}")
    print(f"  Best:   {BEST}")
    print(f"  Last:   {LAST}")
    print(f"  Vocab:  {ROOT / cfg['data']['tokenizer_path']}")
    print(f"  Log:    {log_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/chat_gpu.json")
    p.add_argument("--resume", default=None)
    args = p.parse_args()
    train(args.config, args.resume)
