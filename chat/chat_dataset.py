"""
ChatDataset — load JSONL chat data and produce (input_ids, labels) tensors
where labels are -100 for non-assistant tokens (loss masking).

Format of each JSONL line:
  {"messages": [{"role": "system|user|assistant", "content": "..."}, ...]}

ChatML format used:
  <|im_start|>role\ncontent<|im_end|>\n
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset

LOSS_IGNORE = -100


class ChatDataset(Dataset):
    """
    Each item: (input_ids, labels)
      - input_ids: token IDs of the rendered ChatML conversation
      - labels: same as input_ids but -100 for system/user tokens (only assistant counts)
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,            # sentencepiece.SentencePieceProcessor
        max_seq_len: int = 1024,
        im_start_id: int = None,
        im_end_id: int = None,
        pad_id: int = 0,
    ):
        self.path = Path(jsonl_path)
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        self.im_start_id = im_start_id if im_start_id is not None else tokenizer.PieceToId("<|im_start|>")
        self.im_end_id = im_end_id if im_end_id is not None else tokenizer.PieceToId("<|im_end|>")
        self.newline_id = tokenizer.EncodeAsIds("\n")[0] if tokenizer.EncodeAsIds("\n") else None

        self.examples: List[Dict] = self._load()
        print(f"[ChatDataset] {self.path.name}: {len(self.examples)} conversations")

    def _load(self) -> List[Dict]:
        rows = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows

    def __len__(self):
        return len(self.examples)

    def _encode_role(self, role: str) -> List[int]:
        # "<|im_start|>role\n" — start token + role string + newline
        ids = [self.im_start_id]
        ids += self.tok.EncodeAsIds(role + "\n")
        return ids

    def _encode_content(self, content: str) -> List[int]:
        # content + "<|im_end|>\n"
        ids = self.tok.EncodeAsIds(content)
        ids.append(self.im_end_id)
        nl = self.tok.EncodeAsIds("\n")
        if nl:
            ids += nl
        return ids

    def _build(self, messages: List[Dict[str, str]]) -> Tuple[List[int], List[int]]:
        """Build input_ids + labels with loss mask."""
        input_ids: List[int] = []
        labels: List[int] = []

        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")

            role_ids = self._encode_role(role)
            content_ids = self._encode_content(content)

            input_ids.extend(role_ids)
            labels.extend([LOSS_IGNORE] * len(role_ids))

            input_ids.extend(content_ids)
            if role == "assistant":
                labels.extend(content_ids)
            else:
                labels.extend([LOSS_IGNORE] * len(content_ids))

        # Truncate if too long
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
            labels = labels[: self.max_seq_len]
        return input_ids, labels

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ex = self.examples[idx]
        input_ids, labels = self._build(ex["messages"])
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


def chat_collate(batch, pad_id: int = 0):
    """Pad to max length in batch; return (input_ids, labels, attention_mask)."""
    max_len = max(b[0].size(0) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), LOSS_IGNORE, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, (ids, lbls) in enumerate(batch):
        L = ids.size(0)
        input_ids[i, :L] = ids
        labels[i, :L] = lbls
        attention_mask[i, :L] = 1

    return input_ids, labels, attention_mask
