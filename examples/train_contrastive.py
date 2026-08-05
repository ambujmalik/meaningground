#!/usr/bin/env python3
"""
Simple contrastive training example:
- Uses two encoders (shared or separate) to map (context, gloss) -> embeddings
- Uses InfoNCE / cosine temperature loss to align positive pairs in the batch
This is a minimal script for prototyping. For production, add better batching, caching, and data pipelines.
"""
import argparse
import json
from pathlib import Path
from typing import List
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# Simple dataset assuming JSONL with sentence + gloss fields
class GlossContextDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_len=128):
        self.items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                # require minimal fields
                if "sentence" in obj and "gloss" in obj:
                    self.items.append(obj)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        obj = self.items[idx]
        return obj["sentence"], obj["gloss"]

def collate_batch(batch, tokenizer, max_len):
    sentences, glosses = zip(*batch)
    sent_tokens = tokenizer(list(sentences), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    gloss_tokens = tokenizer(list(glosses), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return sent_tokens, gloss_tokens

class Encoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        # pooling: use CLS for simplicity
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = out.last_hidden_state[:,0]  # [CLS]
        return self.proj(pooled)

def info_nce_loss(x, y, temperature=0.05):
    # x, y: normalized embeddings (batch_size, dim)
    logits = torch.matmul(x, y.t()) / temperature
    labels = torch.arange(x.size(0), device=x.device)
    loss_x = nn.CrossEntropyLoss()(logits, labels)
    loss_y = nn.CrossEntropyLoss()(logits.t(), labels)
    return (loss_x + loss_y) / 2

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = GlossContextDataset(args.train, tokenizer, max_len=args.max_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_batch(b, tokenizer, args.max_len))

    # Use shared encoder or two separate encoders
    context_encoder = Encoder(args.model).to(device)
    gloss_encoder = Encoder(args.model).to(device)
    optimizer = torch.optim.AdamW(list(context_encoder.parameters()) + list(gloss_encoder.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        context_encoder.train(); gloss_encoder.train()
        total_loss = 0.0
        for step, (sent_tokens, gloss_tokens) in enumerate(dataloader):
            optimizer.zero_grad()
            s_input_ids = sent_tokens["input_ids"].to(device)
            s_attn = sent_tokens["attention_mask"].to(device)
            g_input_ids = gloss_tokens["input_ids"].to(device)
            g_attn = gloss_tokens["attention_mask"].to(device)

            s_emb = context_encoder(s_input_ids, s_attn)
            g_emb = gloss_encoder(g_input_ids, g_attn)

            # normalize
            s_norm = nn.functional.normalize(s_emb, dim=-1)
            g_norm = nn.functional.normalize(g_emb, dim=-1)

            loss = info_nce_loss(s_norm, g_norm, temperature=args.temp)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (step + 1) % args.log_steps == 0:
                avg = total_loss / args.log_steps
                print(f"Epoch {epoch+1} Step {step+1} avg_loss={avg:.4f}")
                total_loss = 0.0

    # save encoders
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(context_encoder.state_dict(), save_dir / "context_encoder.pt")
    torch.save(gloss_encoder.state_dict(), save_dir / "gloss_encoder.pt")
    print("Saved encoders to", save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="train JSONL file")
    parser.add_argument("--model", default="bert-base-uncased", help="backbone")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--log-steps", type=int, default=100)
    args = parser.parse_args()
    train(args)
