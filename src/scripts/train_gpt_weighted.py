# src/scripts/train_gpt_weighted.py

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from src.student_state import StudentState
from src.utils.flatten import flatten_dict
from src.utils.state_vector import state_to_tensor
from src.utils.state_encoder import StateEncoder

DATA_PATH = "src/data/synthetic_state_prompt_ext.jsonl"
OUTPUT_DIR = "ft-gpt-weighted"

class RewardPromptDataset(Dataset):
    def __init__(self, jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            self.examples = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "state":       ex["state"],
            "prompt":      ex["prompt"],
            "reward":      ex["reward"],
        }

def make_collate_fn(tokenizer, model, encoder, prefix_len, device):
    hidden_size = model.config.hidden_size
    def collate_fn(batch):
        batch_embeds = []
        batch_masks  = []
        batch_labels = []
        batch_rewards= []

        for ex in batch:
            # 1) State → prefix embeddings
            sv = torch.tensor(
                state_to_tensor(ex["state"]),
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)                         # [1, D]
            prefix_emb = encoder(sv)[0]            # [P, H]

            # 2) Prompt → token IDs → embeddings
            prompt_ids = tokenizer(
                ex["prompt"], add_special_tokens=False
            ).input_ids                            # List[int]
            text_ids   = torch.tensor([prompt_ids], device=device)
            text_emb   = model.transformer.wte(text_ids)[0]  # [T, H]

            # 3) Concat prefix + prompt
            emb   = torch.cat([prefix_emb, text_emb], dim=0) # [P+T, H]
            mask  = torch.ones(emb.size(0), dtype=torch.long, device=device)
            labels= torch.tensor(
                       [-100]*prefix_len + prompt_ids,
                       dtype=torch.long, device=device,
                    )                                  # [P+T]

            batch_embeds.append(emb)
            batch_masks .append(mask)
            batch_labels.append(labels)
            batch_rewards.append(ex["reward"])

        # 4) Pad batch to max length
        max_len = max(e.size(0) for e in batch_embeds)
        inputs_embeds, attention_mask, labels, rewards = [], [], [], []

        for emb, mask, lbl, r in zip(batch_embeds, batch_masks, batch_labels, batch_rewards):
            pad = max_len - emb.size(0)
            # pad embeddings
            emb_pad = torch.zeros((pad, hidden_size), device=device)
            inputs_embeds.append(torch.cat([emb, emb_pad], dim=0))
            # pad mask
            mask_pad = torch.zeros(pad, dtype=mask.dtype, device=device)
            attention_mask.append(torch.cat([mask, mask_pad], dim=0))
            # pad labels with -100
            lbl_pad  = torch.full((pad,), -100, dtype=lbl.dtype, device=device)
            labels.append(torch.cat([lbl, lbl_pad], dim=0))
            # collect reward
            rewards.append(r)

        return {
            "inputs_embeds":   torch.stack(inputs_embeds),    # [B, L, H]
            "attention_mask":  torch.stack(attention_mask),   # [B, L]
            "labels":          torch.stack(labels),           # [B, L]
            "rewards":         torch.tensor(rewards,         # [B]
                                  dtype=torch.float32,
                                  device=device),
        }

    return collate_fn

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract rewards
        rewards = inputs.pop("rewards")                       # [B]
        # Forward pass
        outputs = model(**inputs)
        logits  = outputs.logits                              # [B, L, V]
        labels  = inputs["labels"]                            # [B, L]

        # Compute per-token negative log likelihood
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        # Flatten [B*L, V] and [B*L]
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        loss_flat   = loss_fct(logits_flat, labels_flat)      # [B*L]
        loss_per_token = loss_flat.view(labels.size())       # [B, L]
        # Sum over tokens → per-example loss
        loss_per_example = loss_per_token.sum(dim=1)         # [B]

        # Weight by rewards and average
        weighted_loss = (loss_per_example * rewards).mean()  # scalar

        return (weighted_loss, outputs) if return_outputs else weighted_loss

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model.config.pad_token_id = tokenizer.eos_token_id

    # 2) Build encoder
    sample_vec   = StudentState().to_vector()
    state_dim    = len(sample_vec)
    prefix_len   = 10
    encoder      = StateEncoder(state_dim, prefix_len, model.config.hidden_size)
    encoder.load_state_dict(torch.load("ft-gpt-prefix/state_encoder.pt", map_location=device))
    encoder.to(device)

    # 3) Dataset & collator
    train_ds     = RewardPromptDataset(DATA_PATH)
    collate_fn   = make_collate_fn(tokenizer, model, encoder, prefix_len, device)

    # 4) TrainingArguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        logging_steps=50,
        save_steps=200,
        report_to="none",
    )

    # 5) Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collate_fn,
        tokenizer=None,
    )

    # 6) Train & save
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
