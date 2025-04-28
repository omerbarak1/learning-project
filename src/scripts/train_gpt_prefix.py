# src/scripts/train_gpt_prefix.py

import os
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from src.student_state import StudentState
from src.utils.state_vector import state_to_tensor
from src.utils.state_encoder import StateEncoder
from src.utils.flatten import flatten_dict

# Which JSONL to load
DATA_FILE_NAME = "synthetic_state_prompt_ext.jsonl"  # or "synthetic_state_prompt_ext.jsonl"

class PrefixPromptDataset(Dataset):
    def __init__(self, jsonl_path):
        self.examples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'state' not in ex or 'prompt' not in ex:
                    continue
                self.examples.append(ex)
        print(f"✅ Loaded {len(self.examples)} valid examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # ← Add this debug print:
        return ex


def make_collate_fn(tokenizer, model, encoder, prefix_len, device):
    hidden_size = model.config.hidden_size

    def collate_fn(batch):
        all_embeds, all_masks, all_labels = [], [], []

        for ex in batch:
            # state → vector → prefix embeddings
            sv = torch.tensor(
                state_to_tensor(ex['state']),
                dtype=torch.float32, device=device
            ).unsqueeze(0)  # [1, D]
            prefix_emb = encoder(sv)[0]  # [P, H]

            # prompt → ids → embeddings
            prompt_ids = tokenizer(
                ex['prompt'], add_special_tokens=False
            ).input_ids
            text_ids = torch.tensor([prompt_ids], device=device)
            text_emb = model.transformer.wte(text_ids)[0]  # [T, H]

            # concat
            emb = torch.cat([prefix_emb, text_emb], dim=0)  # [P+T, H]
            mask = torch.ones(emb.size(0), dtype=torch.long, device=device)
            labels = torch.tensor(
                [-100] * prefix_len + prompt_ids,
                dtype=torch.long,
                device=device
            )

            all_embeds.append(emb)
            all_masks.append(mask)
            all_labels.append(labels)

        # pad batch to max length
        max_len = max(e.size(0) for e in all_embeds)
        batch_inputs = {'inputs_embeds': [], 'attention_mask': [], 'labels': []}

        for emb, mask, labels in zip(all_embeds, all_masks, all_labels):
            pad = max_len - emb.size(0)
            batch_inputs['inputs_embeds'].append(
                torch.cat([emb, torch.zeros((pad, hidden_size), device=device)], dim=0)
            )
            batch_inputs['attention_mask'].append(
                torch.cat([mask, torch.zeros(pad, dtype=mask.dtype, device=device)], dim=0)
            )
            batch_inputs['labels'].append(
                torch.cat([labels, torch.full((pad,), -100, dtype=labels.dtype, device=device)], dim=0)
            )

        return {
            'inputs_embeds':   torch.stack(batch_inputs['inputs_embeds']),
            'attention_mask':  torch.stack(batch_inputs['attention_mask']),
            'labels':          torch.stack(batch_inputs['labels']),
        }

    return collate_fn



def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Correct relative path: project/src/scripts -> project/src/data
    script_dir = os.path.dirname(__file__)
    data_path = os.path.abspath(
        os.path.join(script_dir, os.pardir, 'data', DATA_FILE_NAME)
    )
    # ————————————————
# quick scan for lines that are literally "{}"
    empties = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            if line.strip() == '{}':
                empties.append(i)
    if empties:
        print(f"⚠️  Found completely empty JSON objects on lines: {empties}")
# ————————————————

    print(f'Loading data from: {data_path}')
    assert os.path.isfile(data_path), f'Data file not found at {data_path}'
    # model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
    model.config.pad_token_id = tokenizer.eos_token_id
    # encoder
    sample = StudentState()
    state_dim = len(sample.to_vector())
    prefix_len = 10
    encoder = StateEncoder(state_dim, prefix_len, model.config.hidden_size).to(device)
    # dataset
    train_ds = PrefixPromptDataset(data_path)
    print("🕵️ Sample dataset entries:")
    for i in range(3):
        ex = train_ds[i]
        print(f"  example {i} keys = {list(ex.keys())}")

    collate = make_collate_fn(tokenizer, model, encoder, prefix_len, device)
    # training args
    training_args = TrainingArguments(
        output_dir='ft-gpt-prefix',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        logging_steps=100,
        save_steps=500,
        report_to='none',
        remove_unused_columns=False,      # << here
        dataloader_pin_memory=False,
)

    # trainer
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    data_collator=collate,
    tokenizer=tokenizer,             # give it your tokenizer
     # ← keep “state” & “prompt” intact
)

    from torch.utils.data import DataLoader

    print("🔧 Sanity-checking collate_fn with manual DataLoader…")
    dl = DataLoader(
        train_ds,
        batch_size=4,
        collate_fn=collate,
        num_workers=0,          # single-process so we see all prints
        shuffle=False           # fixed batch for reproducibility
    )
    batch = next(iter(dl))
    print("✅ Manual batch keys & shapes:", {k: v.shape for k,v in batch.items()})

    # train
    trainer.train()
    model.save_pretrained('ft-gpt-prefix')
    tokenizer.save_pretrained('ft-gpt-prefix')
    torch.save(encoder.state_dict(), 'ft-gpt-prefix/state_encoder.pt')

if __name__ == '__main__':
    main()

