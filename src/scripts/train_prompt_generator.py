from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from src.utils.flatten import flatten_dict

# 1) Load your JSONL (no need to fiddle with .json/.jsonl)
ds = load_dataset(
    "json",
    data_files={"train": "src/data/synthetic_state_prompt.jsonl"},
)[ "train" ]

# 2) Prepare tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 3) Helper to turn nested state dict → flat string
def serialize_state(s: dict) -> str:
    flat = flatten_dict(s)          # e.g. {"mastery.arithmetic.fractions": 0.0, ...}
    parts = [f"{k}={v}" for k, v in flat.items()]
    return "state: " + ", ".join(parts) + "\n"

# 4) Map each example to input_ids + labels
def tokenize(example):
    prefix = serialize_state(example["state"])
    target = example["prompt"]
    full = prefix + target
    # By default Trainer will use full sequence as labels (i.e. teacher-forcing)
    return tokenizer(full, truncation=True, max_length=512)

# 5) Apply the map (one example at a time for simplicity)
tok_ds = ds.map(
    tokenize,
    batched=False,
    remove_columns=["state", "prompt"],
)

# 6) Set up collator & model
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.config.pad_token_id = tokenizer.eos_token_id

# 7) Training arguments
training_args = TrainingArguments(
    output_dir="ft-prompt-agent",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_steps=50,
    save_steps=200,
    report_to="none",
)

# 8) Trainer & train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tok_ds,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("ft-prompt-agent")
tokenizer.save_pretrained("ft-prompt-agent")
