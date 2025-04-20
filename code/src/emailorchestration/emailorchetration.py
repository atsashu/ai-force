from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import torch

# === 1. Create dummy email dataset ===
data = {
    "prompt": [
        "Write a formal email declining a meeting.",
        "Write a follow-up email after a networking event.",
    ],
    "response": [
        "Dear [Name], Thank you for the invitation, but I won’t be able to attend...",
        "Hi [Name], It was a pleasure meeting you at [event]. I’d love to stay in touch...",
    ]
}
dataset = Dataset.from_dict(data)

# === 2. Choose a lightweight model for CPU ===
model_id = "distilgpt2"  # Much smaller than Mistral

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # Prevent tokenizer padding errors

model = AutoModelForCausalLM.from_pretrained(model_id)

# === 3. Format the dataset for fine-tuning ===
def format_email(example):
    input_text = f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['response']}"
    return tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=256
    )

tokenized_dataset = dataset.map(format_email)

# === 4. Set training arguments ===
training_args = TrainingArguments(
    output_dir="./email-gpt2",
    per_device_train_batch_size=1,
    num_train_epochs=5,
    logging_steps=1,
    save_steps=5,
    save_total_limit=1,
    report_to="none",
    no_cuda=True,  # <== This forces CPU usage
    fp16=False      # <== Don’t use mixed precision on CPU
)

# === 5. Trainer setup ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# === 6. Train! ===
trainer.train()

# === 7. Generate a sample email ===
#prompt = "Write a friendly email to a colleague ashutosh about a meeting reschedule."
#input_ids = tokenizer(f"### Instruction:\n{prompt}\n\n### Response:\n", return_tensors="pt").input_ids

prompt = "Write a follow-up email to Rishaan after a networking event."
input_ids = tokenizer(f"### Instruction:\n{prompt}\n\n### Response:\n", return_tensors="pt").input_ids
# Generate email on CPU
output = model.generate( input_ids,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2,
    do_sample=True,
    num_return_sequences=1)
print(tokenizer.decode(output[0], skip_special_tokens=True))
