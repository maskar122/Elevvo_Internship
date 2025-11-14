# preprocess_tokenize.py
from datasets import load_from_disk
from transformers import AutoTokenizer

MODEL_NAME = "facebook/bart-large-cnn"
MAX_INPUT_LENGTH = 1024

print("Loading cleaned dataset (train) from disk...")
ds = load_from_disk("cnn_small_train")
print("Loaded train samples:", len(ds))

print(f"Loading tokenizer for {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_batch(batch):
    encoding = tokenizer(
        batch["article"],
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        padding="max_length",
    )
    return encoding

print("Tokenizing dataset...")
tokenized_ds = ds.map(tokenize_batch, batched=True, batch_size=32)

# حفظ النسخة المجهزة
tokenized_ds.save_to_disk("cnn_tokenized_train")

print("Tokenization complete!")
print("Saved tokenized dataset to cnn_tokenized_train")
