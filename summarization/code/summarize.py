# summarize.py
from datasets import load_from_disk
from transformers import BartForConditionalGeneration, BartTokenizer

MODEL_NAME = "facebook/bart-large-cnn"

print("Loading tokenized dataset (train)...")
ds = load_from_disk("cnn_small_train")   # نستخدم النسخة الأصلية عشان ناخد الـ article فقط
print("Loaded samples:", len(ds))

print(f"Loading model: {MODEL_NAME} ...")
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

def generate_summary(text):
    inputs = tokenizer(
        text,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )
    
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=150,
        min_length=40,
        early_stopping=True,
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# تجربة على أول 3 مقالات
num_samples = 3

for i in range(num_samples):
    article = ds[i]["article"]
    reference = ds[i]["highlights"]
    generated = generate_summary(article)

    print("\n" + "="*80)
    print(f"Example {i+1}")
    print("\nARTICLE (first 400 chars):")
    print(article[:400], "...")
    print("\nREFERENCE SUMMARY:")
    print(reference)
    print("\nGENERATED SUMMARY:")
    print(generated)
    print("="*80)
