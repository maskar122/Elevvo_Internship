# load_data.py
from datasets import load_dataset
import re

def clean_text(text):
    if text is None:
        return ""
    # إزالة newlines ومساحات زائدة
    text = re.sub(r"\s+", " ", text)
    return text.strip()

print("Loading full dataset (this may take a minute)...")
dataset = load_dataset("cnn_dailymail", "3.0.0")
print("Full dataset loaded.")

print("Creating small subset and cleaning text...")
def make_clean_split(split_name, n):
    ds = dataset[split_name].select(range(n))
    def _clean(example):
        example["article"] = clean_text(example.get("article", ""))
        example["highlights"] = clean_text(example.get("highlights", ""))
        return example
    return ds.map(_clean)

small_dataset = {
    "train": make_clean_split("train", 1000),
    "validation": make_clean_split("validation", 50),
    "test": make_clean_split("test", 50)
}

print("Small subset created successfully!")
print("Train size:", len(small_dataset["train"]))
print("Validation size:", len(small_dataset["validation"]))
print("Test size:", len(small_dataset["test"]))

# احفظ للتحميل السريع بعدين
small_dataset["train"].save_to_disk("cnn_small_train")
small_dataset["validation"].save_to_disk("cnn_small_val")
small_dataset["test"].save_to_disk("cnn_small_test")

# مثال سريع
example = small_dataset["train"][0]
print("\nExample article (first 500 chars):")
print(example["article"][:500], "...")
print("\nExample highlights:")
print(example["highlights"])
