from datasets import load_from_disk
from transformers import BartForConditionalGeneration, BartTokenizer
from rouge_score import rouge_scorer

MODEL_NAME = "facebook/bart-large-cnn"

print("Loading dataset (train) from disk...")
ds = load_from_disk("cnn_small_train")
print("Loaded samples:", len(ds))

print(f"Loading model: {MODEL_NAME} ...")
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
print("Model loaded successfully!")

# إعداد ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# احنا نقيم على أول 10 أمثلة بس عشان السرعة
num_samples = 10

rouge1_list = []
rouge2_list = []
rougeL_list = []

for i in range(num_samples):
    article = ds[i]["article"]
    reference = ds[i]["highlights"]

    # توليد summary
    inputs = tokenizer(article, max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=40,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
    )
    generated = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # حساب ROUGE
    scores = scorer.score(reference, generated)
    rouge1_list.append(scores["rouge1"].fmeasure)
    rouge2_list.append(scores["rouge2"].fmeasure)
    rougeL_list.append(scores["rougeL"].fmeasure)

    print(f"\nSample {i+1} ROUGE scores:")
    print("ROUGE-1:", scores["rouge1"].fmeasure)
    print("ROUGE-2:", scores["rouge2"].fmeasure)
    print("ROUGE-L:", scores["rougeL"].fmeasure)

# المتوسط
def avg(x): return sum(x)/len(x) if len(x)>0 else 0.0

print("\n" + "="*60)
print(f"Average ROUGE scores on {num_samples} samples:")
print("ROUGE-1:", avg(rouge1_list))
print("ROUGE-2:", avg(rouge2_list))
print("ROUGE-L:", avg(rougeL_list))
print("="*60)
