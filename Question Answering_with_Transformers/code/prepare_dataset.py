import argparse
import json
import os
from typing import List, Dict

from datasets import Dataset, DatasetDict
from transformers import DistilBertTokenizerFast


def parse_squad(path: str) -> List[Dict]:
    """
    يحوّل ملف SQuAD v1.1 (بنفس هيكلة ستانفورد) إلى قائمة أمثلة مسطحة.
    كل مثال: {id, context, question, answer_text, answer_start}
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    examples = []
    for article in raw["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                qid = qa.get("id", "")
                # SQuAD v1.1 عادةً فيها إجابة واحدة (لكن أحيانًا أكثر من واحدة)
                # هنا نأخذ أول إجابة فقط (كما هو شائع للتدريب البسيط)
                ans = qa["answers"][0]
                examples.append(
                    {
                        "id": qid,
                        "context": context,
                        "question": qa["question"],
                        "answer_text": ans["text"],
                        "answer_start": ans["answer_start"],
                    }
                )
    return examples


def build_hf_dataset(train_path: str, dev_path: str) -> DatasetDict:
    train_examples = parse_squad(train_path)
    dev_examples = parse_squad(dev_path)

    train_ds = Dataset.from_list(train_examples)
    dev_ds = Dataset.from_list(dev_examples)

    return DatasetDict({"train": train_ds, "validation": dev_ds})


def prepare_features(tokenizer: DistilBertTokenizerFast, max_length=384, doc_stride=128):
    """
    تُرجع دالة map لمعالجة الداتا إلى ميزات تدريب جاهزة:
    - ترميز (question, context)
    - التعامل مع الفقرات الطويلة عبر overflow + doc_stride
    - حساب start_positions / end_positions بالاعتماد على offset_mapping
    """
    def _fn(examples):
        questions = [q.strip() for q in examples["question"]]
        contexts = examples["context"]
        answers = examples["answer_text"]
        answer_starts = examples["answer_start"]

        tokenized = tokenizer(
            questions,
            contexts,
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # نحتاج ربط كل feature بالمثال الأصلي (لأن overflow ينسخ المثال لعدة شرائح)
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            # هذا الـ feature يعود لأي عينة أصلًا؟
            sample_idx = sample_mapping[i]
            answer_start_char = answer_starts[sample_idx]
            answer_text = answers[sample_idx]
            answer_end_char = answer_start_char + len(answer_text)

            # نحدد أي توكنات تخص الـ context (sequence_ids: 0=>question, 1=>context, None=>special)
            sequence_ids = tokenized.sequence_ids(i)

            # حدّد حدود الـ context داخل السلسلة المرمّزة
            # أول token للـ context:
            context_start = 0
            while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
                context_start += 1
            # آخر token للـ context:
            context_end = len(sequence_ids) - 1
            while context_end >= 0 and sequence_ids[context_end] != 1:
                context_end -= 1

            # لو لم نجد سياق فعلي (نادرًا)، عالج كـ CLS (0,0)
            if context_start > context_end:
                start_positions.append(0)
                end_positions.append(0)
                continue

            # لو كانت الإجابة خارج هذه الشريحة (بسبب القصّ)، ضعها على CLS
            if not (offsets[context_start][0] <= answer_start_char and
                    offsets[context_end][1] >= answer_end_char):
                start_positions.append(0)
                end_positions.append(0)
                continue

            # تحريك start_positions إلى أول token يغطي بداية الإجابة
            start_token = context_start
            while start_token <= context_end and offsets[start_token][0] <= answer_start_char:
                if offsets[start_token][1] > answer_start_char:
                    break
                start_token += 1

            # تحريك end_positions إلى آخر token يغطي نهاية الإجابة
            end_token = context_end
            while end_token >= context_start and offsets[end_token][1] >= answer_end_char:
                if offsets[end_token][0] < answer_end_char:
                    break
                end_token -= 1

            # أحيانًا اللوجيك أعلاه يحتاج ضبط لمواضع التماس — بديل آمن:
            # ابحث عن أول/آخر توكن يغطي أي جزء من الإجابة:
            # (لو تحب يمكن استبدال الكتلة أعلاه بهذا النهج)
            if start_token > context_end or end_token < context_start:
                # fallback
                # العثور على أول توكن يغطي بداية الإجابة
                start_token = context_start
                while start_token <= context_end and offsets[start_token][0] <= answer_start_char:
                    start_token += 1
                start_token = max(context_start, start_token - 1)

                # العثور على آخر توكن يغطي نهاية الإجابة
                end_token = context_end
                while end_token >= context_start and offsets[end_token][1] >= answer_end_char:
                    end_token -= 1
                end_token = min(context_end, end_token + 1)

            start_positions.append(start_token)
            end_positions.append(end_token)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        return tokenized

    return _fn


def main():
    parser = argparse.ArgumentParser(description="Prepare SQuAD v1.1 for QA fine-tuning")
    parser.add_argument("--train_path", type=str, default="train-v1.1.json", help="Path to train-v1.1.json")
    parser.add_argument("--dev_path", type=str, default="dev-v1.1.json", help="Path to dev-v1.1.json")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Tokenizer model name")
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Output directory for saved datasets")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("🔹 Loading raw SQuAD json ...")
    ds = build_hf_dataset(args.train_path, args.dev_path)

    print("🔹 Loading tokenizer:", args.model_name)
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name, use_fast=True)

    print("🔹 Tokenizing with sliding window (doc_stride) ...")
    features_fn = prepare_features(
        tokenizer,
        max_length=args.max_length,
        doc_stride=args.doc_stride
    )

    tokenized = ds.map(
        features_fn,
        batched=True,
        remove_columns=ds["train"].column_names,  # نحذف الأعمدة الخام ونُبقي الميزات فقط
        desc="Tokenizing"
    )

    # حفظ النسخة المعالجة
    train_out = os.path.join(args.out_dir, "train")
    val_out = os.path.join(args.out_dir, "validation")

    print(f"💾 Saving tokenized train to: {train_out}")
    tokenized["train"].save_to_disk(train_out)

    print(f"💾 Saving tokenized validation to: {val_out}")
    tokenized["validation"].save_to_disk(val_out)

    print("✅ Done! Datasets are ready for training.")


if __name__ == "__main__":
    main()
