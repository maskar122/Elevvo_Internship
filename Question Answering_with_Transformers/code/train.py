import os
import torch
from datasets import load_from_disk
from transformers import (
    DistilBertForQuestionAnswering,
    DistilBertTokenizerFast,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

def main():
    # مسارات البيانات والموديل
    train_path = r"C:\Users\LAP-STORE\Desktop\Amit\NLP_Intern\Question Answering with Transformers\data\processed\train"
    val_path = r"C:\Users\LAP-STORE\Desktop\Amit\NLP_Intern\Question Answering with Transformers\data\processed\validation"
    model_name = "distilbert-base-uncased"
    out_dir = "models/qa_model"

    os.makedirs(out_dir, exist_ok=True)

    print("🔹 Loading tokenized datasets ...")
    train_dataset = load_from_disk(train_path)
    val_dataset = load_from_disk(val_path)

    print("🔹 Loading model and tokenizer ...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = DistilBertForQuestionAnswering.from_pretrained(model_name)

    # تحديد الأعمدة التي يستخدمها Trainer
    columns = ["input_ids", "attention_mask", "start_positions", "end_positions"]
    train_dataset.set_format(type="torch", columns=columns)
    val_dataset.set_format(type="torch", columns=columns)

    print("🔹 Setting up TrainingArguments ...")
    training_args = TrainingArguments(
        output_dir=out_dir,
        eval_strategy="epoch",              # ← الاسم الجديد في Transformers 4.57
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir=f"{out_dir}/logs",
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",                  # يعطل Weights & Biases logs
    )

    print("🔹 Creating Trainer ...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    if torch.cuda.is_available():
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ GPU not found, training will be slower (using CPU).")

    print("🚀 Starting training ...")
    trainer.train()

    print("💾 Saving final model ...")
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"✅ Training complete! Model saved to: {out_dir}")


if __name__ == "__main__":
    main()
