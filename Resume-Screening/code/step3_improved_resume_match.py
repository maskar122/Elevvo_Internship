import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util

# ===== مسار ملف السير الذاتية المنظّف =====
DATA_PATH = r"C:\Users\LAP-STORE\Desktop\Amit\NLP_Intern\Resume Screening\artifacts\resumes_cleaned.csv"
# ============================================

# تحميل البيانات
df = pd.read_csv(DATA_PATH)

# تحميل نموذج أقوى (أدق من all-MiniLM)
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# وصف الوظيفة - عدّل حسب المطلوب
job_description = """
Job Role: Financial Accountant
Responsibilities: financial reporting, budgeting, and auditing.
Skills: accounting, GAAP, Excel, SAP, ERP systems, financial analysis, reconciliation, cost control.
Experience: 3-5 years of experience in corporate finance.
Education: Bachelor's degree in Accounting or Finance.
"""

# تجهيز الـ embedding لوصف الوظيفة
job_embedding = model.encode(job_description, convert_to_tensor=True)

def split_into_chunks(text, words_per_chunk=200):
    """تقسيم النص إلى مقاطع صغيرة لرفع دقة التشابه"""
    words = text.split()
    return [" ".join(words[i:i+words_per_chunk]) for i in range(0, len(words), words_per_chunk)]

results = []

print("🔄 Generating embeddings and calculating similarity...")

for idx, row in df.iterrows():
    text = str(row['clean_text'])
    chunks = split_into_chunks(text)

    # تحويل كل chunk إلى embedding
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

    # حساب التشابه لكل مقطع وأخذ الأعلى
    cosine_scores = util.cos_sim(job_embedding, chunk_embeddings)[0]
    best_score = float(cosine_scores.max())

    results.append({
        'filename': row['filename'],
        'category': row['category'],
        'best_score': best_score,
    })

# تحويل النتائج إلى DataFrame
res_df = pd.DataFrame(results)
res_df = res_df.sort_values(by='best_score', ascending=False).reset_index(drop=True)

# عرض أعلى 5
print("\n🏆 Top 5 Matching Resumes:")
for i in range(5):
    print(f"{i+1}. File: {res_df.iloc[i]['filename']} | Category: {res_df.iloc[i]['category']} | Score: {res_df.iloc[i]['best_score']:.4f}")

# حفظ النتائج في CSV
res_df.to_csv(r"C:\Users\LAP-STORE\Desktop\Amit\NLP_Intern\Resume Screening\artifacts\resume_similarity_scores_final.csv", index=False)
print("\n✅ Results saved to artifacts/resume_similarity_scores.csv")

# ======================================================
# 🧠 حفظ الـ embeddings علشان تستخدمها في Streamlit
# ======================================================

print("\n💾 Generating and saving resume embeddings for future use...")

# حساب embeddings لكل السير الذاتية مرة واحدة فقط
all_texts = df["clean_text"].tolist()
resume_embeddings = model.encode(all_texts, batch_size=16, show_progress_bar=True)

# حفظ embeddings + filenames + categories في ملف واحد
np.savez(
    r"C:\Users\LAP-STORE\Desktop\Amit\NLP_Intern\Resume Screening\artifacts\resume_embeddings.npz",
    embeddings=resume_embeddings,
    filenames=df["filename"].tolist(),
    categories=df["category"].tolist()
)

print("✅ Embeddings saved successfully to artifacts/resume_embeddings.npz")
