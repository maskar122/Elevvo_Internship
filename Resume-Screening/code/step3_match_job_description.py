import pandas as pd
from sentence_transformers import SentenceTransformer, util

# ====== مسار ملف السير الذاتية بعد التنظيف ======
DATA_PATH = r"C:\Users\LAP-STORE\Desktop\Amit\NLP_Intern\Resume Screening\artifacts\resumes_cleaned.csv"
# ==================================================

# تحميل البيانات
df = pd.read_csv(DATA_PATH)

# تحميل النموذج (خفيف وسريع ودقيق)
model = SentenceTransformer('all-MiniLM-L6-v2')

# نحصل على نصوص السير الذاتية
texts = df['clean_text'].tolist()

print("🔄 Generating embeddings for resumes... (قد يستغرق دقيقة أو أكثر حسب عدد الملفات)")

# تحويل كل Resume إلى embedding
embeddings = model.encode(texts, convert_to_tensor=True)

# مثال على وصف وظيفة (ممكن تغيّره حسب الوظيفة المطلوبة)
job_description = """
We are looking for an experienced Financial Accountant who can manage budgeting,
financial reporting, reconciliation, and auditing processes.
Knowledge of GAAP principles and experience with ERP systems is preferred.
"""

# إنشاء embedding لوصف الوظيفة
job_embedding = model.encode(job_description, convert_to_tensor=True)

# حساب تشابه الكوزاين بين وصف الوظيفة والسير الذاتية
cosine_scores = util.cos_sim(job_embedding, embeddings)[0]

# ترتيب النتائج تنازليًا
top_results = sorted(list(enumerate(cosine_scores)), key=lambda x: x[1], reverse=True)[:5]

print("\n🏆 Top 5 Matching Resumes:")
for idx, score in top_results:
    print(f"File: {df.iloc[idx]['filename']} | Category: {df.iloc[idx]['category']} | Score: {score:.4f}")
    print(f"Preview: {df.iloc[idx]['clean_text'][:200]}...")
    print("-" * 80)
