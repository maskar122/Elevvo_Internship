import pandas as pd
import re
import string
from nltk.corpus import stopwords
import nltk

# تحميل stopwords لأول مرة
nltk.download('stopwords')

# ======= المسار إلى ملف الـ CSV الناتج من الخطوة السابقة =======
DATA_PATH = r"C:\Users\LAP-STORE\Desktop\Amit\NLP_Intern\Resume Screening\artifacts\resumes_parsed.csv"
OUT_PATH = r"C:\Users\LAP-STORE\Desktop\Amit\NLP_Intern\Resume Screening\artifacts\resumes_cleaned.csv"
# ===============================================================

# تحميل البيانات
df = pd.read_csv(DATA_PATH)

# تجهيز stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """تنظيف النص من الرموز والأرقام والكلمات التافهة"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # إزالة المسافات الزائدة
    text = re.sub(r'http\S+|www\S+', '', text)  # إزالة الروابط
    text = text.translate(str.maketrans('', '', string.punctuation))  # إزالة الرموز
    text = re.sub(r'\d+', '', text)  # إزالة الأرقام
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

# تطبيق التنظيف
df['clean_text'] = df['text'].apply(clean_text)

# حذف الصفوف اللي النص فيها صغير جدًا بعد التنظيف
df = df[df['clean_text'].str.len() > 50].reset_index(drop=True)

# حفظ النتيجة
df.to_csv(OUT_PATH, index=False, encoding='utf-8')

print(f"✅ Cleaned resumes saved to: {OUT_PATH}")
print(df[['category', 'filename', 'clean_text']].head(3))
