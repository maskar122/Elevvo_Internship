import os
from pathlib import Path
import fitz  # PyMuPDF
import pdfplumber
import chardet
from docx import Document
import pandas as pd
from tqdm import tqdm

# ======= عدّل المسار ده حسب جهازك =======
DATA_DIR = Path(r"C:\Users\LAP-STORE\Desktop\Amit\NLP_Intern\Resume Screening\data\data")
OUT_PARQUET = Path(r"C:\Users\LAP-STORE\Desktop\Amit\NLP_Intern\Resume Screening\artifacts\resumes_parsed.parquet")
OUT_CSV = Path(r"C:\Users\LAP-STORE\Desktop\Amit\NLP_Intern\Resume Screening\artifacts\resumes_parsed.csv")
# =======================================

OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

def extract_text_pdf(path: Path) -> str:
    """جرّب PyMuPDF أولًا؛ لو فشل جرّب pdfplumber كـ باك أب."""
    text = ""
    try:
        with fitz.open(path) as doc:
            for page in doc:
                text += page.get_text() or ""
    except Exception:
        # fallback: pdfplumber
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except Exception:
            pass
    return (text or "").strip()

def extract_text_txt(path: Path) -> str:
    """قراءة txt مع كشف الترميز."""
    raw = path.read_bytes()
    enc = chardet.detect(raw).get("encoding") or "utf-8"
    try:
        return raw.decode(enc, errors="ignore")
    except Exception:
        return raw.decode("utf-8", errors="ignore")

def extract_text_docx(path: Path) -> str:
    try:
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""

def extract_text_generic(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_text_pdf(path)
    elif ext == ".txt":
        return extract_text_txt(path)
    elif ext == ".docx":
        return extract_text_docx(path)
    else:
        return ""  # تجاهل باقي الامتدادات في المرحلة دي

def walk_dataset(root: Path):
    """يمرّ على كل الفولدرات (كل فولدر = category) ويجمع الملفات."""
    rows = []
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")
    categories = [p for p in root.iterdir() if p.is_dir()]
    for cat_dir in tqdm(categories, desc="Categories"):
        category = cat_dir.name
        files = list(cat_dir.glob("**/*"))
        for f in tqdm(files, desc=f"{category}", leave=False):
            if f.suffix.lower() in {".pdf", ".txt", ".docx"}:
                text = extract_text_generic(f)
                rows.append(
                    {
                        "category": category,
                        "filename": f.name,
                        "path": str(f),
                        "text": text,
                        "n_chars": len(text),
                    }
                )
    return pd.DataFrame(rows)

def main():
    df = walk_dataset(DATA_DIR)

    # تنظيف بسيط: شيل الملفات اللي ماطلعش منها نص
    df = df[df["n_chars"] > 30].reset_index(drop=True)

    # احفظ نسختين (parquet أسرع للتحميل، و csv للمراجعة اليدوية)
    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    # تقرير سريع
    by_cat = df.groupby("category")["filename"].count().sort_values(ascending=False)
    print("\nRésumé count by category:")
    print(by_cat.head(20))
    print(f"\nSaved: {OUT_CSV}")
    print(f"Saved: {OUT_PARQUET}")

if __name__ == "__main__":
    main()
