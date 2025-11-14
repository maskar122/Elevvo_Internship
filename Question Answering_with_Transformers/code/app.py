import streamlit as st
from transformers import pipeline
import re
import pandas as pd
import PyPDF2
import docx
import io
import base64
from typing import List, Dict

# Page configuration
st.set_page_config(
    page_title="AI Question Answering System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model
@st.cache_resource
def load_model():
    qa = pipeline(
        "question-answering",
        model="models/qa_model",
        tokenizer="models/qa_model"
    )
    return qa

qa_pipeline = load_model()

# File processing functions
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

def extract_text_from_excel(file):
    df = pd.read_excel(file)
    text = ""
    for column in df.columns:
        text += f"{column}: " + " ".join(df[column].dropna().astype(str)) + "\n"
    return text

# Custom CSS with enhanced visual design
st.markdown("""
<style>
    /* Main Styles */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 800;
        font-family: 'Inter', sans-serif;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    /* Card Styles */
    .card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
    }
    
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem;
    }
    
    /* Answer Box */
    .answer-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .answer-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 70%);
    }
    
    /* File Upload */
    .file-upload-box {
        border: 3px dashed #667eea;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .file-upload-box:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #f0f2ff 0%, #e8ebff 100%);
    }
    
    /* Progress and Metrics */
    .metric-box {
        background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .confidence-bar {
        background: linear-gradient(90deg, #ff6b6b 0%, #ffd93d 50%, #6bcf7f 100%);
        height: 12px;
        border-radius: 10px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        font-size: 1.2rem;
        border-radius: 12px;
        transition: all 0.3s ease;
        width: 100%;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Highlighting */
    .context-highlight {
        background: linear-gradient(135deg, #fff9e6 0%, #fff0cc 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #ffd93d;
        line-height: 1.8;
        font-size: 1.1rem;
    }
    
    mark {
        background: linear-gradient(135deg, #ffd93d 0%, #ff9d00 100%);
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        font-weight: bold;
        color: #333;
    }
    
    /* File Info */
    .file-info {
        background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    
    /* Icons and Images */
    .icon-large {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Background image with base64 encoding
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Sidebar with enhanced design
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <div style='font-size: 4rem; margin-bottom: 1rem;'>🧠</div>
        <h2 style='color: white; margin-bottom: 0.5rem;'>AI QA System</h2>
        <p style='color: rgba(255,255,255,0.8);'>Professional Document Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class='card' style='background: rgba(255,255,255,0.1); color: white; border: none;'>
        <h4 style='color: white;'>🎯 Quick Guide</h4>
        <p style='color: rgba(255,255,255,0.9);'>
        Upload documents or enter text, then ask questions to get AI-powered answers extracted directly from your content.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Supported Formats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='text-align: center; color: white;'>
            <div style='font-size: 2rem;'>📄</div>
            <small>PDF</small>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='text-align: center; color: white;'>
            <div style='font-size: 2rem;'>📊</div>
            <small>Excel</small>
        </div>
        """, unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        <div style='text-align: center; color: white;'>
            <div style='font-size: 2rem;'>📝</div>
            <small>Word</small>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div style='text-align: center; color: white;'>
            <div style='font-size: 2rem;'>📃</div>
            <small>Text</small>
        </div>
        """, unsafe_allow_html=True)

# Main header section
st.markdown("<h1 class='main-header fade-in'>🧠 Intelligent Document QA System</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header fade-in'>Transform your documents into interactive knowledge bases with AI-powered question answering</p>", unsafe_allow_html=True)

# Feature cards
st.markdown("### 🚀 Key Features")
feature_col1, feature_col2, feature_col3 = st.columns(3)

with feature_col1:
    st.markdown("""
    <div class='feature-card'>
        <div class='icon-large'>🤖</div>
        <h4>AI-Powered</h4>
        <p>Advanced NLP for accurate answers</p>
    </div>
    """, unsafe_allow_html=True)

with feature_col2:
    st.markdown("""
    <div class='feature-card'>
        <div class='icon-large'>📁</div>
        <h4>Multi-Format</h4>
        <p>Support for all major document types</p>
    </div>
    """, unsafe_allow_html=True)

with feature_col3:
    st.markdown("""
    <div class='feature-card'>
        <div class='icon-large'>⚡</div>
        <h4>Real-Time</h4>
        <p>Instant answers with confidence scoring</p>
    </div>
    """, unsafe_allow_html=True)

# Input Methods Section
st.markdown("---")
st.markdown("## 📥 Input Your Content")

input_method = st.radio(
    "Choose input method:",
    ["📝 Enter Text", "📁 Upload Documents"],
    horizontal=True,
    key="input_method"
)

context = ""
uploaded_files_info = []

if input_method == "📁 Upload Documents":
    st.markdown("""
    <div class='file-upload-box'>
        <div style='font-size: 4rem; margin-bottom: 1rem;'>📁</div>
        <h3 style='color: #667eea; margin-bottom: 1rem;'>Drag & Drop Your Files</h3>
        <p style='color: #6c757d;'>Supported formats: PDF, Excel, Word, Text files</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "docx", "xlsx", "xls"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload one or multiple documents"
    )
    
    if uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name}... ({i + 1}/{len(uploaded_files)})")
            
            file_info = {
                "name": uploaded_file.name,
                "size": uploaded_file.size,
                "type": uploaded_file.type
            }
            
            try:
                if uploaded_file.type == "application/pdf":
                    extracted_text = extract_text_from_pdf(uploaded_file)
                    file_info["status"] = "✅ Success"
                    file_info["pages"] = len(PyPDF2.PdfReader(uploaded_file).pages)
                elif uploaded_file.type == "text/plain":
                    extracted_text = extract_text_from_txt(uploaded_file)
                    file_info["status"] = "✅ Success"
                elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                                          "application/msword"]:
                    extracted_text = extract_text_from_docx(uploaded_file)
                    file_info["status"] = "✅ Success"
                elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                          "application/vnd.ms-excel"]:
                    extracted_text = extract_text_from_excel(uploaded_file)
                    file_info["status"] = "✅ Success"
                else:
                    extracted_text = ""
                    file_info["status"] = "❌ Unsupported format"
                
                context += extracted_text + "\n\n"
                uploaded_files_info.append(file_info)
                
            except Exception as e:
                file_info["status"] = f"❌ Error: {str(e)}"
                uploaded_files_info.append(file_info)
        
        progress_bar.empty()
        status_text.empty()
        
        # Display file information
        if uploaded_files_info:
            st.markdown("### 📊 Processing Summary")
            for file_info in uploaded_files_info:
                st.markdown(f"""
                <div class='file-info'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <strong>📄 {file_info['name']}</strong><br>
                            <small>Size: {file_info['size']:,} bytes • {file_info['status']}</small>
                            {f"• Pages: {file_info['pages']}" if 'pages' in file_info else ''}
                        </div>
                        <div style='font-size: 1.5rem;'>
                            { '✅' if '✅' in file_info['status'] else '❌' }
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Show extracted text preview
        if context.strip():
            st.markdown("### 📖 Extracted Content Preview")
            with st.expander("View processed content", expanded=False):
                st.text_area("", context, height=200, label_visibility="collapsed")

else:
    # Text input method
    st.markdown("""
    <div class='card'>
        <h3>📄 Enter Your Text</h3>
        <p>Paste or type your reference text below</p>
    </div>
    """, unsafe_allow_html=True)
    
    context = st.text_area(
        "Enter your reference text here", 
        height=300, 
        placeholder="Example: Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems. Specific applications include expert systems, natural language processing, speech recognition, and machine vision...",
        label_visibility="collapsed"
    )

# Question Input Section
st.markdown("---")
st.markdown("## ❓ Ask Your Question")

question_col1, question_col2 = st.columns([3, 1])

with question_col1:
    question = st.text_input(
        "Enter your question here", 
        placeholder="Example: What are the main applications of artificial intelligence?",
        label_visibility="collapsed"
    )

with question_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    get_answer = st.button("🚀 Get Answer", use_container_width=True)

# Process Results
if get_answer:
    if not context.strip() or not question.strip():
        st.error("""
        ⚠️ **Please complete all fields**
        
        • Provide reference text or upload documents
        • Enter your question
        """)
    else:
        with st.spinner("🔍 AI is analyzing your content and searching for answers..."):
            try:
                result = qa_pipeline(question=question, context=context)
                answer = result["answer"]
                score = result["score"]
                
                # Highlight answer in context
                def highlight_answer(context, answer):
                    pattern = re.escape(answer)
                    return re.sub(pattern, f"<mark>{answer}</mark>", context, flags=re.IGNORECASE)

                highlighted_context = highlight_answer(context, answer)

                # Display Results
                st.success("✅ Answer extracted successfully!")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class='metric-box'>
                        <div style='font-size: 2rem;'>📊</div>
                        <h3>{len(context.split()):,}</h3>
                        <small>Words Processed</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class='metric-box'>
                        <div style='font-size: 2rem;'>🎯</div>
                        <h3>{len(answer.split())}</h3>
                        <small>Answer Length</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    confidence_percentage = round(score * 100, 1)
                    st.markdown(f"""
                    <div class='metric-box'>
                        <div style='font-size: 2rem;'>💡</div>
                        <h3>{confidence_percentage}%</h3>
                        <small>Confidence</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Answer Box
                st.markdown("<div class='answer-box fade-in'>", unsafe_allow_html=True)
                st.markdown("### 🎯 AI Answer")
                st.markdown(f"<p style='font-size: 1.8rem; font-weight: bold; margin: 1rem 0; line-height: 1.4;'>{answer}</p>", unsafe_allow_html=True)
                
                # Confidence visualization
                st.markdown("### 📈 Confidence Level")
                confidence_width = min(100, confidence_percentage)
                st.markdown(f"""
                <div class='confidence-bar'>
                    <div style='width: {confidence_width}%; height: 100%; background: linear-gradient(90deg, #ff6b6b 0%, #ffd93d 50%, #6bcf7f 100%); border-radius: 10px;'></div>
                </div>
                <div style='text-align: center; color: white; font-weight: bold; font-size: 1.2rem;'>
                    {confidence_percentage}% Confidence
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Context with highlighted answer
                st.markdown("### 🔍 Source Context")
                st.markdown(f"<div class='context-highlight fade-in'>{highlighted_context}</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error processing your request: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; padding: 2rem 0;'>
    <div style='display: flex; justify-content: center; gap: 2rem; margin-bottom: 1rem;'>
        <div>🤖 AI Powered</div>
        <div>⚡ Real Time</div>
        <div>🔒 Secure</div>
        <div>🌐 Multi-Format</div>
    </div>
    <p>© 2024 Intelligent Document QA System | Built with Streamlit & Transformers</p>
</div>
""", unsafe_allow_html=True)