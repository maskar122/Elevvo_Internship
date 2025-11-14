import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
from rouge_score import rouge_scorer
import PyPDF2
import io

# Page setup
st.set_page_config(
    page_title="Text Summarization Pro",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model
MODEL_NAME = "facebook/bart-large-cnn"

@st.cache_resource
def load_model():
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    return tokenizer, model

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .summary-box {
        background-color: #f0f7ff;
        border-left: 5px solid #1f77b4;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
    .upload-box {
        border: 2px dashed #1f77b4;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .css-1d391kg, .css-12oz5g7 {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
    st.title("Text Summarization Pro")
    st.markdown("---")
    st.markdown("### About the App")
    st.markdown("""
    This application uses advanced BART model for automatic text summarization
    with the ability to evaluate summary quality using ROUGE metrics.
    """)
    
    st.markdown("### Instructions")
    st.markdown("""
    1. Enter text directly or upload PDF file
    2. Enter reference summary (optional) for evaluation
    3. Click 'Generate Summary' button
    4. Review results and quality metrics
    """)
    
    st.markdown("### Technical Information")
    st.markdown(f"""
    - Model: {MODEL_NAME}
    - Maximum text length: 1024 tokens
    - Maximum summary length: 150 tokens
    - Minimum summary length: 40 tokens
    """)

# Main content
st.markdown('<p class="main-header">📝 Text Summarization Pro</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #555;">Use AI to summarize texts and evaluate summary quality</p>', unsafe_allow_html=True)

# Load model with progress indicator
with st.spinner('Loading AI model...'):
    tokenizer, model = load_model()
st.success('Model loaded successfully!')

# Split page into columns
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<p class="sub-header">📄 Input Text</p>', unsafe_allow_html=True)
    
    # PDF upload
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF file (optional)", type="pdf", help="You can upload a PDF file to automatically extract text from it")
    st.markdown('</div>', unsafe_allow_html=True)
    
    article_text = ""
    if uploaded_file is not None:
        with st.spinner('Extracting text from PDF file...'):
            try:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    article_text += page.extract_text() + "\n"
                st.success(f'Text extracted successfully! Number of pages: {len(pdf_reader.pages)}')
            except Exception as e:
                st.error(f'Error reading PDF file: {str(e)}')
    
    # Text input
    article = st.text_area(
        "Enter the text you want to summarize:",
        value=article_text,
        height=300,
        placeholder="Paste text here or use PDF file above...",
        help="Text should be in English for best results"
    )
    
    # Reference summary input
    st.markdown('<p class="sub-header">📋 Reference Summary (Optional)</p>', unsafe_allow_html=True)
    reference = st.text_area(
        "Enter reference summary for quality comparison:",
        height=150,
        placeholder="Enter reference summary here to compare with generated summary...",
        help="This field is optional, used only to evaluate the quality of generated summary"
    )

with col2:
    st.markdown('<p class="sub-header">⚙️ Settings</p>', unsafe_allow_html=True)
    
    # Model settings
    max_length = st.slider(
        "Maximum summary length:",
        min_value=50,
        max_value=200,
        value=150,
        step=10,
        help="Maximum number of words in summary"
    )
    
    min_length = st.slider(
        "Minimum summary length:",
        min_value=20,
        max_value=100,
        value=40,
        step=5,
        help="Minimum number of words in summary"
    )
    
    num_beams = st.selectbox(
        "Search strategy:",
        options=[2, 4, 6],
        index=1,
        help="Number of beams used in text generation (higher values give better results but slower)"
    )
    
    # Generate summary button
    generate_btn = st.button(
        "Generate Summary",
        key="generate",
        use_container_width=True,
        type="primary"
    )

# Text processing and summary generation
if generate_btn:
    if article.strip() == "":
        st.warning("⚠️ Please enter some text to summarize.")
    else:
        with st.spinner('Processing text and generating summary...'):
            # Progress bar
            progress_bar = st.progress(0)
            for i in range(100):
                # Simulate progress
                pass
            progress_bar.progress(100)
            
            # Generate summary
            inputs = tokenizer(article, max_length=1024, truncation=True, return_tensors="pt")
            summary_ids = model.generate(
                inputs["input_ids"],
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                early_stopping=True
            )
            generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Display results
        st.markdown("---")
        st.markdown('<p class="sub-header">📊 Results</p>', unsafe_allow_html=True)
        
        # Split results into columns
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.markdown("#### Generated Summary")
            st.markdown(f'<div class="summary-box">{generated_summary}</div>', unsafe_allow_html=True)
            
            # Text statistics
            st.markdown("#### Text Statistics")
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Original Text Words", len(article.split()))
            with col_stat2:
                st.metric("Summary Words", len(generated_summary.split()))
            with col_stat3:
                reduction = ((len(article.split()) - len(generated_summary.split())) / len(article.split())) * 100
                st.metric("Reduction Rate", f"{reduction:.1f}%")
        
        with res_col2:
            # ROUGE evaluation if reference is provided
            if reference.strip() != "":
                st.markdown("#### Quality Evaluation (ROUGE)")
                scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                scores = scorer.score(reference, generated_summary)
                
                # Display metrics in boxes
                col_rouge1, col_rouge2, col_rougeL = st.columns(3)
                
                with col_rouge1:
                    st.markdown(f'<div class="metric-box">'
                               f'<h4>ROUGE-1</h4>'
                               f'<h3>{scores["rouge1"].fmeasure:.3f}</h3>'
                               f'<p>Word-level matching</p>'
                               f'</div>', unsafe_allow_html=True)
                
                with col_rouge2:
                    st.markdown(f'<div class="metric-box">'
                               f'<h4>ROUGE-2</h4>'
                               f'<h3>{scores["rouge2"].fmeasure:.3f}</h3>'
                               f'<p>Word-pairs matching</p>'
                               f'</div>', unsafe_allow_html=True)
                
                with col_rougeL:
                    st.markdown(f'<div class="metric-box">'
                               f'<h4>ROUGE-L</h4>'
                               f'<h3>{scores["rougeL"].fmeasure:.3f}</h3>'
                               f'<p>Longest common sequence matching</p>'
                               f'</div>', unsafe_allow_html=True)
                
                # Results interpretation
                st.markdown("#### Results Interpretation")
                avg_score = (scores["rouge1"].fmeasure + scores["rouge2"].fmeasure + scores["rougeL"].fmeasure) / 3
                if avg_score >= 0.7:
                    st.success("✅ Excellent summary quality - High similarity with reference summary")
                elif avg_score >= 0.5:
                    st.info("ℹ️ Good summary quality - Medium similarity with reference summary")
                elif avg_score >= 0.3:
                    st.warning("⚠️ Acceptable summary quality - Low similarity with reference summary")
                else:
                    st.error("❌ Low summary quality - Weak similarity with reference summary")
            else:
                st.info("ℹ️ No reference summary entered. Add a reference summary to get quality evaluation.")

# Page footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #777;'>Text Summarization Pro - Developed with BART and Streamlit</p>", 
    unsafe_allow_html=True
)