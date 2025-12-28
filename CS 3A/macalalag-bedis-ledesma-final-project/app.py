import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for clean Notion-like styling
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    /* Main container */
    .main {
        max-width: 720px;
        padding: 3rem 2rem;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global text sizing */
    * {
        font-size: 15px;
        line-height: 1.6;
    }
    
    /* Title styling */
    h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em;
        margin-bottom: 0.5rem !important;
        color: #1a1a1a;
    }
    
    /* Subtitle */
    .subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    /* Text area */
    .stTextArea textarea {
        border-radius: 4px;
        border: none;
        font-size: 0.95rem;
        padding: 0.75rem;
        transition: border-color 0.15s;
        font-family: inherit;
        background-color: #ffffff;
        color: #1a1a1a;
    }
    
    .stTextArea textarea:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 1px #2563eb;
        outline: none;
    }
    
    .stTextArea textarea::placeholder {
        color: #9ca3af;
    }
    
    /* Form submit button */
    .stFormSubmitButton button {
        background-color: #2563eb;
        color: white !important;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-size: 0.9rem;
        border: none;
        transition: background-color 0.15s;
        width: auto;
    }
    
    .stFormSubmitButton button:hover {
        background-color: #1d4ed8;
        color: white !important;
    }
    
    /* Result box */
    .result-box {
        border-radius: 6px;
        padding: 1.25rem;
        margin-top: 1.5rem;
        border: none;
        background-color: #f9fafb;
    }
    
    .sentiment-positive {
        background-color: #ecfdf5;
        border-color: #10b981;
    }
    
    .sentiment-negative {
        background-color: #fef2f2;
        border-color: #ef4444;
    }
    
    .sentiment-label {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
        color: #1a1a1a;
    }
    
    .sentiment-positive .sentiment-label {
        color: #047857;
    }
    
    .sentiment-negative .sentiment-label {
        color: #dc2626;
    }
    
    .confidence-label {
        font-size: 0.85rem;
        color: #6b7280;
    }
    
    /* Examples section */
    .example-section {
        margin-top: 2.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e5e7eb;
    }
    
    .example-section h3 {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: #6b7280;
        margin-bottom: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Form container */
    .stForm {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
    }
    
    /* Any container borders */
    [data-testid="stForm"] {
        border: none !important;
        background: transparent !important;
    }
    
    /* Remove any remaining borders */
    .stContainer, .stVerticalBlock {
        border: none !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    try:
        model = load_model("sentiment_model.h5")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Make sure to train the model first by running the notebook")
        return None, None


model, tokenizer = load_model_and_tokenizer()

# Header
st.markdown("# Sentiment Analyzer")
st.markdown(
    '<p class="subtitle">Analyze the sentiment of movie reviews</p>',
    unsafe_allow_html=True,
)

# Check if example was clicked
if "example" in st.session_state:
    # Set the text area value directly
    if "review_input" not in st.session_state:
        st.session_state.review_input = st.session_state.example
    else:
        st.session_state.review_input = st.session_state.example
    # Clear the example after setting it
    del st.session_state.example

# Form for enter key submission
with st.form(key="sentiment_form"):
    # Main input
    review_text = st.text_area(
        "Review Text",
        placeholder="Enter your review...",
        height=120,
        label_visibility="collapsed",
        key="review_input",
    )

    # Analyze button
    submitted = st.form_submit_button("Analyze", use_container_width=False)

    if submitted:
        if not review_text.strip():
            st.warning("Please enter a review first")
        elif model is None or tokenizer is None:
            st.error("Model not loaded. Please train the model first.")
        else:
            with st.spinner("Analyzing..."):
                # Predict
                seq = tokenizer.texts_to_sequences([review_text])
                padded = pad_sequences(seq, maxlen=200)
                pred = model.predict(padded, verbose=0)[0][0]

                # Determine sentiment
                is_positive = pred > 0.5
                confidence = pred if is_positive else (1 - pred)
                sentiment = "Positive" if is_positive else "Negative"

                # Display result
                result_class = (
                    "sentiment-positive" if is_positive else "sentiment-negative"
                )
                st.markdown(
                    f"""
                    <div class="result-box {result_class}">
                        <div class="sentiment-label">{sentiment}</div>
                        <div class="confidence-label">{confidence*100:.1f}% confidence</div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

# Examples section
st.markdown('<div class="example-section">', unsafe_allow_html=True)
st.markdown("### Examples")

col1, col2 = st.columns(2, gap="small")

with col1:
    if st.button("Positive", use_container_width=True, key="pos_btn"):
        st.session_state.example = "This movie was absolutely amazing! The acting was superb and the storyline kept me engaged throughout."
        st.rerun()

with col2:
    if st.button("Negative", use_container_width=True, key="neg_btn"):
        st.session_state.example = "This was the worst movie I've ever seen. The plot was confusing and the acting was terrible."
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
