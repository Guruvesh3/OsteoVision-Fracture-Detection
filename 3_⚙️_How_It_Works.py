import streamlit as st

st.set_page_config(page_title="How It Works", page_icon="⚙️", layout="wide")

# --- Custom CSS for Animation and Layout ---
st.markdown("""
<style>
    :root {
        --primary-color: #0068c9;
        --primary-light: #e6f0fa;
        --bg-color: #f0f2f6;
        --card-bg-color: #ffffff;
        --text-color: #31333F;
        --light-text-color: #555555;
        --border-radius: 10px;
        --box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .stApp {
        background-color: var(--bg-color);
    }
    .main .block-container {
        padding-top: 2rem;
        color: var(--text-color);
    }
    h1 {
        color: var(--primary-color);
        font-weight: 700;
    }
    h3 {
        color: var(--primary-color);
        border-bottom: 2px solid var(--primary-light);
        padding-bottom: 5px;
    }
    .card {
        background-color: var(--card-bg-color);
        border-radius: var(--border-radius);
        padding: 25px 30px;
        box-shadow: var(--box-shadow);
        border: 1px solid #e0e0e0;
        margin-bottom: 20px;
    }
    
    /* --- Animated Flowchart --- */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .flow-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
    }
    
    .flow-step {
        background-color: var(--card-bg-color);
        border: 2px solid var(--primary-color);
        border-radius: var(--border-radius);
        padding: 20px;
        margin: 10px 0;
        width: 80%;
        box-shadow: var(--box-shadow);
        animation: fadeIn 0.5s ease-out both;
    }
    
    .flow-step h4 {
        color: var(--primary-color);
        margin: 0;
        padding: 0;
    }
    
    .flow-step p {
        color: var(--light-text-color);
        margin-bottom: 0;
    }
    
    .flow-arrow {
        font-size: 2.5rem;
        color: var(--primary-color);
        margin: -10px 0;
        animation: fadeIn 0.5s ease-out both;
    }
    
    /* Animation delays */
    .flow-step:nth-child(1) { animation-delay: 0.1s; }
    .flow-arrow:nth-child(2) { animation-delay: 0.3s; }
    .flow-step:nth-child(3) { animation-delay: 0.5s; }
    .flow-arrow:nth-child(4) { animation-delay: 0.7s; }
    .flow-step:nth-child(5) { animation-delay: 0.9s; }
    .flow-arrow:nth-child(6) { animation-delay: 1.1s; }
    .flow-step:nth-child(7) { animation-delay: 1.3s; }
    .flow-arrow:nth-child(8) { animation-delay: 1.5s; }
    .flow-step:nth-child(9) { animation-delay: 1.7s; }

</style>
""", unsafe_allow_html=True)

st.title("⚙️ How Our AI Pipeline Works")
st.markdown("We believe in transparency. Here is an animated breakdown of the AI models powering this application.")

st.markdown("---")

st.header("The 4-Step Analysis Pipeline")
st.markdown("""
When you upload an image and click "Run Analysis," this is the exact process that happens in sequence.
""")

# --- Animated Flowchart ---
st.markdown("""
<div class="flow-container">
    <div class="flow-step">
        <h4>1. Image Validation</h4>
        <p>A simple check confirms your upload is a valid, high-contrast X-ray image.</p>
    </div>
    <div class="flow-arrow">↓</div>
    <div class="flow-step">
        <h4>2. Fracture Detection (The Ensemble)</h4>
        <p>Your image is sent to three models (ViT, Swin, DenseNet). They vote to determine if a fracture is present.</p>
    </div>
    <div class="flow-arrow">↓</div>
    <div class="flow-step">
        <h4>3. Fracture Localization (YOLOv11)</h4>
        <p>If a fracture is detected, our champion 81.6% mAP model (YOLOv11) draws the bounding boxes to find its exact location.</p>
    </div>
    <div class="flow-arrow">↓</div>
    <div class="flow-step">
        <h4>4. AI Reporting (Gemini)</h4>
        <p>When you click "Generate Report," the original image is sent to Google's Gemini model to analyze the Body Part, Type, and Severity.</p>
    </div>
    <div class="flow-arrow">↓</div>
    <div class="flow-step">
        <h4>5. PDF Generation (fpdf2)</h4>
        <p>All this data is compiled into a single, downloadable PDF document for your records.</p>
    </div>
</div>
""", unsafe_allow_html=True)