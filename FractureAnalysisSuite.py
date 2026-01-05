import streamlit as st
import base64
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="Fracture Analysis Suite - Home",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a Professional Medical UI ---
def load_css():
    st.markdown("""
    <style>
        /* --- Base & Colors --- */
        :root {
            --primary-color: #0068c9; /* Medical Blue */
            --primary-light: #e6f0fa;  /* Light blue for highlights */
            --bg-color: #f0f2f6;      /* Light Gray Background */
            --card-bg-color: #ffffff; /* White Card */
            --text-color: #31333F;
            --light-text-color: #555555;
            --border-radius: 10px;
            --box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
        
        /* --- Main App Styling --- */
        .stApp {
            background-color: var(--bg-color);
        }
        
        /* --- Main Content Area --- */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            color: var(--text-color);
        }

        /* --- Sidebar Styling --- */
        .st-emotion-cache-16txtl3 {
            background-color: var(--card-bg-color);
            border-right: 1px solid #e0e0e0;
        }
        .st-emotion-cache-16txtl3 .st-emotion-cache-1cypcdb {
            color: var(--primary-color);
            font-weight: 600;
        }

        /* --- Title Styling --- */
        h1 {
            color: var(--primary-color);
            font-weight: 700;
        }

        /* --- Custom "Card" Element --- */
        .card {
            background-color: var(--card-bg-color);
            border-radius: var(--border-radius);
            padding: 25px 30px;
            box-shadow: var(--box-shadow);
            border: 1px solid #e0e0e0;
            margin-bottom: 20px;
            height: 100%; /* Makes cards in columns the same height */
        }
        
        .card h3 {
            color: var(--primary-color);
            margin-bottom: 15px;
            border-bottom: 2px solid var(--primary-light);
            padding-bottom: 10px;
        }
        
        .card p, .card li {
            color: var(--light-text-color);
            line-height: 1.6;
        }
        
        /* --- Button Styling --- */
        .stButton button {
            background-color: var(--primary-color);
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
            transition: all 0.3s;
        }
        .stButton button:hover {
            background-color: #0056b3;
            transform: scale(1.02);
        }
        
        /* --- Expander Styling --- */
        .st-emotion-cache-p5msec {
            border-radius: var(--border-radius);
            border: 1px solid #e0e0e0;
            box-shadow: none;
        }
        .st-emotion-cache-p5msec summary {
            font-weight: 600;
            color: var(--primary-color);
        }
        
        /* --- Disclaimer Styling --- */
        .disclaimer {
            font-size: 0.9rem;
            color: #888888;
            font-style: italic;
            text-align: center;
            border-top: 1px solid #e0e0e0;
            padding-top: 20px;
            margin-top: 40px;
        }

    </style>
    """, unsafe_allow_html=True)

# Load the custom CSS
load_css()

# --- Page Content ---

st.sidebar.title("Navigation")
st.sidebar.info("Select a page from the list to get started.")

# --- Header ---
col1, col2 = st.columns([1, 4])
with col1:
    # Using an open-source, generic medical icon
    st.image("https://i.imgur.com/gQ27H8m.png", width=150) 

with col2:
    st.title("AI Fracture Analysis Suite")
    st.markdown("### Professional AI-powered tools for medical image analysis.")

st.markdown("---")

# --- Main Content using "Cards" ---
st.markdown(
    """
    <div class="card">
        <h3>Welcome to the Future of Radi-ology</h3>
        <p>This suite is a professional tool designed to assist medical professionals in the
        analysis of X-ray imagery. Our advanced 4-stage AI pipeline provides a
        robust framework for fracture detection, localization, and automated reporting.</p>
        <p>This system integrates multiple deep learning models to ensure high accuracy and provides
        transparent insights into its decision-making process.</p>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="card">
            <h3>How to Use This Tool</h3>
            <ol>
                <li><strong>Go to `🔬 Analyze X-Ray`</strong>: Use the sidebar to navigate to the main tool.</li>
                <li><strong>Upload Your Image</strong>: Select a high-quality X-ray (PNG or JPG).</li>
                <li><strong>Run Analysis</strong>: Click 'Run Analysis' to see the step-by-step results.</li>
                <li><strong>Generate Report</strong>: Click 'Generate PDF Report' for a detailed clinical analysis and a downloadable PDF file.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )
    
with col2:
    st.markdown(
        """
        <div class="card">
            <h3>Additional Resources</h3>
            <p>Our suite includes supplementary modules for education and transparency:</p>
            <ul>
                <li><strong>📚 Fracture Encyclopedia</strong>: Browse common types of bone fractures and their clinical descriptions.</li>
                <li><strong>⚙️ How It Works</strong>: View an animated explanation of the AI models that power this application.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Footer ---
st.markdown(
    """
    <div class="disclaimer">
        <p><strong>Disclaimer:</strong> This tool is for informational and academic purposes only and is not a substitute for
        professional medical diagnosis or treatment. Always consult with a qualified healthcare provider.</p>
    </div>
    """,
    unsafe_allow_html=True
)