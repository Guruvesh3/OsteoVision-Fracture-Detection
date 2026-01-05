import streamlit as st
import torch
import timm
from torchvision import transforms as T
from PIL import Image, ImageOps
import monai
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import os
import cv2
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from fpdf import FPDF
import io
import time

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(page_title="Analyze X-Ray", page_icon="🔬", layout="wide")

# --- Custom CSS for a Professional Medical UI ---
def load_css():
    st.markdown("""
    <style>
        :root {
            --primary-color: #0068c9; /* Medical Blue */
            --bg-color: #f0f2f6;      /* Light Gray Background */
            --card-bg-color: #ffffff; /* White Card */
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
        /* --- Custom "Card" Element --- */
        .card {
            background-color: var(--card-bg-color);
            border-radius: var(--border-radius);
            padding: 25px 30px;
            box-shadow: var(--box-shadow);
            border: 1px solid #e0e0e0;
            margin-bottom: 20px;
        }
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
        .stDownloadButton button {
            background-color: #28a745; /* Green for download */
        }
        .stDownloadButton button:hover {
            background-color: #218838;
        }
        /* --- Sidebar Styling --- */
        .st-emotion-cache-16txtl3 .st-emotion-cache-1cypcdb {
            color: var(--primary-color);
            font-weight: 600;
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
        /* --- Status Box --- */
        .status-box {
            border: 1px solid;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            font-weight: 500;
        }
        .status-valid {
            border-color: #28a745;
            background-color: #eaf7ec;
            color: #218838;
        }
        .status-detect {
            border-color: #0068c9;
            background-color: #e6f0fa;
            color: #0056b3;
        }
        .status-no-detect {
            border-color: #28a745;
            background-color: #eaf7ec;
            color: #218838;
        }
        .status-warn {
            border-color: #ffc107;
            background-color: #fff8e1;
            color: #b38600;
        }
    </style>
    """, unsafe_allow_html=True)

load_css()
st.title("🔬 Analyze X-Ray")
st.markdown("Upload your X-ray image to begin the step-by-step analysis.")

# --- IMPORTANT: API KEY SETUP ---
GOOGLE_API_KEY = "AIzaSyCf-7SVPNKRe69rV1sOcPC4GmOmbYjh_T4"  # <--- !!! PASTE YOUR KEY HERE !!!

try:
    if GOOGLE_API_KEY != "YOUR_API_KEY_HERE":
        genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"API Key Configuration Error: {e}")

# -----------------------------
# MODEL PATHS & CONFIG
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIT_PATH = Path(r"C:\Users\Guruvesh\Downloads\extracted_files\vit_retrained.pth")
SWIN_PATH = Path(r"C:\Users\Guruvesh\Downloads\extracted_files\swin_retrained.pth")
DENSENET_PATH = Path(r"C:\Users\Guruvesh\Downloads\extracted_files\monai_fracatlas_model.pth")
YOLO_MODEL_PATH = Path(r"C:\Users\Guruvesh\Downloads\extracted_files\Yolov11.pt")
FRACTURE_THRESHOLD = 0.63

# -----------------------------
# LOAD MODELS (Cached)
# -----------------------------
# (All functions are identical to the working version)
@st.cache_resource
def load_detection_models():
    try:
        vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
        vit.load_state_dict(torch.load(VIT_PATH, map_location=DEVICE))
        vit.to(DEVICE).eval()
        swin = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=2)
        swin.load_state_dict(torch.load(SWIN_PATH, map_location=DEVICE))
        swin.to(DEVICE).eval()
        densenet = monai.networks.nets.DenseNet121(spatial_dims=2, in_channels=1, out_channels=2)
        densenet.load_state_dict(torch.load(DENSENET_PATH, map_location=DEVICE))
        densenet.to(DEVICE).eval()
        return vit, swin, densenet
    except FileNotFoundError as e:
        st.error(f"Error loading detection models. File not found: {e.filename}")
        return None

@st.cache_resource
def load_yolo_model():
    try:
        model = YOLO(YOLO_MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Error loading YOLO model. File not found: {YOLO_MODEL_PATH}")
        return None

@st.cache_resource
def load_gemini_model():
    if GOOGLE_API_KEY != "YOUR_API_KEY_HERE":
        return genai.GenerativeModel('models/gemini-flash-latest')
    return None

# -----------------------------
# TOOL 1: VALIDATE IMAGE (NEW)
# -----------------------------
def validate_image(image: Image.Image) -> bool:
    """A simple check to see if the image is a valid X-ray."""
    try:
        # Check if it's grayscale (most X-rays are)
        if image.mode != 'L' and image.mode != 'RGB':
            # Try to convert
            image = image.convert('L')
        
        # Check for very small images
        if image.width < 100 or image.height < 100:
            st.markdown('<div class="status-box status-warn">⚠️ **Validation Failed:** Image is too small. Please upload a higher-resolution X-ray.</div>', unsafe_allow_html=True)
            return False
        
        # A simple "is this an X-ray?" check (simulated)
        # In a real app, this would be a simple classifier.
        # For now, we just check if it's mostly grayscale.
        if image.mode == 'RGB':
            img_gray = ImageOps.grayscale(image)
        else:
            img_gray = image
        
        extrema = img_gray.getextrema()
        if (extrema[1] - extrema[0]) < 50: # Very low contrast
            st.markdown('<div class="status-box status-warn">⚠️ **Validation Failed:** Image appears to have very low contrast or is not a valid X-ray.</div>', unsafe_allow_html=True)
            return False

        return True
        
    except Exception as e:
        st.error(f"Image validation error: {e}")
        return False

# -----------------------------
# TOOL 2: FRACTURE DETECTION
# -----------------------------
def detect_fracture(image: Image.Image, models: tuple) -> dict:
    if models is None: return {"fracture_status": "Error: Models not loaded."}
    vit, swin, densenet = models
    transform_rgb = T.Compose([T.Resize((224, 224)), T.Grayscale(3), T.ToTensor(), T.Normalize([0.5]*3,[0.5]*3)])
    transform_gray = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5],[0.5])])
    img_rgb = transform_rgb(image.convert('RGB')).unsqueeze(0).to(DEVICE)
    img_gray = transform_gray(image.convert('L')).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out_vit = torch.softmax(vit(img_rgb), dim=1)
        out_swin = torch.softmax(swin(img_rgb), dim=1)
        out_dense = torch.softmax(densenet(img_gray), dim=1)
        avg_prob_fracture = (out_vit[0, 0] + out_swin[0, 0] + out_dense[0, 0]) / 3
    status = "🦴 Fracture Detected" if avg_prob_fracture < FRACTURE_THRESHOLD else "✅ No Fracture Detected"
    return {"fracture_status": status, "image": image}

# -----------------------------
# TOOL 3: FRACTURE LOCALIZATION
# -----------------------------
def localize_fracture(input_data: dict, model: YOLO) -> dict:
    if model is None: return {**input_data, "localization_boxes": [], "localized_image": None}
    image = input_data["image"]
    fracture_status = input_data.get("fracture_status") 
    if fracture_status != "🦴 Fracture Detected":
        return {**input_data, "localization_boxes": [], "localized_image": None}
    results = model.predict(source=image, save=False, conf=0.25, device=DEVICE)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    annotated_image_bgr = results[0].plot()
    annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
    return {**input_data, "localization_boxes": boxes, "localized_image": annotated_image_rgb}

# -----------------------------
# TOOL 4: GENAI REPORT GENERATION
# -----------------------------
def generate_gemini_report(input_data: dict, model) -> dict:
    if model is None: return {**input_data, "gemini_report": "Error: Gemini model not loaded. Check API Key."}
    if input_data.get("fracture_status") != "🦴 Fracture Detected":
        return {**input_data, "gemini_report": "No fracture detected."}
    try:
        image = input_data["image"]
        prompt = (
            "You are an expert radiologist. An AI detection model has already confirmed "
            "a high-probability fracture in this X-ray image.\n\n"
            "Your task is to analyze the image and provide a structured report on the following topics ONLY:\n"
            "1.  **Body Part:** (e.g., Hand, Wrist, Femur)\n"
            "2.  **Predicted Fracture Type:** (e.g., Hairline, Comminuted, Oblique)\n"
            "3.  **Estimated Severity:** (e.g., Mild, Moderate, Severe)\n\n"
            "Please be concise and stick to this format."
        )
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        }
        response = model.generate_content([prompt, image], safety_settings=safety_settings)
        final_report = response.text
        return {**input_data, "gemini_report": final_report}
    except Exception as e:
        st.error(f"--- REAL GEMINI API ERROR ---")
        st.error(f"{e}")
        st.error("This is likely an invalid API key, an expired key, or a billing issue with your Google account.")
        return {**input_data, "gemini_report": "Error: Could not generate report. See error above."}

# -----------------------------
# TOOL 5: PDF GENERATION
# -----------------------------
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", 'B', 12)
        self.cell(0, 10, 'AI Fracture Analysis Report', 0, 0, 'C')
        self.ln(20)
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        self.cell(0, 10, "Medical Disclaimer: For informational purposes only.", 0, 0, 'R')

def create_pdf_report(report_data: dict) -> bytes:
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", '', 12)
    
    # --- 1. Gemini AI Report ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. AI Clinical Analysis", 0, 1)
    pdf.set_font("Arial", '', 11)
    report_text = report_data.get("gemini_report", "No AI report generated.").encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 5, report_text)
    pdf.ln(5)

    # --- 2. Technical Data ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. Technical Localization Data", 0, 1)
    pdf.set_font("Arial", '', 11)
    boxes = report_data.get("localization_boxes", [])
    if not boxes:
        pdf.cell(0, 5, "No fractures localized.")
    else:
        for i, box in enumerate(boxes):
            coords = [round(coord, 2) for coord in box]
            pdf.cell(0, 5, f"  - Fracture {i+1} [x1, y1, x2, y2]: {coords}", 0, 1)
    pdf.ln(5)

    # --- 3. Localized Image ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "3. Localized Image", 0, 1)
    localized_image = report_data.get("localized_image")
    if localized_image is not None:
        try:
            pil_img = Image.fromarray(localized_image)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            buffer.seek(0)
            pdf.image(buffer, x=pdf.get_x() + 10, y=pdf.get_y(), w=170, type='PNG')
        except Exception as e:
            pdf.set_font("Arial", 'I', 11)
            pdf.cell(0, 5, f"(Error embedding image: {e})", 0, 1)
            
    return bytes(pdf.output(dest='S'))

# ============================================
# 🖥️ STREAMLIT UI LAYOUT (NEW STEP-BY-STEP FLOW)
# ============================================

# --- 1. Uploader ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("1. Upload Image")
uploaded_file = st.file_uploader("Browse your files...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # --- 2. "Run Analysis" Button ---
    if st.button("Run Analysis", type="primary", use_container_width=True):
        # Clear old results
        st.session_state.clear()
        st.session_state.image_processed = True
        st.session_state.original_image = image
        
        # --- STEP-BY-STEP EXECUTION ---
        # We will now show the flow as it happens
        
        # Step A: Validation
        with st.spinner("Step 1/4: Validating image format..."):
            time.sleep(0.5) # Simulate work
            is_valid = validate_image(image)
            st.session_state.is_valid = is_valid
        
        if is_valid:
            st.markdown('<div class="status-box status-valid">✅ **Validation Complete:** Image is a valid format.</div>', unsafe_allow_html=True)

            # Step B: Detection
            with st.spinner("Step 2/4: Running Fracture Detection (ViT, Swin, DenseNet)..."):
                detection_models = load_detection_models()
                detection_result = detect_fracture(image, detection_models)
                st.session_state.analysis_result = detection_result
            
            fracture_status = detection_result.get("fracture_status")
            st.session_state.fracture_status = fracture_status

            if fracture_status == "✅ No Fracture Detected":
                st.markdown('<div class="status-box status-no-detect">✅ **Detection Complete:** No fracture was detected by the AI ensemble.</div>', unsafe_allow_html=True)

            elif fracture_status == "🦴 Fracture Detected":
                st.markdown('<div class="status-box status-detect">🦴 **Detection Complete:** A fracture has been detected. Proceeding to localization...</div>', unsafe_allow_html=True)
                
                # Step C: Localization
                with st.spinner("Step 3/4: Running Localization (YOLOv11)..."):
                    yolo_model = load_yolo_model()
                    localization_result = localize_fracture(st.session_state.analysis_result, yolo_model)
                    st.session_state.analysis_result = localization_result
            
            st.success("Analysis Pipeline Complete!")
            time.sleep(1)
            
        else:
            st.error("Analysis Halted: Image is not valid.")
            
# --- 3. Results Display (Side-by-Side) ---
if 'analysis_result' in st.session_state:
    
    st.markdown("---")
    st.header("Analysis Results")
    
    result_data = st.session_state.analysis_result
    localized_img = result_data.get("localized_image")
    fracture_status = result_data.get("fracture_status")
    
    if fracture_status == "✅ No Fracture Detected":
        st.success("✅ **Final Result:** No fracture was detected.")
        st.image(st.session_state.original_image, caption="Original Image", use_column_width=True)
    
    elif fracture_status == "🦴 Fracture Detected":
        st.markdown("### Side-by-Side Comparison")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.original_image, caption="Original Image (Input)", use_column_width=True)
        with col2:
            if localized_img is not None:
                st.image(localized_img, caption="Localized Fracture (YOLOv11)", use_column_width=True)
            else:
                st.warning("Detection model found a fracture, but localization model did not find a box.")
        
        st.markdown("---")
        st.header("Report Generation")
        
        # --- Generate Report Button ---
        if st.button("Generate PDF Report", use_container_width=True):
            if GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
                st.error("Please paste your GOOGLE_API_KEY at the top of this script.")
            else:
                with st.spinner("Generating AI clinical report and building PDF..."):
                    gemini_model = load_gemini_model()
                    report_data = generate_gemini_report(result_data, gemini_model)
                    pdf_bytes = create_pdf_report(report_data)
                    st.session_state.pdf_data = pdf_bytes
                st.success("PDF Report Generated!")
                st.balloons()
                
        # --- Download Button ---
        if 'pdf_data' in st.session_state:
            st.download_button(
                label="⬇️ Download Report as PDF",
                data=st.session_state.pdf_data,
                file_name=f"fracture_report_{uploaded_file.name.split('.')[0]}.pdf",
                mime="application/pdf",
                use_container_width=True
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