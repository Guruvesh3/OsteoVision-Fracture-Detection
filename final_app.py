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
from fpdf import FPDF
import io
import time
import base64
import json
from datetime import datetime

# ---------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="OsteoVision Pro",
    page_icon="assets/logo.jpg", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------
# API KEY SETUP
# ---------------------------------------------------------------------
GOOGLE_API_KEY = "AIzaSyCf-7SVPNKRe69rV1sOcPC4GmOmbYjh_T4" 

try:
    if GOOGLE_API_KEY != "YOUR_API_KEY_HERE":
        genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"API Key Configuration Error: {e}")

# ---------------------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------------------
def load_css():
    st.markdown("""
        <style>
            html, body, .stApp, .stMarkdown, p, div, span, li, a {
                font-size: 1.15rem !important;
                line-height: 1.6;
            }
            /* Hide duplicate sidebar nav */
            [data-testid="stSidebarNav"] { display: none; }
            
            [data-testid="stHeader"] { background-color: #1a5c4f; visibility: hidden; }
            [data-testid="stFooter"] { visibility: hidden; }
            .stApp { background-color: #f0f2f6; }
            
            .custom-header {
                background-color: #1a5c4f;
                color: white;
                padding: 1rem 1.5rem;
                border-radius: 8px;
                display: flex;
                align-items: center;
                margin-bottom: 1.5rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
            .custom-header img {
                width: 50px;
                height: 50px;
                border-radius: 50%;
                margin-right: 15px;
            }
            .custom-header h1 {
                font-size: 2.4rem !important;
                color: white;
                margin: 0;
                font-weight: 600;
            }
            [data-testid="stSidebar"] {
                background-color: #FFFFFF;
                border-right: 1px solid #e0e0e0;
            }
            
            /* Reduced Sidebar Text Size */
            [data-testid="stRadio"] label {
                font-size: 1.2rem !important;
                font-weight: 600 !important;
                padding: 0.75rem 1rem !important;
                border-radius: 8px;
                margin: 0.25rem 0.5rem;
                transition: all 0.3s;
                display: block;
            }
            [data-testid="stRadio"] label:hover {
                background-color: #f0f2f6;
                color: #1a5c4f;
            }
            [data-testid="stRadio"] div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) {
                background-color: #1a5c4f;
                color: white;
            }
            
            h1 { font-size: 2.2rem !important; text-align: center; }
            h2 { text-align: center; color: #333; font-weight: 600; font-size: 2rem !important; }
            h3 { font-size: 1.7rem !important; }
            h4 { text-align: center; color: #555; font-weight: 400; margin-top: -10px; font-size: 1.3rem !important; }
            
            .stFileUploader {
                background-color: #ffffff;
                border: 2px dashed #1a5c4f;
                border-radius: 10px;
                padding: 2rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            }
            .stFileUploader > div > div { font-size: 1.2rem !important; }
            
            .stButton button {
                border-radius: 8px;
                padding: 0.75rem 1.5rem;
                font-size: 1.2rem !important;
                font-weight: 600;
                border: none;
                transition: all 0.3s;
            }
            /* Make sidebar buttons full width */
            [data-testid="stSidebar"] .stButton button, .stDownloadButton button { width: 100%; }
            
            /* Custom Call Button Style */
            .custom-call-button a {
                display: block;
                background-color: #1a5c4f !important;
                color: white !important;
                text-align: center;
                border-radius: 8px;
                padding: 0.75rem 1.5rem;
                font-size: 1.2rem !important;
                font-weight: 600;
                border: none;
                transition: all 0.3s;
                text-decoration: none;
                width: 100%;
            }
            .custom-call-button a:hover { opacity: 0.8; }

            .stButton button:hover { transform: scale(1.02); }
            .stButton button[kind="primary"] { background-color: #0068c9; color: white; }
            .stButton button[kind="secondary"] { background-color: #1a5c4f; color: white; }
            .stDownloadButton button { background-color: #28a745; color: white; }

            .status-box {
                border: 2px solid;
                border-radius: 8px;
                padding: 1.5rem;
                margin: 1.5rem 0;
                font-weight: 600;
                font-size: 1.4rem !important;
                text-align: center;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .status-box-fracture { border-color: #dc3545; background-color: #fde8ea; color: #b02a37; }
            .status-box-healthy { border-color: #28a745; background-color: #eaf7ec; color: #218838; }
            
            .report-section {
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 1.5rem;
                background-color: #fafafa;
                font-size: 1.15rem !important;
                line-height: 1.7;
            }
            .report-section ul { list-style: none; padding-left: 0; }
            .report-section li { margin-bottom: 12px; padding-left: 0; }
            .report-section strong { color: #1a5c4f; font-size: 1.2rem !important; }

            .chat-bubble {
                background-color: #ffffff;
                border-radius: 10px;
                padding: 1.5rem;
                margin-top: 1rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                border: 1px solid #e0e0e0;
                position: relative;
                font-size: 1.2rem !important;
                line-height: 1.6;
            }
            .chat-bubble::before {
                content: '';
                position: absolute;
                top: -10px;
                left: 50%;
                transform: translateX(-50%);
                width: 0;
                height: 0;
                border-left: 10px solid transparent;
                border-right: 10px solid transparent;
                border-bottom: 10px solid #ffffff;
            }
        </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# MODEL PATHS & CONFIG
# ---------------------------------------------------------------------
ASSET_DIR = Path("assets")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PATHS FROM YOUR WORKING CODE ---
VIT_PATH = ASSET_DIR / "vit_retrained.pth"
SWIN_PATH = ASSET_DIR / "swin_retrained.pth"
DENSENET_PATH = ASSET_DIR / "monai_fracatlas_model.pth"
YOLO_MODEL_PATH = Path(r"C:\Users\Guruvesh\Downloads\extracted_files\Yolov11.pt")
LOGO_PATH = ASSET_DIR / "logo.jpg"

# --- ORIGINAL THRESHOLD ---
FRACTURE_THRESHOLD = 0.63

# ---------------------------------------------------------------------
# LOAD MODELS (Cached)
# ---------------------------------------------------------------------
@st.cache_resource
def load_detection_models():
    try:
        vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
        vit.load_state_dict(torch.load(VIT_PATH, map_location=DEVICE))
        vit.to(DEVICE).eval()
        
        swin = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=2)
        swin.load_state_dict(torch.load(SWIN_PATH, map_location=DEVICE))
        swin.to(DEVICE).eval()
        
        # Original DenseNet (1 channel / Grayscale)
        densenet = monai.networks.nets.DenseNet121(spatial_dims=2, in_channels=1, out_channels=2)
        densenet.load_state_dict(torch.load(DENSENET_PATH, map_location=DEVICE))
        densenet.to(DEVICE).eval()
        return (vit, swin, densenet)
    except FileNotFoundError as e:
        st.error(f"Fatal Error: A detection model file was not found: {e}")
        st.stop()
        return None
    except RuntimeError as e:
        st.error(f"Fatal Model Error: {e}")
        st.stop()
        return None

@st.cache_resource
def load_yolo_model():
    try:
        model = YOLO(YOLO_MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Fatal Error loading YOLO model from {YOLO_MODEL_PATH}: {e}")
        st.stop()
        return None

@st.cache_resource
def load_gemini_model():
    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
        st.warning("Google API Key not set.")
        return None
    try:
        model = genai.GenerativeModel('models/gemini-flash-latest')
        return model
    except Exception as e:
        st.error(f"Error loading Gemini model: {e}")
        return None

# ---------------------------------------------------------------------
# AI PIPELINE FUNCTIONS
# ---------------------------------------------------------------------
def detect_fracture(image: Image.Image, models: tuple, demo_mode: str = "") -> dict:
    """
    Original Simple Average Logic + Demo Mode Override.
    """
    # --- DEMO MODE OVERRIDE ---
    if demo_mode == "force_healthy":
        return {"status": "No Fracture Detected", "confidence": 0.99}
    # --------------------------

    if models is None: return {"status": "Error", "confidence": 0.0}
    vit, swin, densenet = models
    
    # Original Transforms (RGB and Grayscale)
    transform_rgb = T.Compose([T.Resize((224, 224)), T.Grayscale(3), T.ToTensor(), T.Normalize([0.5]*3,[0.5]*3)])
    transform_gray = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5],[0.5])])
    
    img_rgb = transform_rgb(image.convert('RGB')).unsqueeze(0).to(DEVICE)
    img_gray = transform_gray(image.convert('L')).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        p_vit = torch.softmax(vit(img_rgb), dim=1)
        p_swin = torch.softmax(swin(img_rgb), dim=1)
        p_dense = torch.softmax(densenet(img_gray), dim=1)
        
        # Simple Average (The logic from your pasted code)
        # Class 0 is assumed to be "No Fracture/Healthy"
        avg_prob_healthy = (p_vit[0, 0] + p_swin[0, 0] + p_dense[0, 0]) / 3
        
        if avg_prob_healthy < FRACTURE_THRESHOLD:
            status = "Fracture Detected"
            confidence = 1.0 - float(avg_prob_healthy)
        else:
            status = "No Fracture Detected"
            confidence = float(avg_prob_healthy)

    return {"status": status, "confidence": confidence}

def localize_fracture(image: Image.Image, model: YOLO) -> (Image.Image, int):
    if model is None: return image, 0
    results = model.predict(source=image, save=False, conf=0.25, device=DEVICE)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    annotated_image_bgr = results[0].plot()
    annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_image_rgb), len(boxes)

def generate_gemini_report(image: Image.Image, model) -> dict:
    if model is None: 
        return {"error": "Gemini model not loaded."}

    generation_config = genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema={
            "type": "OBJECT",
            "properties": {
                "body_part": {"type": "STRING"},
                "fracture_type": {"type": "STRING"},
                "severity": {"type": "STRING"}
            },
            "required": ["body_part", "fracture_type", "severity"]
        }
    )

    prompt = (
        "You are an expert radiologist. An AI detection model has already confirmed "
        "a high-probability fracture in this X-ray image.\n\n"
        "Analyze the image and return a JSON object with three keys: "
        "'body_part', 'fracture_type', and 'severity'. "
        "Be clinical and descriptive."
    )

    try:
        response = model.generate_content([prompt, image], generation_config=generation_config)
        report_json = json.loads(response.text)
        return report_json
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        return {"error": f"Error generating report: {e}"}

# ---------------------------------------------------------------------
# PDF GENERATION
# ---------------------------------------------------------------------
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", 'B', 12)
        self.cell(0, 10, 'OsteoVision Pro - Clinical Report', 0, 0, 'C')
        self.ln(20)
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        self.cell(0, 10, "Generated by OsteoVision AI.", 0, 0, 'R')

def create_pdf_report(report_data: dict, patient_info: dict) -> bytes:
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Patient Details:", 0, 1)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 6, f"Name: {patient_info.get('name', 'N/A')}", 0, 1)
    pdf.cell(0, 6, f"Patient ID: {patient_info.get('id', 'N/A')}", 0, 1)
    pdf.cell(0, 6, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. AI Clinical Analysis", 0, 1)
    
    report_dict = report_data.get("report_dict", {})
    if "error" in report_dict:
        pdf.set_font("Arial", 'I', 11)
        pdf.set_text_color(255, 0, 0)
        pdf.multi_cell(0, 5, f"Error: {report_dict['error']}")
        pdf.set_text_color(0, 0, 0)
    else:
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 5, "Body Part:", 0, 1)
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 5, report_dict.get('body_part', 'N/A').encode('latin-1', 'replace').decode('latin-1'))
        pdf.ln(3)
        
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 5, "Predicted Fracture Type:", 0, 1)
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 5, report_dict.get('fracture_type', 'N/A').encode('latin-1', 'replace').decode('latin-1'))
        pdf.ln(3)
        
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 5, "Estimated Severity:", 0, 1)
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 5, report_dict.get('severity', 'N/A').encode('latin-1', 'replace').decode('latin-1'))

    pdf.ln(5)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. Visual Analysis", 0, 1)
    
    try:
        buffer_orig = io.BytesIO()
        report_data["original_image"].save(buffer_orig, format="PNG")
        buffer_orig.seek(0)
        pdf.image(buffer_orig, x=10, y=pdf.get_y(), w=85, type='PNG')
        
        if report_data.get("localized_image"):
            buffer_loc = io.BytesIO()
            report_data["localized_image"].save(buffer_loc, format="PNG")
            buffer_loc.seek(0)
            pdf.image(buffer_loc, x=110, y=pdf.get_y(), w=85, type='PNG')
    
    except Exception as e:
        pdf.cell(0, 5, f"(Error embedding images)", 0, 1)

    return bytes(pdf.output(dest='S'))

# ---------------------------------------------------------------------
# PAGE DEFINITIONS
# ---------------------------------------------------------------------

def render_main_page():
    load_css()
    
    st.title("AI-Powered X-ray Fracture Detection", anchor=False)
    st.markdown("<h4>Empowering Diagnostics with Artificial Intelligence</h4>", unsafe_allow_html=True)

    col_main, col_sidebar = st.columns([3, 1])

    # --- SIDEBAR CONTROLS ---
    with col_sidebar:
        with st.container(border=True):
            st.subheader("Patient Details")
            p_name = st.text_input("Patient Name", placeholder="Enter name")
            p_id = st.text_input("Patient ID", placeholder="ID-12345")
            
            st.divider()
            
            uploaded_file = st.file_uploader(
                "Upload X-Ray",
                type=["jpg", "jpeg", "png"]
            )
            
            if uploaded_file is not None:
                st.session_state.original_image = Image.open(uploaded_file)
                st.session_state.uploaded_filename = uploaded_file.name
                st.image(st.session_state.original_image, caption="Preview", use_column_width=True)

        # Demo Code Input (Hidden)
        demo_pass = st.text_input("Report Error", type="password", key="demo_input", help="Enter 'force_healthy' to override.")

        if uploaded_file is not None:
            if st.button("Run Analysis", type="primary", key="run_analysis"):
                img_buffer = st.session_state.original_image
                filename_buffer = st.session_state.uploaded_filename
                st.session_state.clear()
                st.session_state.original_image = img_buffer
                st.session_state.uploaded_filename = filename_buffer
                
                st.session_state.p_name = p_name
                st.session_state.p_id = p_id
                
                st.session_state.analysis_running = True
                
                with st.spinner("Step 1/3: Running Fracture Detection..."):
                    time.sleep(0.5)
                    detection_models = load_detection_models()
                    
                    det_result = detect_fracture(
                        st.session_state.original_image, 
                        detection_models,
                        demo_mode=demo_pass
                    )
                    st.session_state.detection_status = det_result["status"]
                    st.session_state.detection_conf = det_result["confidence"]
                
                if st.session_state.detection_status == "Fracture Detected":
                    with st.spinner("Step 2/3: Running Localization..."):
                        time.sleep(0.5)
                        yolo_model = load_yolo_model()
                        st.session_state.localized_image, st.session_state.box_count = localize_fracture(st.session_state.original_image, yolo_model)
                    
                    with st.spinner("Step 3/3: Generating AI Clinical Report..."):
                        time.sleep(0.5)
                        gemini_model = load_gemini_model()
                        st.session_state.report_data = generate_gemini_report(st.session_state.original_image, gemini_model)
                        
                st.success("Analysis pipeline completed successfully.", icon="✅")
                st.session_state.analysis_complete = True

    # --- RESULTS AREA ---
    if st.session_state.get('analysis_complete', False):
        
        detection_status = st.session_state.get('detection_status')
        confidence = st.session_state.get('detection_conf', 0.0)
        
        with col_main:
            st.markdown("---")
            st.header("Diagnosis Result")

            if detection_status == "Fracture Detected":
                st.markdown('<div class="status-box status-box-fracture">⚠️ Fracture Detected</div>', unsafe_allow_html=True)
                
                st.write(f"**AI Confidence:** {confidence*100:.1f}%")
                st.progress(confidence)
                
                with st.container(border=True):
                    st.subheader("Visual Analysis")
                    col_img1, col_img2 = st.columns(2)
                    with col_img1:
                        st.image(st.session_state.original_image, caption="Original Image", use_column_width=True)
                    with col_img2:
                        st.image(st.session_state.localized_image, caption=f"Localized ({st.session_state.box_count} regions)", use_column_width=True)
                
                with st.container(border=True):
                    st.subheader("AI Clinical Report")
                    
                    report_data = st.session_state.get('report_data', {})
                    if "error" in report_data:
                        st.error(f"Error: {report_data['error']}")
                    else:
                        st.markdown(f"""
                        <div class="report-section">
                            <ul>
                                <li><strong>Body Part:</strong> {report_data.get('body_part', 'N/A')}</li>
                                <li><strong>Fracture Type:</strong> {report_data.get('fracture_type', 'N/A')}</li>
                                <li><strong>Severity:</strong> {report_data.get('severity', 'N/A')}</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

                    try:
                        pdf_data = {
                            "report_dict": st.session_state.report_data,
                            "original_image": st.session_state.original_image,
                            "localized_image": st.session_state.localized_image
                        }
                        p_info = {"name": st.session_state.get("p_name", ""), "id": st.session_state.get("p_id", "")}
                        pdf_bytes = create_pdf_report(pdf_data, p_info)
                        
                        st.download_button(
                            label="⬇️ Download Clinical Report (PDF)",
                            data=pdf_bytes,
                            file_name=f"Report_{st.session_state.uploaded_filename}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error creating PDF: {e}")

                st.markdown("---")
                st.subheader("💬 Ask Dr. George (AI Assistant)")
                
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if prompt := st.chat_input("Ask about this fracture..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        gemini = load_gemini_model()
                        if gemini:
                            try:
                                chat_prompt = f"The user is asking about the X-ray you analyzed. User question: {prompt}. Keep answer brief and clinical."
                                response = gemini.generate_content([chat_prompt, st.session_state.original_image])
                                st.markdown(response.text)
                                st.session_state.messages.append({"role": "assistant", "content": response.text})
                            except Exception as e:
                                st.error("AI Error.")
                        else:
                            st.error("AI not available.")


            elif detection_status == "No Fracture Detected":
                st.markdown('<div class="status-box status-box-healthy">✅ No Fracture Detected</div>', unsafe_allow_html=True)
                st.write(f"**AI Confidence (Healthy):** {confidence*100:.1f}%")
                st.progress(confidence)
                
                with st.container(border=True):
                    st.image(st.session_state.original_image, caption="Analyzed Image", use_column_width=True)
            
            else:
                st.error(f"An error occurred: {detection_status}")


    with col_sidebar:
        with st.container(border=True):
            st.subheader("Features")
            st.markdown("""
            - 🧑‍⚕️ **Second Opinions:** Verify diagnosis with leading experts.
            - ⚡ **Instant Detection:** Instant results.
            - 🌍 **Available Globally:** Remote access.
            """)
        
        with st.container(border=True):
            st.subheader("Teleconsult")
            st.info("Click button to call patient advisors.")
            st.markdown(
                '<div class="custom-call-button"><a href="tel:+919356738636" target="_blank">Book a Call</a></div>',
                unsafe_allow_html=True
            )
        
        with st.container(border=True):
            st.subheader("Help & Support")
            st.markdown("**support@osteovision.demo**")

def render_encyclopedia_page():
    load_css()
    st.title("📚 Fracture Encyclopedia")
    st.markdown("A visual reference guide to common types of bone fractures.")
    st.info("This information is for educational purposes only.")

    # Restored Full Detailed Text
    fractures = [
        {
            "name": "Stable (Hairline or Stress) Fracture",
            "desc": "A hairline or stress fracture is a small crack or severe bruise within a bone. This injury is most common in athletes, especially those in sports that involve repetitive running or jumping.",
            "details": "**Cause:** Repetitive force, overuse, or suddenly increasing activity.\n\n**Commonly Affects:** Weight-bearing bones, such as the tibia (shin) or metatarsals (foot).",
            "img_path": "C:\\Users\\Guruvesh\\Downloads\\extracted_files\\stable_fracture_image_animated.png"
        },
        {
            "name": "Oblique Fracture",
            "desc": "An oblique fracture is a break that is diagonal across the bone, at an angle.",
            "details": "**Cause:** Typically caused by a sharp, angled blow or a fall where force is applied at an angle.\n\n**Commonly Affects:** Long bones such as the femur (thigh), tibia (shin), and humerus (upper arm).",
            "img_path": "C:\\Users\\Guruvesh\\Downloads\\extracted_files\\oblique_fracture_animated.png"
        },
        {
            "name": "Comminuted Fracture",
            "desc": "A comminuted fracture is a severe type of break where the bone shatters into three or more pieces.",
            "details": "**Cause:** Almost always caused by high-impact trauma, such as a severe fall, a car accident, or a crushing injury.\n\n**Note:** These fractures often require surgery to realign the bone fragments.",
            "img_path": "C:\\Users\\Guruvesh\\Downloads\\extracted_files\\comminuted_fracture_animated.png"
        },
        {
            "name": "Greenstick Fracture",
            "desc": "A greenstick fracture is an incomplete break in which the bone is bent. This type of fracture occurs most often in children.",
            "details": "**Description:** The bone cracks on one side but does not break all the way through, similar to bending a 'green' stick.\n\n**Commonly Affects:** Long bones in children, such as the forearm.",
            "img_path": "C:\\Users\\Guruvesh\\Downloads\\extracted_files\\greenstick_fracture_animated.png"
        }
    ]

    for fracture in fractures:
        with st.container(border=True):
            st.subheader(fracture["name"])
            col_img, col_desc = st.columns([1, 2])
            with col_img:
                try:
                    st.image(fracture["img_path"], caption=f"Illustration", use_column_width=True)
                except:
                    st.error(f"Image missing: {fracture['img_path']}")
            with col_desc:
                st.markdown(fracture["desc"])
                st.markdown("---")
                st.markdown(fracture["details"])
    
    st.warning("**Open (Compound) Fracture:** Medical emergency where bone pierces skin.")
    
    _, col_img_open, _ = st.columns([1, 2, 1])
    with col_img_open:
        try:
            st.image("C:\\Users\\Guruvesh\\Downloads\\extracted_files\\open_fracture_animated.png", caption="Open Fracture", use_column_width=True)
        except:
            pass

def render_how_it_works_page():
    load_css()
    st.title("⚙️ How Our AI Pipeline Works")
    st.markdown("An interactive explanation of the technology.")

    if 'how_it_works_step' not in st.session_state:
        st.session_state.how_it_works_step = 'intro'

    col1, col2 = st.columns([1, 2])

    with col1:
        try:
            st.image("C:\\Users\\Guruvesh\\Downloads\\extracted_files\\dr.George.png", caption="Dr. George Vision", use_column_width=True)
        except:
            st.error("Avatar missing.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Step 1: Detection", use_container_width=True, type="primary" if st.session_state.how_it_works_step == 'step1' else "secondary"):
            st.session_state.how_it_works_step = 'step1'
            st.rerun()
        if st.button("Step 2: Localization", use_container_width=True, type="primary" if st.session_state.how_it_works_step == 'step2' else "secondary"):
            st.session_state.how_it_works_step = 'step2'
            st.rerun()
        if st.button("Step 3: Reporting", use_container_width=True, type="primary" if st.session_state.how_it_works_step == 'step3' else "secondary"):
            st.session_state.how_it_works_step = 'step3'
            st.rerun()
        st.markdown("---")
        if st.session_state.how_it_works_step != 'intro':
            if st.button("Back to Overview", use_container_width=True):
                st.session_state.how_it_works_step = 'intro'
                st.rerun()

    with col2:
        chat_placeholder = st.empty()
        # Restored Full Detailed Text
        content = {
            'intro': """
                Hello! I'm <b>Dr. George Vision</b>, and I'm here to be your guide.
                <br><br>
                Our "OsteoVision" app uses a powerful 3-stage pipeline to analyze X-rays. It's a sophisticated process designed for maximum accuracy, turning a complex image into a clear, understandable report.
                <br><br>
                <b>These are the main steps of execution:</b>
                <ol>
                    <li><b>Fracture Detection</b> ("Is it broken?")</li>
                    <li><b>Fracture Localization</b> ("Where is it?")</li>
                    <li><b>AI Clinical Report</b> ("What does it mean?")</li>
                </ol>
                Which step would you like to learn more about? Please select an option from the buttons on the left.
            """,
            'step1': """
                <b>Step 1: Fracture Detection (The "Is it broken?" stage)</b>
                <br><br>
                This is the most critical step. To ensure we don't miss anything and provide a reliable 'second opinion', we use a <b>"panel of experts"</b> approach. Your X-ray is simultaneously analyzed by <b>three different, powerful AI models</b>:
                <ul>
                    <li><b>ViT (Vision Transformer):</b> A state-of-the-art model that sees the image holistically, understanding the overall context.</li>
                    <li><b>Swin Transformer:</b> Another advanced model that is excellent at finding small, localized details that might otherwise be missed.</li>
                    <li><b>DenseNet:</b> A classic, highly reliable model famous in medical imaging for its robust feature extraction.</li>
                </ul>
                We take the results from all three, and if our ensemble agrees there's a high probability of a fracture (based on our `FRACTURE_THRESHOLD`), we confidently move to the next stage.
            """,
            'step2': """
                <b>Step 2: Fracture Localization (The "Where is it?" stage)</b>
                <br><br>
                Great! Once we've confirmed <i>that</i> a fracture exists, we must find <i>where</i> it is. For this, we use our champion localization model: <b>YOLOv11</b>.
                <br><br>
                This model, which scored an incredible <b>81.6% mAP</b> (a measure of accuracy) in our tests, is specifically trained to scan the image and draw precise bounding boxes around every fracture it finds.
                <br><br>
                This is what you see in the "Localized Result" image, and it's critical for understanding the exact location and extent of the injury.
            """,
            'step3': """
                <b>Step 3: AI Clinical Report (The "What does it mean?" stage)</b>
                <br><br>
                This is the final piece of the puzzle. We now send the <i>original</i> X-ray (along with the confirmation that a fracture was found) to Google's <b>Gemini</b> model, a powerful generative AI. We give it a very specific prompt:
                <br><br>
                <i>"You are an expert radiologist. A fracture has been detected. Provide a structured report on Body Part, Fracture Type, and Severity."</i>
                <br><br>
                Gemini analyzes the image's visual data and provides the structured, human-readable report you see in the app. This is then combined with the images to generate your final, downloadable PDF report.
            """
        }
        current_step_content = content.get(st.session_state.how_it_works_step, 'intro')
        chat_placeholder.markdown(f'<div class="chat-bubble">{current_step_content}</div>', unsafe_allow_html=True)

try:
    if not (ASSET_DIR / "logo.jpg").exists():
        st.error("Logo not found in assets.")
        st.stop()
    with open(LOGO_PATH, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""<div class="custom-header"><img src="data:image/jpeg;base64,{logo_base64}" alt="Logo"><h1>OsteoVision Pro</h1></div>""", unsafe_allow_html=True)
except:
    pass

st.sidebar.title("More Options")
page = st.sidebar.radio("Go to", ["Analyze X-Ray", "Fracture Encyclopedia", "How It Works"], label_visibility="collapsed")

if page == "Analyze X-Ray": render_main_page()
elif page == "Fracture Encyclopedia": render_encyclopedia_page()
elif page == "How It Works": render_how_it_works_page()