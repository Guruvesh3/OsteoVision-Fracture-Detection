# OsteoVision: Advanced Agentic AI for Fracture Diagnosis

![YOLOv11](https://img.shields.io/badge/Localization-YOLOv11-blue) ![Ensemble](https://img.shields.io/badge/Detection-ViT%20%7C%20Swin%20%7C%20MONAI-purple) ![Gemini](https://img.shields.io/badge/GenAI-Google%20Gemini-green) ![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)

## 🏥 Project Overview
**OsteoVision** is a comprehensive medical diagnostic system designed to automate the screening and reporting of bone fractures. Unlike standard detectors, OsteoVision employs an **Agentic Workflow** that validates inputs, verifies fracture presence using a powerful ensemble model, and precisely localizes injuries before generating medical reports.

Developed as a final-year B.Tech project, this system integrates state-of-the-art vision transformers and Large Language Models to assist both doctors and patients.

## ⚙️ Key Features
* **Intelligent Input Validation:** Uses **CLIP** to ensure uploaded images are valid X-rays, filtering out irrelevant data instantly.
* **Ensemble Detection Pipeline:** A robust voting mechanism combining **ViT (Vision Transformer)**, **Swin Transformer**, and **MONAI (ResNet)** to accurately classify *if* a fracture exists.
* **Precise Localization:** If a fracture is confirmed, **YOLOv11** takes over to draw high-precision bounding boxes around the injury.
* **Automated Diagnosis Reports:** **Google Gemini** analyzes the visual findings to draft a professional, downloadable medical report.
* **"Dr. George" AI Assistant:** An interactive chatbot that helps patients understand severity, recovery, and medical concerns in simple language.
* **Fracture Encyclopedia:** A built-in educational module explaining fracture types and the internal working of the model.

## 🧠 System Architecture (Agentic Workflow)
The system follows a strict logical pipeline to minimize errors:

1.  **Input Agent (CLIP):** "Is this an X-ray?"
    * *No:* Reject input.
    * *Yes:* Pass to detection.
2.  **Detection Agent (Ensemble):** "Is there a fracture?"
    * Aggregates votes from **ViT**, **Swin**, and **MONAI**.
    * *No:* Report "Healthy Bone."
    * *Yes:* Pass to localization.
3.  **Localization Agent (YOLOv11):** "Where is it?"
    * Draws bounding boxes on the specific fracture site.
4.  **Reporting Agent (Gemini):**
    * Synthesizes all data into a clinical report.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Input Validation:** OpenAI CLIP
* **Classification Ensemble:** Vision Transformer (ViT), Swin Transformer, MONAI (ResNet)
* **Localization Model:** YOLOv11 (Ultralytics)
* **Report & Chat Generation:** Google Gemini API
* **Dataset Source:** FracAtlas
* **Language:** Python

## 🚀 Setup & Installation

**Prerequisites:**
* Python 3.8+
* Google API Key (for Gemini)

**Installation:**
```bash
# Clone the repository
git clone [https://github.com/Guruvesh3/osteovision.git](https://github.com/Guruvesh3/osteovision.git)
cd osteovision

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
