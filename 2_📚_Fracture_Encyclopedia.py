import streamlit as st

st.set_page_config(page_title="Fracture Encyclopedia", page_icon="📚", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
    :root {
        --primary-color: #0068c9;
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
    .stImage img {
        border-radius: 8px;
        box-shadow: var(--box-shadow);
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 Fracture Encyclopedia")
st.markdown("A reference guide to common types of bone fractures.")

st.info("This information is for educational purposes only and not a substitute for professional medical diagnosis.")

# --- Fracture Types ---

st.markdown('<div class="card">', unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("1. Stable (Hairline or Stress) Fracture")
    st.markdown("""
    A hairline or stress fracture is a small crack or severe bruise within a bone. This injury is most common in athletes, especially those in sports that involve repetitive running or jumping. 
    * **Description:** A tiny, slender crack in the bone.
    * **Cause:** Repetitive force, overuse, or suddenly increasing the intensity of an activity.
    * **Commonly Affects:** Weight-bearing bones, such as the tibia (shin bone) or the bones of the foot (metatarsals).
    """)
with col2:
    st.image("https://placehold.co/400x300/e6f0fa/0068c9?text=Hairline+Fracture+Diagram", caption="Illu. of a hairline fracture")
st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="card">', unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("2. Oblique Fracture")
    st.markdown("""
    An oblique fracture is a break that is diagonal across the bone, at an angle.
    * **Description:** The break is at an angle relative to the bone's long axis.
    * **Cause:** Typically caused by a sharp, angled blow or a fall where force is applied at an angle.
    * **Commonly Affects:** Long bones such as the femur (thigh), tibia (shin), and humerus (upper arm).
    """)
with col2:
    st.image("https://placehold.co/400x300/e6f0fa/0068c9?text=Oblique+Fracture+Diagram", caption="Illu. of an oblique fracture")
st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="card">', unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("3. Comminuted Fracture")
    st.markdown("""
    A comminuted fracture is a severe type of break where the bone shatters into three or more pieces.
    * **Description:** The bone is broken or splintered into multiple fragments.
    * **Cause:** This fracture is almost always caused by high-impact trauma, such as a severe fall, a car accident, or a crushing injury.
    * **Note:** These fractures often require surgery to realign the bone fragments.
    """)
with col2:
    st.image("https://placehold.co/400x300/e6f0fa/0068c9?text=Comminuted+Fracture", caption="Illu. of a comminuted fracture")
st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="card">', unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("4. Greenstick Fracture")
    st.markdown("""
    A greenstick fracture is an incomplete break in which the bone is bent. This type of fracture occurs most often in children. 
    * **Description:** The bone cracks on one side but does not break all the way through, similar to bending a "green" or fresh stick of wood.
    * **Cause:** Falls or blows. It is common in children because their bones are softer and more flexible than adult bones.
    * **Commonly Affects:** Long bones, such as the forearm (radius or ulna).
    """)
with col2:
    st.image("https://placehold.co/400x300/e6f0fa/0068c9?text=Greenstick+Fracture", caption="Illu. of a greenstick fracture")
st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("5. Open (Compound) Fracture - Professional Note")
st.warning("""
**This is a medical emergency.** An open fracture (or compound fracture) occurs when the broken bone breaks and pierces the skin. It requires immediate medical attention due to the high risk of severe infection.
""")
st.markdown('</div>', unsafe_allow_html=True)