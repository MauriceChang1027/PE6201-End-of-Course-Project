import os

import streamlit as st
from PIL import Image

from retinaguard import OpenRouterClient, VisionClassifier, apply_triage_rules
from retinaguard.config import CONFIDENCE_THRESHOLD, GRADE_LABELS, OPENROUTER_MODEL


@st.cache_resource
def load_classifier() -> VisionClassifier:
    return VisionClassifier().load()


st.set_page_config(page_title="RetinaGuard", page_icon="👁️", layout="centered")
st.title("RetinaGuard")
st.caption("Diabetic retinopathy referral triage prototype")
st.warning(
    "Research prototype only. It is not a diagnostic device and every result requires licensed clinician review."
)

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "OpenRouter API key",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        type="password",
    )
    model_name = st.text_input("OpenRouter model", value=OPENROUTER_MODEL)
    patient_reference = st.text_input("Demo patient reference", value="DEMO-001")
    st.caption("The API key is used for this session only and is not stored by the app.")

uploaded_file = st.file_uploader("Upload a colour fundus image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded fundus image", use_container_width=True)

    if st.button("Analyse image", type="primary"):
        with st.spinner("Loading the vision model and analysing the image..."):
            prediction = load_classifier().predict(image)
            result = apply_triage_rules(prediction)
        st.session_state["triage_result"] = result

result = st.session_state.get("triage_result")
if result:
    st.subheader("Triage result")
    left, right = st.columns(2)
    left.metric("Automated grade", f"Grade {result.prediction.grade}")
    right.metric("Confidence", f"{result.prediction.confidence:.1%}")
    st.write(f"**Finding:** {result.prediction.label}")
    st.write(f"**Urgency:** {result.urgency}")
    st.write(f"**Action:** {result.action}")

    probability_data = [
        {
            "Grade": f"Grade {index}: {label}",
            "Probability": probability,
        }
        for index, (label, probability) in enumerate(zip(GRADE_LABELS, result.prediction.probabilities))
    ]
    st.bar_chart(probability_data, x="Grade", y="Probability", horizontal=True)

    if result.abstained:
        st.error(
            f"The confidence is below the {CONFIDENCE_THRESHOLD:.0%} threshold. "
            "RetinaGuard abstained and will not generate a routine referral conclusion."
        )
    else:
        clinician_confirmed = st.checkbox(
            "I confirm that a licensed clinician will review this draft before use."
        )
        if st.button("Generate referral draft", disabled=not clinician_confirmed):
            if not api_key:
                st.error("Enter an OpenRouter API key in the sidebar.")
            else:
                with st.spinner("Generating a constrained referral draft..."):
                    client = OpenRouterClient(api_key=api_key, model=model_name)
                    draft = client.draft_referral(result, patient_reference)
                st.subheader("Referral draft")
                st.write(draft)
                st.info("Draft only. Verify every statement before copying it into a clinical record.")
