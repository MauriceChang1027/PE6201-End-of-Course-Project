import hashlib
import os

import streamlit as st
from PIL import Image

from retinaguard import GradCamError, OpenRouterClient, RetinaGuardPipeline
from retinaguard.config import GRADE_LABELS, OPENROUTER_MODEL, load_confidence_setting


@st.cache_resource
def load_pipeline(threshold: float) -> RetinaGuardPipeline:
    return RetinaGuardPipeline(confidence_threshold=threshold)


st.set_page_config(page_title="RetinaGuard", page_icon="👁️", layout="centered")
st.title("RetinaGuard")
st.caption("Diabetic retinopathy referral triage prototype")
st.warning(
    "Research prototype only. It is not a diagnostic device and every result "
    "requires licensed clinician review."
)
confidence_setting = load_confidence_setting()
st.caption(
    f"Confidence threshold: {confidence_setting.threshold:.0%} "
    f"({confidence_setting.source})."
)
if confidence_setting.target_met is False:
    st.warning("The APTOS evaluation did not meet the sensitivity and abstention targets.")

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

if uploaded_file is None:
    st.session_state.pop("analysis_context", None)
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("referral_draft", None)
    st.session_state.pop("referral_context", None)
    st.session_state.pop("explanation", None)
    st.session_state.pop("explanation_error", None)

if uploaded_file:
    upload_bytes = uploaded_file.getvalue()
    upload_digest = hashlib.sha256(upload_bytes).hexdigest()
    analysis_context = (upload_digest, confidence_setting.threshold)
    if st.session_state.get("analysis_context") != analysis_context:
        st.session_state["analysis_context"] = analysis_context
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("referral_draft", None)
        st.session_state.pop("explanation", None)
        st.session_state.pop("explanation_error", None)
    referral_context = (upload_digest, patient_reference, model_name)
    if st.session_state.get("referral_context") != referral_context:
        st.session_state.pop("referral_draft", None)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded fundus image", width="stretch")

    if st.button("Analyse image", type="primary"):
        pipeline = load_pipeline(confidence_setting.threshold)
        with st.spinner("Checking image quality and running the vision model..."):
            analysis = pipeline.analyse(image)
        st.session_state["analysis_result"] = analysis
        st.session_state.pop("referral_draft", None)
        st.session_state.pop("explanation", None)
        st.session_state.pop("explanation_error", None)
        if analysis.triage is not None:
            with st.spinner("Generating the Grad-CAM explanation..."):
                try:
                    st.session_state["explanation"] = pipeline.explain(image, analysis)
                except GradCamError as exc:
                    st.session_state["explanation_error"] = str(exc)

analysis = st.session_state.get("analysis_result")
if analysis:
    st.subheader("Image-quality gate")
    if analysis.quality.passed:
        st.success("The image passed the automated quality checks.")
    else:
        st.error("The image was rejected before vision-model inference.")
        for issue in analysis.quality.issues:
            st.write(f"- {issue.message}")
    with st.expander("Quality measurements"):
        st.json(analysis.quality.metrics)

result = analysis.triage if analysis else None
if result is not None:
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
        for index, (label, probability) in enumerate(
            zip(GRADE_LABELS, result.prediction.probabilities)
        )
    ]
    st.bar_chart(probability_data, x="Grade", y="Probability", horizontal=True)

    explanation = st.session_state.get("explanation")
    explanation_error = st.session_state.get("explanation_error")
    st.subheader("Model explanation")
    if explanation:
        heatmap_column, overlay_column = st.columns(2)
        heatmap_column.image(
            explanation.heatmap_image,
            caption="Grad-CAM heatmap",
            width="stretch",
        )
        overlay_column.image(
            explanation.overlay,
            caption="Heatmap overlay",
            width="stretch",
        )
        st.caption(
            f"Explains the Grade {explanation.class_index} score using layer "
            f"{explanation.layer_name}. Warmer yellow and red regions contributed "
            "more strongly."
        )
        st.warning(
            "Grad-CAM is a coarse attribution map, not lesion segmentation or proof that "
            "the highlighted region is clinically abnormal."
        )
    elif explanation_error:
        st.warning(f"Grad-CAM was unavailable: {explanation_error}")

    if result.abstained:
        st.error(
            f"The confidence is below the {confidence_setting.threshold:.0%} threshold. "
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
                    draft = client.generate_safe_referral(result, patient_reference)
                st.session_state["referral_draft"] = draft
                st.session_state["referral_context"] = referral_context

        draft = st.session_state.get("referral_draft")
        if draft:
            st.subheader("Referral draft")
            st.write(draft.text)
            if draft.validation_passed:
                st.success("LLM draft passed the structured safety validation.")
            else:
                st.warning(
                    "The LLM output was rejected. A deterministic safe fallback is shown."
                )
                with st.expander("Validation details"):
                    for issue in draft.validation_issues:
                        st.write(f"- {issue}")
            st.info(
                "Draft only. Verify every statement before copying it into a clinical record."
            )
