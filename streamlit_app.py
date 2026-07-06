"""
Streamlit frontend for the Repository-Specific Bug Classification System.

This is a thin client — all inference logic lives in the FastAPI backend
(app.py). Run the backend first, then this app.
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="Bug Classifier",
    page_icon="🐛",
    layout="centered",
)

st.title("🐛 Repository-Specific Bug Classifier")
st.caption(
    "Predicts whether a GitHub issue is a **Bug** or **Not Bug**, using a "
    "domain-specific specialist model trained on that repository's own "
    "vocabulary and reporting style."
)

st.divider()

repository = st.selectbox(
    "Repository",
    options=["Machine Learning", "JavaScript"],
    help="Select the domain this issue belongs to. Each domain is served "
         "by its own independently trained model.",
)

title = st.text_input("Title", placeholder="e.g. RuntimeError: CUDA out of memory")

body = st.text_area(
    "Body",
    placeholder="Paste the full issue description here...",
    height=200,
)

predict_clicked = st.button("Predict", type="primary", use_container_width=True)

if predict_clicked:
    if not title.strip() and not body.strip():
        st.warning("Please enter a title or body before predicting.")
    else:
        with st.spinner("Running inference..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"repository": repository, "title": title, "body": body},
                    timeout=15,
                )
                response.raise_for_status()
                result = response.json()

                st.divider()

                if result["prediction"] == "Bug":
                    st.error(f"### Prediction: {result['prediction']} 🐛")
                else:
                    st.success(f"### Prediction: {result['prediction']} ✅")

                col1, col2, col3 = st.columns(3)
                col1.metric("Confidence", f"{result['confidence']}%")
                col2.metric("Bug Probability", f"{result['bug_probability']}%")
                col3.metric("Not Bug Probability", f"{result['not_bug_probability']}%")

                st.progress(result["bug_probability"] / 100)

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to the backend API. "
                    "Make sure it's running: `uvicorn app:app --reload`"
                )
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e.response.json().get('detail', str(e))}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

st.divider()
st.caption(
    "Note: repository selection is currently manual (hardcoded routing). "
    "Automatic domain detection from issue text is planned as future work."
)