"""
Image Recognition — Streamlit App
Project 2 from AI_Playground_4_Real_World_AI_Projects_v4.ipynb

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import time
import urllib.request

import pandas as pd
import streamlit as st
from PIL import Image

from model import load_model, predict_image

FALLBACK_IMAGE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg"
FALLBACK_IMAGE_PATH = "sample_image.jpg"

st.set_page_config(page_title="Image Recognition", page_icon="🖼️", layout="centered")

cached_load_model = st.cache_resource(load_model)


def get_fallback_image() -> Image.Image:
    if not os.path.exists(FALLBACK_IMAGE_PATH):
        urllib.request.urlretrieve(FALLBACK_IMAGE_URL, FALLBACK_IMAGE_PATH)
    return Image.open(FALLBACK_IMAGE_PATH)


st.title("🖼️ Image Recognition")
st.caption("Project 2 — Transfer learning with MobileNetV2, pretrained on ImageNet (1,000 categories)")

with st.spinner("Loading MobileNetV2 (first run downloads ~14 MB of weights)..."):
    model = cached_load_model()

st.subheader("Provide an image")
uploaded_file = st.file_uploader("Upload a photo (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    source_label = uploaded_file.name
else:
    st.info("No image uploaded — using a sample image. Upload your own above to try it.")
    image = get_fallback_image()
    source_label = "sample_image.jpg"

st.image(image, caption=f"Input image ({source_label})", use_container_width=True)

if st.button("Classify Image", type="primary"):
    with st.spinner("Running inference..."):
        decoded, elapsed_ms = predict_image(model, image, top=5)

    st.success(f"Prediction took {elapsed_ms:.1f} ms")

    st.subheader("Top predictions")
    results_df = pd.DataFrame(decoded, columns=["label", "probability"])
    results_df["probability (%)"] = (results_df["probability"] * 100).round(2)

    top_label, top_prob = decoded[0]
    st.metric("Best guess", top_label.replace("_", " ").title(), f"{top_prob * 100:.1f}% confidence")

    st.bar_chart(
        results_df.set_index("label")["probability (%)"],
        horizontal=True,
    )

    with st.expander("Raw prediction table"):
        st.dataframe(results_df[["label", "probability (%)"]])

st.divider()

with st.expander("What actually happened here?"):
    st.write(
        "The model did not 'see' the image the way a human does. It processed "
        "a grid of numbers through many mathematical layers, each transforming "
        "those numbers slightly, until the final layer produced a probability "
        "for each of the 1,000 categories it was trained to recognize. A high "
        "score reflects how strongly the pattern matched something learned "
        "during training — not a guarantee of correctness."
    )

with st.expander("Try these"):
    st.markdown(
        "- A dog\n"
        "- A cat\n"
        "- A car\n"
        "- A bird\n"
        "- Food\n"
        "- A bottle\n\n"
        "Then consider: what happens with a blurry photo, or one with "
        "multiple objects in the frame? Does the highest confidence score "
        "guarantee the prediction is correct?"
    )
