# Image Recognition App

A Streamlit app for Project 2 (Image Recognition) from
`AI_Playground_4_Real_World_AI_Projects_v4.ipynb`.

Classifies any photo into one of 1,000 ImageNet categories using
**transfer learning** with a pretrained **MobileNetV2** network — no
training required.

## Folder structure

```
image_recognition_app/
├── app.py                  # Streamlit UI — entry point
├── model/
│   ├── __init__.py         # exposes load_model, prepare_image, predict_image
│   └── classifier.py       # MobileNetV2 loading, preprocessing, inference
├── requirements.txt
└── README.md
```

## Setup

```bash
cd image_recognition_app
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The first run downloads ~14 MB of
pretrained MobileNetV2 weights — this happens once and is then cached
by Keras.

## How it works

1. **Model** (`model/classifier.py`) — loads `MobileNetV2(weights="imagenet")`,
   a network already trained on over a million images across 1,000
   object categories.
2. **Preprocessing** — any uploaded image is resized to 224×224, converted
   to a numeric array, wrapped in a batch of size 1, and scaled with
   `preprocess_input` to match what the network saw during training.
3. **Inference** — the image is passed through the network, producing
   1,000 probability scores; `decode_predictions` converts these into
   the top 5 most likely object names.
4. **App** (`app.py`) — lets you upload a photo (or use a bundled sample
   image if you skip that), then displays the top predictions as a bar
   chart along with inference time.

## Notes

- A high confidence score reflects how strongly the pattern matched
  something learned during training — not a guarantee of correctness.
- Try photos with blurry focus or multiple objects in frame to see how
  the predictions change.

## Extending

- Swap in a different pretrained network (e.g. `ResNet50`, `EfficientNetB0`).
- Fine-tune the model on your own labeled images instead of using raw
  ImageNet classes.
- Add webcam capture with `st.camera_input()` for live classification.
