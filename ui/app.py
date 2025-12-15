import gradio as gr
import requests

API_URL = "http://localhost:8500/predict"

def classify(img):
    files = {'file': ('upload.png', img)}
    res = requests.post(API_URL, files=files)
    out = res.json()
    return f"{out['label']} ({out['confidence']*100:.2f}%)"

gr.Interface(
    fn=classify,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="AI Handwriting Detector",
    flagging_mode="never"
).launch()
