# gradio_app.py
import gradio as gr
import torch
from predict import load_model, predict_image

MODEL_PATH = "best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(MODEL_PATH, device)

def classify_image(img):  # img: PIL image or numpy array provided by gradio
    # Gradio passes path-like or PIL image; predict_image expects path, so save temp
    import tempfile, os
    from PIL import Image
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp_path = tf.name
    img.save(tmp_path)
    try:
        pred, probs, lap_var = predict_image(model, tmp_path, device)
        label = "AI-generated" if pred==1 else "Human-written"
        text = f"{label}\nHuman={probs[0]:.3f}, AI={probs[1]:.3f}\nLaplacian variance={lap_var:.2f}"
    finally:
        os.remove(tmp_path)
    return text

iface = gr.Interface(fn=classify_image,
                     inputs=gr.Image(type="pil"),
                     outputs="text",
                     title="Handwriting: Human vs AI",
                     description="Upload an image of a handwritten page")
if __name__ == "__main__":
    iface.launch(share=False)  # set share=True to get public link
