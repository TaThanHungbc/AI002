from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
import os

from preprocess import preprocess_single_image

MODEL_FILE = 'handwriting_detector.keras'
BEST_MODEL_FILE = 'best_handwriting_detector.keras'

print("Loading model...")
path = BEST_MODEL_FILE if os.path.exists(BEST_MODEL_FILE) else MODEL_FILE
model = tf.keras.models.load_model(path)
print("Model loaded!")

app = FastAPI(title="Handwriting Detector API")

# Allow access from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    arr = preprocess_single_image(image)  # Output shape: (1,128,128,1)

    prob = float(model.predict(arr)[0][0])
    label = "AI" if prob > 0.5 else "Human"

    return {
        "label": label,
        "confidence": round(prob if label == "AI" else 1 - prob, 4)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8500)
