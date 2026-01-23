# predict.py
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from preprocess import process_image_resize
from model import get_resnet18
import numpy as np

def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = get_resnet18(num_classes=2, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def predict_image(model, img_path, device):
    # preprocess
    img_np, lap_var = process_image_resize(img_path, target_size=(224,224), debug=False)
    # convert to PIL, normalize
    img_pil = Image.fromarray(img_np)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    x = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
    return pred, probs, lap_var

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image to predict")
    parser.add_argument("--model", type=str, default="best_model.pth", help="Path to saved model .pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    pred, probs, lap_var = predict_image(model, args.image, device)
    label = "AI-generated" if pred==1 else "Human-written"
    print(f"Prediction: {label}")
    print(f"Probabilities: Human={probs[0]:.4f}, AI={probs[1]:.4f}")
    print(f"Laplacian variance (blur proxy): {lap_var:.2f}")
    if lap_var < 100:
        print("Warning: Image might be too blurry (Laplacian variance < 100). Results may be unreliable.")
