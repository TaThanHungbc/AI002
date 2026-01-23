# predict.py (mới)
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from preprocess import process_image_resize
from model import get_resnet18
from features import analyze_image_for_evidence
import numpy as np

def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = get_resnet18(num_classes=2, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def model_predict(model, img_np, device):
    # img_np: numpy RGB uint8 224x224
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
    return pred, float(probs[0]), float(probs[1])

def predict_and_explain(model_path, image_path, device):
    # 1) evidence extraction
    evidence = analyze_image_for_evidence(image_path)

    # count flagged criteria
    flags = []
    if evidence['ssim']['flag']:
        flags.append('Độ tương đồng tuyệt đối giữa các ký tự (SSIM)')
    if evidence['spacing']['flag']:
        flags.append('Độ đều của khoảng cách dòng & chữ (Spacing Regularity)')
    if evidence['stroke']['flag']:
        flags.append('Độ rung và biến thiên của nét bút (Stroke Variability)')
    if evidence['baseline']['flag']:
        flags.append('Độ thẳng hàng (Baseline alignment)')
    if evidence['continuity']['flag']:
        flags.append('Điểm nối nét bất thường (Continuity)')

    # 2) model prediction (only if needed)
    # load model
    model = load_model(model_path, device)
    # produce model input image (preprocessed)
    img_np, _ = process_image_resize(image_path, target_size=(224,224), debug=False)
    model_pred, p_human, p_ai = model_predict(model, img_np, device)

    # decision logic:
    # If >=2 evidence flags -> AI
    # Else use model prediction
    if len(flags) >= 2:
        final_label = "AI"
    else:
        final_label = "AI" if model_pred == 1 else "Human"

    # Prepare explanation lines
    explanations = []
    if evidence['ssim']['flag']:
        # Explain groups: list groups with counts
        for g in evidence['ssim']['groups']:
            explanations.append(f"Phát hiện {g['count']} ký tự có hình thái giống hệt (SSIM>0.98) ở gần y={int(g['line_y'])}.")
    if evidence['spacing']['flag']:
        explanations.append(f"Khoảng cách giữa các ký tự/dòng rất đều (CV={evidence['spacing']['cv']:.3f}, mean={evidence['spacing']['mean_distance']:.1f}px).")
    if evidence['stroke']['flag']:
        explanations.append(f"Độ dày nét không biến thiên (CV_w={evidence['stroke']['cv_w']:.3f}, std_I={evidence['stroke']['std_I']:.3f}, R={evidence['stroke']['R']:.3f}).")
    if evidence['baseline']['flag']:
        explanations.append(f"Chân ký tự nằm thẳng hàng (std_baseline={evidence['baseline']['std_baseline']:.3f}px).")
    if evidence['continuity']['flag']:
        explanations.append(f"Phát hiện {evidence['continuity']['n_bad']} điểm nối nét bất thường (continuity < 0.4).")

    # If explanations empty, add neutral message
    if len(explanations) == 0:
        explanations = ["Không phát hiện đặc trưng bất thường đủ mạnh theo 5 tiêu chí."]

    # Output only final label and satisfied criteria lines (Vietnamese)
    return final_label, explanations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image to predict")
    parser.add_argument("--model", type=str, default="best_model.pth", help="Path to saved model .pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_label, explanations = predict_and_explain(args.model, args.image, device)
    # print only label and criteria
    lab_text = "AI" if final_label=="AI" else "Human"
    print(lab_text)
    print("Detected evidence:")
    for ex in explanations:
        print("- " + ex)
