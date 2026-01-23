import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from preprocess import process_image_resize
from model import get_resnet18
from features import analyze_image_for_evidence


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = get_resnet18(num_classes=2, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def model_predict(model, img_np, device):
    img_pil = Image.fromarray(img_np)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    x = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
    return pred, float(probs[0]), float(probs[1])


def predict_and_explain(model_path, image_path, device):
    # ===== 1) MODEL DECIDES LABEL =====
    model = load_model(model_path, device)
    img_np, _ = process_image_resize(image_path, target_size=(224, 224), debug=False)
    model_pred, p_human, p_ai = model_predict(model, img_np, device)

    final_label = "AI" if model_pred == 1 else "Human"

    # ===== 2) RULES ONLY FOR EXPLANATION =====
    evidence = analyze_image_for_evidence(image_path)

    explanations = []

    if evidence['ssim']['flag']:
        n_pairs = evidence['ssim'].get('n_pairs', 0)
        if n_pairs >= 10:
            explanations.append(f"Phát hiện {n_pairs} cặp ký tự có hình thái giống hệt nhau.")

    if evidence['spacing']['flag']:
        explanations.append("Khoảng cách giữa các ký tự và dòng rất đều nhau bất thường.")

    if evidence['stroke']['flag']:
        explanations.append("Độ dày nét gần như không thay đổi, thiếu biến thiên lực bút tự nhiên.")

    if evidence['baseline']['flag']:
        explanations.append("Chân các ký tự thẳng hàng bất thường như được căn chỉnh tự động.")

    if evidence['continuity']['flag']:
        s = evidence['continuity']['n_bad']
        fb = ""
        if s >= 200:
            if s < 300:
                fb = "hơi có dấu hiệu là AI"
            elif s < 400:
                fb = "nghi ngờ là AI"
            else:
                fb = "khá chắc chắn là AI"
            explanations.append(f"Phát hiện {s} điểm nối nét bất thường giữa các ký tự ({fb}).")

    if len(explanations) == 0 or final_label == "Human":
        explanations = ["Không phát hiện đặc trưng hình học bất thường rõ ràng theo 5 tiêu chí."]

    return final_label, explanations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image to predict")
    parser.add_argument("--model", type=str, default="best_model.pth", help="Path to saved model .pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_label, explanations = predict_and_explain(args.model, args.image, device)

    print(final_label)
    print("Detected evidence:")
    for ex in explanations:
        print("- " + ex)
