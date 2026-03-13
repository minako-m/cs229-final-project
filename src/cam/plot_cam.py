"""
Documentation: https://jacobgil.github.io/pytorch-gradcam-book/introduction.html

Running grad CAM for examples on both datasets to understand what regions the 
model attends to when making predictions.
"""

import sys
sys.path.append("/Users/amiramahmedjan/Documents/cs229/final-project/src")
sys.path.append("/Users/amiramahmedjan/Documents/cs229/final-project/src")

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import Image as PILImage
import torchvision.transforms as transforms

from kazakh.kazakh_train import CRNN, KAZAKH_VOCAB, ctc_greedy_decode, cer
from french.french_train  import FRENCH_VOCAB

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


KAZAKH_MODEL   = "/Users/amiramahmedjan/Documents/cs229/final-project/src/kazakh/kazakh_best_crnn.pt"
FRENCH_MODEL   = "/Users/amiramahmedjan/Documents/cs229/final-project/src/french/best_french_crnn.pt"

KAZAKH_IMG_DIR = "/Users/amiramahmedjan/Documents/cs229/final-project/data/KOHTD_dataset/HK_dataset/img"
KAZAKH_ANN_DIR = "/Users/amiramahmedjan/Documents/cs229/final-project/data/KOHTD_dataset/HK_dataset/ann"

FRENCH_IMG_DIR = "/Users/amiramahmedjan/Documents/cs229/final-project/data/french_dataset/images"
FRENCH_ANN     =  "/Users/amiramahmedjan/Documents/cs229/final-project/data/french_dataset/image_label_pairs.json"

OUTPUT_DIR     = "cam_outputs"
N_SAMPLES      = 16    # how many examples to visualize per language
IMG_HEIGHT     = 32
DEVICE         = torch.device("mps")
os.makedirs(OUTPUT_DIR, exist_ok=True)



# ========== models loading =============
def load_kazakh_model():
    num_classes = len(KAZAKH_VOCAB) + 1
    model = CRNN(num_classes, rnn_hidden=128).to(DEVICE)
    model.load_state_dict(torch.load(KAZAKH_MODEL, map_location=DEVICE))
    model.eval()
    return model


def load_french_model():
    num_classes = len(FRENCH_VOCAB) + 1
    model = CRNN(num_classes, rnn_hidden=128).to(DEVICE)
    model.load_state_dict(torch.load(FRENCH_MODEL, map_location=DEVICE))
    model.eval()
    return model

def preprocess(img_path, width=128):
    """Load image and return tensor for model and numpy array for vizualizatn)"""
    image = Image.open(img_path).convert("L")
    image = image.resize((width, IMG_HEIGHT), Image.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    tensor = transform(image).unsqueeze(0)   # (1, 1, H, W)

    img_np = np.array(image, dtype=np.float32) / 255.0
    img_rgb = np.stack([img_np] * 3, axis=-1)
    return tensor, img_rgb

def predict(model, input_tensor, idx2char, device):
    """Run inference and return predicted string for the tebsor"""
    model.eval()
    with torch.no_grad():
        log_probs = model(input_tensor.to(device))
    preds = ctc_greedy_decode(log_probs.cpu(), idx2char)
    return preds[0]

# =============== samples loading =============
def get_kazakh_samples(n=N_SAMPLES):
    import json, random
    char2idx = {c: i+1 for i, c in enumerate(KAZAKH_VOCAB)}
    char2idx["<blank>"] = 0
    idx2char  = {v: k for k, v in char2idx.items()}

    samples = []
    files = [f for f in os.listdir(KAZAKH_ANN_DIR) if f.endswith(".json")]
    random.seed(47)
    random.shuffle(files)

    for fname in files:
        if len(samples) >= n * 3:   # get extra so we can find errors too
            break
        with open(os.path.join(KAZAKH_ANN_DIR, fname), encoding="utf-8") as f:
            data = json.load(f)
        img_path = os.path.join(KAZAKH_IMG_DIR, data["name"])
        if not os.path.exists(img_path):
            continue
        label = data.get("description", "").strip()
        if not label:
            continue
        samples.append((img_path, label, idx2char))
    return samples


def get_french_samples(n=N_SAMPLES):
    import json, random
    char2idx = {c: i+1 for i, c in enumerate(FRENCH_VOCAB)}
    char2idx["<blank>"] = 0
    idx2char  = {v: k for k, v in char2idx.items()}

    with open(FRENCH_ANN, encoding="utf-8") as f:
        raw = json.load(f)

    random.seed(47)
    random.shuffle(raw)

    samples = []
    for entry in raw:
        if len(samples) >= n * 3:
            break
        img_path = os.path.join(FRENCH_IMG_DIR, entry["image"])
        label    = entry.get("label", "").strip()
        if not os.path.exists(img_path) or not label:
            continue
        samples.append((img_path, label, idx2char))
    return samples

# =============== Gradcam functions ===================
class SumTarget:
    """Scalar target: sum all logits"""
    def __call__(self, output):
        return output.sum()

def run_gradcam(model, input_tensor, target_layer, device):
    """Run Grad-CAM and return heatmap array"""

    # move temporarily to cpu for CAM computation
    model_cpu = model.to("cpu")
    tensor_cpu = input_tensor.to("cpu")

    cam = GradCAM(model=model_cpu, target_layers=[target_layer], reshape_transform=None)
    grayscale_cam = cam(input_tensor=tensor_cpu, targets=[SumTarget()])

    # move model back to mps
    model.to(device)
    return grayscale_cam[0]

def visualize_cam(model, samples, target_layer, device, language, n=N_SAMPLES):
    """
    For each sample show original image + CAM overlay.
    Collect correct and incorrect predictions separately.

    For infromativeness, we will only plot incorrect examples with
    significantly high character error rate, calculated by CER (from kazakh_train.py file)
    """
    correct_examples = []
    error_examples   = []

    for img_path, true_label, idx2char in samples:
        if len(correct_examples) >= n//2 and len(error_examples) >= n//2:
            break

        # determine width since French sentences need wider images
        w = 512 if language == "French" else 128
        tensor, img_rgb = preprocess(img_path, width=w)
        pred = predict(model, tensor, idx2char, device)

        grayscale_cam = run_gradcam(model, tensor, target_layer, device)

        # resize CAM to match image
        cam_resized = np.array(
            PILImage.fromarray((grayscale_cam * 255).astype(np.uint8))
            .resize((img_rgb.shape[1], img_rgb.shape[0]), PILImage.LANCZOS)
        ) / 255.0

        overlay = show_cam_on_image(img_rgb, cam_resized, use_rgb=True)
        is_correct = (pred.strip() == true_label.strip())
        is_wrong_enough = cer(pred, true_label) > 0.3

        # is_correct = (pred.strip() == true_label.strip())     #this option looks for full match

        entry = (img_rgb, overlay, pred, true_label, is_correct)
        if is_correct and len(correct_examples) < n//2:
            correct_examples.append(entry)
        elif is_wrong_enough and len(error_examples) < n//2:
            error_examples.append(entry)

    # plot
    all_examples = correct_examples
    all_examples.extend(error_examples)

    fig, axes = plt.subplots(len(all_examples), 2, figsize=(14, 3 * len(all_examples)))
    fig.suptitle(f"{language} — Grad-CAM Visualizations", fontsize=14, y=1.01)

    if len(all_examples) == 1:
        axes = [axes]

    for i, (img_rgb, overlay, pred, true, is_correct) in enumerate(all_examples):
        status = "CORRECT" if is_correct else "ERROR"
        color  = "green" if is_correct else "red"

        axes[i][0].imshow(img_rgb, cmap="gray")
        axes[i][0].set_title(f"Original — True: '{true}'", fontsize=9)
        axes[i][0].axis("off")

        axes[i][1].imshow(overlay)
        axes[i][1].set_title(f"Grad-CAM [{status}] — Pred: '{pred}'",
                             fontsize=9, color=color)
        axes[i][1].axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"cam_results_{language.lower()}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# entry point
if __name__ == "__main__":
    print(f"Using device: {DEVICE}\n")

    kaz_model = load_kazakh_model()
    fre_model = load_french_model()

    # model.cnn[7] is our target layer (conv2d block 3), before CTC decode smashes dimentions
    kaz_layer = kaz_model.cnn[7]
    fre_layer = fre_model.cnn[7]

    print("Loading samples")
    kaz_samples = get_kazakh_samples(n=N_SAMPLES)
    fre_samples = get_french_samples(n=N_SAMPLES)
    print(f"  Kazakh: {len(kaz_samples)} samples")
    print(f"  French: {len(fre_samples)} samples\n")

    # ── Generate CAM figures ──────────────────────────────────────
    print("Generating Kazakh CAMs...")
    visualize_cam(kaz_model, kaz_samples, kaz_layer,
                  DEVICE, language="Kazakh")

    print("Generating French CAMs...")
    visualize_cam(fre_model, fre_samples, fre_layer,
                  DEVICE, language="French")

    print("Files generated:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"  {f}")