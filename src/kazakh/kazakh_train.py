import os
import json
import time
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# =========== local imports ==========
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model import CRNN

CONFIG = {
    "image_dir"    : "/Users/amiramahmedjan/Documents/cs229/final-project/data/KOHTD_dataset/HK_dataset/img",
    "annotation_dir": "/Users/amiramahmedjan/Documents/cs229/final-project/data/KOHTD_dataset/HK_dataset/ann",
    "img_height"   : 32,
    "batch_size"   : 512,
    "lr"           : 0.001,
    "epochs"       : 50,
    "rnn_hidden"   : 128,
    "rnn_layers"   : 2,
    "checkpoint_every": 5,
}

SAMPLE_TEST   = False    # True to run on a smaller dataset
SAMPLE_N      = 1000
KAZAKH_VOCAB = sorted(list(set(
    # Kazakh Cyrillic
    'АаӘәБбВвГгҒғДдЕеЁёЖжЗзИиЙйКкҚқЛлМмНнҢңОоӨөПпРрСсТтУуҰұҮүФфХхҺһЦцЧчШшЩщЪъЫыІіЬьЭэЮюЯя'
    # digits
    '0123456789'
    # punctuation
    ' .,!?-:;()"\'«»'
)))

class KazakhOCRDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, vocab=KAZAKH_VOCAB, img_height=32):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.img_height = img_height

        # load labels
        self.samples = []
        skipped = 0
        for file in os.listdir(annotation_dir):
            if not file.endswith(".json"):
                continue
            path = os.path.join(annotation_dir, file)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            name = data.get("name", "")
            desc = data.get("description", "")
            img_path = os.path.join(image_dir, name)
            if not os.path.exists(img_path):          # skip missing images
                skipped += 1
                continue
            if len(desc) == 0:                        # skip empty labels
                skipped += 1
                continue
            desc = desc.strip().replace('\n', '').replace('\r', '').replace('\t', '')
            if len(desc) > 0:
                self.samples.append((name, desc))

        if skipped:
            print(f"Skipped {skipped} samples (missing image or empty label)")

        self.vocab = vocab

        # index 0 is reserved for CTC blank
        self.char2idx = {char: idx + 1 for idx, char in enumerate(self.vocab)}
        self.char2idx["<blank>"] = 0
        self.idx2char = {v: k for k, v in self.char2idx.items()}

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((img_height, img_height * 4)),   # 32 × 128
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    
    def encode_text(self, text):
        # clean text before encoding
        text = text.strip()     # remove leading/trailing whitespace
        text = re.sub(r'[\n\r\t]', '', text)    # remove newlines
        return torch.tensor(
            [self.char2idx[c] for c in text if c in self.char2idx],
            dtype=torch.long
        )
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, text = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("L")
        image = self.transform(image)
        label = self.encode_text(text)
        return image, label
    

def collate_fn(batch):
    """Stack images; pad labels to the longest in the batch."""
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_padded = torch.zeros(len(labels), int(label_lengths.max()), dtype=torch.long)
    for i, l in enumerate(labels):
        labels_padded[i, : len(l)] = l
    return images, labels_padded, label_lengths
    

# ================= Decoding ==================    
def ctc_greedy_decode(log_probs, idx2char, blank=0):
    """Greedy best-path decode (choose character with highest
    probability). Returns list of strings."""
    pred_indices = log_probs.argmax(2).permute(1, 0)   # (B, T)
    results = []
    for seq in pred_indices:
        chars, prev = [], -1
        for idx in seq.tolist():
            if idx != prev and idx != blank:
                chars.append(idx2char.get(idx, ""))
            prev = idx
        results.append("".join(chars))
    return results


# ================= Metrics ==================   
def word_accuracy(preds, targets):
    correct = sum(p == t for p, t in zip(preds, targets))
    return correct / max(len(targets), 1)


def cer(preds, targets):
    """Character Error Rate via Levenshtein distance."""
    total_dist = total_len = 0
    for p, t in zip(preds, targets):
        m, n = len(p), len(t)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            new_dp = [i] + [0] * n
            for j in range(1, n + 1):
                cost = 0 if p[i - 1] == t[j - 1] else 1
                new_dp[j] = min(new_dp[j - 1] + 1, dp[j] + 1, dp[j - 1] + cost)
            dp = new_dp
        total_dist += dp[n]
        total_len  += n
    return total_dist / max(total_len, 1)
    
# ============== Training ==============
def train(config):
    device = torch.device("mps")

    full_dataset = KazakhOCRDataset(
        config["image_dir"],
        config["annotation_dir"],
        img_height=config["img_height"],
    )
    print(f"Total samples loaded: {len(full_dataset)}")
    print(f"Vocabulary size: {len(full_dataset.vocab)}")

    if SAMPLE_TEST:
        print(f"SAMPLE TEST MODE — using {SAMPLE_N} samples only\n")
        full_dataset = Subset(full_dataset, indices=range(min(SAMPLE_N, len(full_dataset))))

    n_total = len(full_dataset)
    n_test  = max(1, int(n_total * 0.1))
    n_val   = max(1, int(n_total * 0.1))
    n_train = n_total - n_test - n_val

    train_set, val_set, _ = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {n_train}  |  Val: {n_val}  |  Test: {n_test}\n")

    num_workers = 0 if device.type == "mps" else 4
    train_loader = DataLoader(train_set, batch_size=config["batch_size"],
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=(device.type == "cuda"),
                              collate_fn=collate_fn)
    val_loader  = DataLoader(val_set,  batch_size=config["batch_size"],
                              shuffle=False, num_workers=num_workers,
                              pin_memory=(device.type == "cuda"),
                              collate_fn=collate_fn)

    # Model 
    # Grab vocab from the underlying dataset even when wrapped in Subset
    base_dataset = full_dataset.dataset if isinstance(full_dataset, Subset) else full_dataset
    num_classes  = len(base_dataset.vocab) + 1   # +1 for CTC blank

    model = CRNN(num_classes, config["rnn_hidden"], config["rnn_layers"]).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    start_epoch  = 1
    history      = []
    best_word_acc = 0.0

    # Epoch loop 
    for epoch in range(start_epoch, config["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for images, labels, label_lengths in tqdm(
            train_loader, desc=f"Epoch {epoch:03d}/{config['epochs']}", leave=False
        ):
            images = images.to(device)

            log_probs = model(images)                         # (T, B, C)
            T = log_probs.size(0)
            input_lengths = torch.full((images.size(0),), T, dtype=torch.long)

            # Flatten padded labels
            flat_labels = torch.cat([
                labels[i, : label_lengths[i]] for i in range(len(labels))
            ])

            # CTCLoss must run on CPU (MPS doesn't support it; safest everywhere)
            loss = criterion(
                log_probs.log_softmax(2).cpu(),
                flat_labels.cpu(),
                input_lengths.cpu(),
                label_lengths.cpu(),
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss   = epoch_loss / len(train_loader)
        epoch_mins = (time.time() - t0) / 60

        # Evaluate
        w_acc, c_err = 0.0, 1.0  # defaults
        if epoch % config["checkpoint_every"] == 0 or epoch == config["epochs"]:
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for images, labels, label_lengths in val_loader:
                    images = images.to(device)
                    log_probs = model(images)
                    preds = ctc_greedy_decode(log_probs.cpu(), base_dataset.idx2char)
                    targets = [
                        "".join(
                            base_dataset.idx2char.get(labels[i, j].item(), "")
                            for j in range(label_lengths[i])
                        )
                        for i in range(len(labels))
                    ]
                    all_preds.extend(preds)
                    all_targets.extend(targets)
            w_acc = word_accuracy(all_preds, all_targets)
            c_err = cer(all_preds, all_targets)

        print(
            f"Epoch {epoch:03d} | Loss {avg_loss:.4f} | "
            + (f"Word Acc {w_acc*100:.1f}% | CER {c_err*100:.2f}%" if epoch % config["checkpoint_every"] == 0 else "eval skipped")
            + f" | {epoch_mins:.1f} min"
        )
        scheduler.step(avg_loss)

        entry = {"epoch": epoch, "loss": avg_loss, "word_acc": w_acc, "cer": c_err}
        history.append(entry)
        with open("kazakh_training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Save best model
        if w_acc > best_word_acc:
            best_word_acc = w_acc
            torch.save(model.state_dict(), "kazakh_best_crnn.pt")
            print(f"  ✓ kazakh_best_crnn.pt saved  ({w_acc*100:.1f}%)")

        # Periodic checkpoint
        if epoch % config["checkpoint_every"] == 0:
            ckpt_path = f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save({
                "epoch"         : epoch,
                "model_state"   : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history"       : history,
                "best_word_acc" : best_word_acc,
            }, ckpt_path)
            print(f"  💾 Checkpoint saved → {ckpt_path}")

    print(f"\nTraining complete.  Best word accuracy: {best_word_acc*100:.1f}%")
    print("Best model weights saved to: best_crnn.pt")
    return history

# ============== Entry point ================
if __name__ == "__main__":
    history = train(CONFIG)