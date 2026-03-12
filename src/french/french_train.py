import os
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# Import model
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model import CRNN


CONFIG = {
    "image_dir"       : "/Users/amiramahmedjan/Documents/cs229/cs229-project/data/french_dataset/images",
    "annotation_file" : "/Users/amiramahmedjan/Documents/cs229/cs229-project/data/french_dataset/image_label_pairs.json",   # single JSON file (list of records)
    "img_height"      : 32,
    "batch_size"      : 64,
    "lr"              : 0.001,
    "epochs"          : 50,
    "rnn_hidden"      : 128,
    "rnn_layers"      : 2,
    "checkpoint_every": 1,
}

SAMPLES_TEST  = False
SAMPLES_N     = 1000

# French vocabulary
FRENCH_VOCAB = sorted(list(set(
    # lowercase + uppercase latin
    'abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # French accented characters
    'àâäéèêëîïôöùûüÿçœæ'
    'ÀÂÄÉÈÊËÎÏÔÖÙÛÜŸÇŒÆ'
    # digits
    '0123456789'
    # punctuation
    ' .,!?-:;()\'"«»'
)))


# ======== Dataset ===============
class FrenchOCRDataset(Dataset):
    def __init__(self, image_dir, annotation_file, img_height=32):
        self.image_dir  = image_dir
        self.img_height = img_height
        self.vocab      = FRENCH_VOCAB

        # index 0 reserved for CTC blank
        self.char2idx = {char: idx + 1 for idx, char in enumerate(self.vocab)}
        self.char2idx["<blank>"] = 0
        self.idx2char = {v: k for k, v in self.char2idx.items()}

        with open(annotation_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.samples = []
        skipped = 0
        for entry in raw:
            name  = entry.get("image", "")
            label = entry.get("label", "").strip()
            img_path = os.path.join(image_dir, name)
            if not os.path.exists(img_path) or len(label) == 0:
                skipped += 1
                continue
            # Keep only chars in vocab
            label = "".join(c for c in label if c in self.char2idx)
            if len(label) == 0:
                skipped += 1
                continue
            self.samples.append((name, label))

        if skipped:
            print(f"Skipped {skipped} samples")
        print(f"Loaded {len(self.samples)} French samples")

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            # transforms.Resize((img_height, img_height * 4)),   # 32 × 128
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def encode_text(self, text):
        return torch.tensor(
            [self.char2idx[c] for c in text if c in self.char2idx],
            dtype=torch.long
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, text = self.samples[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("L")
        
        # dealing with sentences: keep height = 32 and scale width accordingly
        w, h = image.size
        new_w = max(128, int(w * self.img_height / h))  # scale width proportionally
        new_w = min(new_w, 2048)    # cap at 2048 to avoid memory issues
        new_w = (new_w // 32) * 32      # Round to nearest 32 for efficiency
        image = image.resize((new_w, self.img_height), Image.LANCZOS)
        
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.5], [0.5])(image)
        label = self.encode_text(text)
        if len(label) == 0:
            return self.__getitem__((idx + 1) % len(self.samples))
        return image, label


def collate_fn(batch):
    images, labels = zip(*batch)
    
    # Pad images to the same width in this batch
    max_w = max(img.size(2) for img in images)
    padded_images = []
    for img in images:
        pad_w = max_w - img.size(2)
        # Pad on the right with -1 (normalized black)
        padded = torch.nn.functional.pad(img, (0, pad_w), value=-1.0)
        padded_images.append(padded)
    
    images = torch.stack(padded_images, 0)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_padded = torch.zeros(len(labels), int(label_lengths.max()), dtype=torch.long)
    for i, l in enumerate(labels):
        labels_padded[i, :len(l)] = l
    return images, labels_padded, label_lengths


# ===================== Metrics ========================
def ctc_greedy_decode(log_probs, idx2char, blank=0):
    pred_indices = log_probs.argmax(2).permute(1, 0)
    results = []
    for seq in pred_indices:
        chars, prev = [], -1
        for idx in seq.tolist():
            if idx != prev and idx != blank:
                chars.append(idx2char.get(idx, ""))
            prev = idx
        results.append("".join(chars))
    return results


def word_accuracy(preds, targets):
    """
    Split each predicted/target sentence into words.
    Accuracy = correct words / total words across all sentences.
    """
    correct = 0
    total   = 0
    for pred, target in zip(preds, targets):
        pred_words   = pred.split()
        target_words = target.split()
        total += len(target_words)
        for pw, tw in zip(pred_words, target_words):
            if pw == tw:
                correct += 1
    return correct / max(total, 1)


def sentence_accuracy(preds, targets):
    """Exact full-sentence match — kept for reference."""
    correct = sum(p == t for p, t in zip(preds, targets))
    return correct / max(len(targets), 1)


def cer(preds, targets):
    total_dist = total_len = 0
    for p, t in zip(preds, targets):
        m, n = len(p), len(t)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            new_dp = [i] + [0] * n
            for j in range(1, n + 1):
                cost = 0 if p[i-1] == t[j-1] else 1
                new_dp[j] = min(new_dp[j-1]+1, dp[j]+1, dp[j-1]+cost)
            dp = new_dp
        total_dist += dp[n]
        total_len  += n
    return total_dist / max(total_len, 1)


# ===================== Training ========================

def train(config):
    device = torch.device("mps")

    full_dataset = FrenchOCRDataset(
        config["image_dir"],
        config["annotation_file"],
        img_height=config["img_height"],
    )

    if SAMPLES_TEST:
        from torch.utils.data import Subset
        print(f"SAMPLE TEST — {SAMPLES_N} samples\n")
        full_dataset = Subset(full_dataset, indices=range(min(SAMPLES_N, len(full_dataset))))

    n_total = len(full_dataset)
    n_test  = max(1, int(n_total * 0.1))
    n_val   = max(1, int(n_total * 0.1))
    n_train = n_total - n_test - n_val

    train_set, val_set, _ = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {n_train}  |  Val: {n_val}  |  Test: {n_test}\n")


    base_dataset = full_dataset.dataset if hasattr(full_dataset, "dataset") else full_dataset
    num_workers  = 0 if device.type == "mps" else 4

    train_loader = DataLoader(train_set, batch_size=config["batch_size"],
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=(device.type == "cuda"),
                              collate_fn=collate_fn)
    val_loader  = DataLoader(val_set,  batch_size=config["batch_size"],
                              shuffle=False, num_workers=num_workers,
                              pin_memory=(device.type == "cuda"),
                              collate_fn=collate_fn)

    # model
    num_classes = len(FRENCH_VOCAB) + 1   # +1 for CTC blank
    model       = CRNN(num_classes, config["rnn_hidden"], config["rnn_layers"]).to(device)
    criterion   = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer   = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler   = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    start_epoch   = 1
    history       = []
    best_word_acc = 0.0

    # epoch loop
    for epoch in range(start_epoch, config["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for images, labels, label_lengths in tqdm(
            train_loader, desc=f"Epoch {epoch:03d}/{config['epochs']}", leave=False
        ):
            images = images.to(device)
            log_probs     = model(images)
            T             = log_probs.size(0)
            input_lengths = torch.full((images.size(0),), T, dtype=torch.long)
            flat_labels   = torch.cat([
                labels[i, :label_lengths[i]] for i in range(len(labels))
            ])

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

        # evaluate every
        w_acc = s_acc = c_err = 0.0
        if epoch % config["checkpoint_every"] == 0 or epoch == config["epochs"]:
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for images, labels, label_lengths in val_loader:
                    images    = images.to(device)
                    log_probs = model(images)
                    preds     = ctc_greedy_decode(log_probs.cpu(), base_dataset.idx2char)
                    targets   = [
                        "".join(base_dataset.idx2char.get(labels[i, j].item(), "")
                                for j in range(label_lengths[i]))
                        for i in range(len(labels))
                    ]
                    all_preds.extend(preds)
                    all_targets.extend(targets)

            w_acc = word_accuracy(all_preds, all_targets)       # word-level
            s_acc = sentence_accuracy(all_preds, all_targets)   # sentence-level
            c_err = cer(all_preds, all_targets)

        scheduler.step(avg_loss)

        if epoch % config["checkpoint_every"] == 0 or epoch == config["epochs"]:
            print(
                f"Epoch {epoch:03d} | Loss {avg_loss:.4f} | "
                f"Word Acc {w_acc*100:.1f}% | Sentence Acc {s_acc*100:.1f}% | "
                f"CER {c_err*100:.2f}% | {epoch_mins:.1f} min"
            )
        else:
            print(f"Epoch {epoch:03d} | Loss {avg_loss:.4f} | eval skipped | {epoch_mins:.1f} min")

        entry = {
            "epoch": epoch, "loss": avg_loss,
            "word_acc": w_acc, "sentence_acc": s_acc, "cer": c_err
        }
        history.append(entry)
        with open("french_training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        if w_acc > best_word_acc:
            best_word_acc = w_acc
            torch.save(model.state_dict(), "best_french_crnn.pt")
            print(f"french_best_crnn.pt saved ({w_acc*100:.1f}%)")

        if epoch % config["checkpoint_every"] == 0:
            ckpt_path = f"french_checkpoint_epoch_{epoch:03d}.pt"
            torch.save({
                "epoch"          : epoch,
                "model_state"    : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history"        : history,
                "best_word_acc"  : best_word_acc,
            }, ckpt_path)
            print(f"    Checkpoint saved → {ckpt_path}")

    print(f"\nTraining complete. Best word accuracy: {best_word_acc*100:.1f}%")
    return history


# entry point
if __name__ == "__main__":
    history = train(CONFIG)