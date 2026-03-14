import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import string

def main(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        all_preds = data['preds']
        all_targets = data['targets']
    except FileNotFoundError:
        print(f"{json_path} not found. Please provide a valid JSON file.")
        return

    # Function to normalize sentences by removing punctuation and spaces
    def normalize(s):
        return ''.join(c for c in s if c not in string.punctuation and not c.isspace())

    norm_targets = [normalize(t) for t in all_targets]
    norm_preds = [normalize(p) for p in all_preds]
    lengths = [len(t.split()) for t in all_targets]
    correct = [int(p == t) for p, t in zip(norm_preds, norm_targets)]

    bins = np.arange(1, max(lengths)+2)
    acc_per_len = []
    for b in bins:
        idx = [i for i, l in enumerate(lengths) if l == b]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
        else:
            acc = np.nan
        acc_per_len.append(acc)

    plt.figure(figsize=(8,4))
    plt.plot(bins, acc_per_len, marker='o')
    plt.xlabel('Sentence Length (words)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Sentence Length (French)')
    plt.grid(True)
    # Save the plot
    save_dir = r"C:\Users\Stephanie\OneDrive\Documents\Stanford 25-26\Winter '26\CS229\final_project\cs229-final-project\plots\french_metrics_plots_adjusted_accuracy"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "accuracy_vs_word_length_french.png")
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python accuracy_vs_word_length.py <path_to_json>")
        sys.exit(1)
    main(sys.argv[1])
