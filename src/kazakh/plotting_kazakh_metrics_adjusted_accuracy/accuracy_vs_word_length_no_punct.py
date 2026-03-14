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

    def normalize_string(s):
        return ''.join([c for c in s if c not in string.punctuation and not c.isspace()])

    # Calculate accuracy vs word length (ignoring punctuation and spaces)
    lengths = [len(normalize_string(t)) for t in all_targets]
    correct = [int(normalize_string(p) == normalize_string(t)) for p, t in zip(all_preds, all_targets)]

    if lengths:
        bins = np.arange(1, max(lengths) + 2)
        bin_acc = []
        for b in bins:
            idx = [i for i, l in enumerate(lengths) if l == b]
            if idx:
                acc = sum(correct[i] for i in idx) / len(idx)
            else:
                acc = np.nan
            bin_acc.append(acc)

        plt.figure(figsize=(10, 5))
        plt.plot(bins, bin_acc, marker='o')
        plt.xlabel('Word Length (no punct/space)')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Word Length (No Punctuation/Space)')
        plt.grid(True)
        save_dir = r"C:\Users\Stephanie\OneDrive\Documents\Stanford 25-26\Winter '26\CS229\final_project\cs229-final-project\plots\kazakh_metrics\adjusted_accuracy"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "accuracy_vs_word_length_no_punct.png")
        plt.savefig(save_path, dpi=200)
        plt.show()
        print(f"Plot saved to {save_path}")
    else:
        print("No data to plot. Please ensure the JSON file contains predictions and targets.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python accuracy_vs_word_length_no_punct.py <path_to_json>")
        sys.exit(1)
    main(sys.argv[1])
