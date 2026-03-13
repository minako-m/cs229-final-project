import json
import matplotlib.pyplot as plt
import numpy as np
import sys

def main(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        all_preds = data['preds']
        all_targets = data['targets']
    except FileNotFoundError:
        print(f"{json_path} not found. Please provide a valid JSON file.")
        return

    # Calculate accuracy vs word length
    lengths = []
    correct = []
    for pred, target in zip(all_preds, all_targets):
        l = len(target)
        is_correct = int(pred == target)
        lengths.append(l)
        correct.append(is_correct)

    if lengths:
        # Bin word lengths
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
        plt.xlabel('Word Length')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Word Length')
        plt.grid(True)
        plt.show()
    else:
        print("No data to plot. Please ensure the JSON file contains predictions and targets.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python accuracy_vs_word_length.py <path_to_json>")
        sys.exit(1)
    main(sys.argv[1])