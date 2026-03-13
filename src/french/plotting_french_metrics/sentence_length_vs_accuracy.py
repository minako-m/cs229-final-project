
import json
import matplotlib.pyplot as plt
import numpy as np
import sys

def main(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_preds = data['preds']
    all_targets = data['targets']

    lengths = [len(t.split()) for t in all_targets]
    correct = [int(p == t) for p, t in zip(all_preds, all_targets)]

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
    plt.title('Accuracy vs Sentence Length')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sentence_length_vs_accuracy.py <path_to_json>")
        sys.exit(1)
    main(sys.argv[1])