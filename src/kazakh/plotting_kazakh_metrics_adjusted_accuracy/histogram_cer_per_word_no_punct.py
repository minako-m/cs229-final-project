import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import string

def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        new_dp = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            new_dp[j] = min(new_dp[j - 1] + 1, dp[j] + 1, dp[j - 1] + cost)
        dp = new_dp
    return dp[n]

def normalize_string(s):
    return ''.join([c for c in s if c not in string.punctuation and not c.isspace()])

def main(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        all_preds = data['preds']
        all_targets = data['targets']
    except FileNotFoundError:
        print(f"{json_path} not found. Please provide a valid JSON file.")
        return

    cer_per_word = []
    for pred, target in zip(all_preds, all_targets):
        pred_norm = normalize_string(pred)
        target_norm = normalize_string(target)
        dist = levenshtein_distance(pred_norm, target_norm)
        length = max(len(target_norm), 1)
        cer = dist / length
        cer_per_word.append(cer)

    if cer_per_word:
        plt.figure(figsize=(8, 5))
        counts, bins, patches = plt.hist(cer_per_word, bins=np.linspace(0, 1, 21), edgecolor='black')
        plt.xlabel('CER per word (no punct/space)')
        plt.ylabel('Count')
        plt.title('Histogram of CER Per Word (No Punctuation/Space)')
        for count, patch in zip(counts, patches):
            if count > 0:
                plt.text(patch.get_x() + patch.get_width()/2, count, str(int(count)),
                         ha='center', va='bottom', fontsize=8)
        plt.grid(True)
        save_dir = r"C:\Users\Stephanie\OneDrive\Documents\Stanford 25-26\Winter '26\CS229\final_project\cs229-final-project\plots\kazakh_metrics\adjusted_accuracy"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "histogram_cer_per_word_no_punct.png")
        plt.savefig(save_path, dpi=200)
        plt.show()
        print(f"Plot saved to {save_path}")
    else:
        print("No CER data to plot. Please ensure the JSON file contains predictions and targets.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python histogram_cer_per_word_no_punct.py <path_to_json>")
        sys.exit(1)
    main(sys.argv[1])
