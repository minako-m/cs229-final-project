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

    cer_per_word = []
    for pred, target in zip(all_preds, all_targets):
        dist = levenshtein_distance(pred, target)
        length = max(len(target), 1)
        cer = dist / length
        cer_per_word.append(cer)

    if cer_per_word:
        plt.figure(figsize=(8, 5))
        counts, bins, patches = plt.hist(cer_per_word, bins=np.linspace(0, 1, 21), edgecolor='black')
        plt.xlabel('CER per word')
        plt.ylabel('Count')
        plt.title('Histogram of CER Per Word')
        # Add count labels above each bar
        for count, patch in zip(counts, patches):
            if count > 0:
                plt.text(patch.get_x() + patch.get_width()/2, count, str(int(count)),
                         ha='center', va='bottom', fontsize=8)
        plt.grid(True)
        plt.show()
    else:
        print("No CER data to plot. Please ensure the JSON file contains predictions and targets.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python histogram_cer_per_word.py <path_to_json>")
        sys.exit(1)
    main(sys.argv[1])