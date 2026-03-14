
import json
import matplotlib.pyplot as plt
import numpy as np
import string

def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        new_dp = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            new_dp[j] = min(new_dp[j-1]+1, dp[j]+1, dp[j-1]+cost)
        dp = new_dp
    return dp[n]


import sys

def main(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_preds = data['preds']
    all_targets = data['targets']

    # Function to normalize sentences by removing punctuation and spaces
    def normalize(s):
        return ''.join(c for c in s if c not in string.punctuation and not c.isspace())

    cers = []
    for p, t in zip(all_preds, all_targets):
        norm_p = normalize(p)
        norm_t = normalize(t)
        dist = levenshtein(norm_p, norm_t)
        cers.append(dist / max(len(norm_t), 1))

    plt.figure(figsize=(8,4))
    counts, bins, patches = plt.hist(cers, bins=np.linspace(0,1,21), edgecolor='black')
    plt.xlabel('CER per sentence')
    plt.ylabel('Count')
    plt.title('Histogram of CER Per Sentence')
    # Add count labels above each bar
    for count, patch in zip(counts, patches):
        if count > 0:
            plt.text(patch.get_x() + patch.get_width()/2, count, str(int(count)),
                     ha='center', va='bottom', fontsize=8)
    # Save the plot
    save_dir = r"C:\Users\Stephanie\OneDrive\Documents\Stanford 25-26\Winter '26\CS229\final_project\cs229-final-project\plots\french_metrics_plots_adjusted_accuracy"
    import os
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "cer_histogram_french.png")
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cer_histogram.py <path_to_json>")
        sys.exit(1)
    main(sys.argv[1])