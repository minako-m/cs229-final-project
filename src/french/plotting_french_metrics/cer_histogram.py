import json
import matplotlib.pyplot as plt
import numpy as np

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

    cers = []
    for p, t in zip(all_preds, all_targets):
        dist = levenshtein(p, t)
        cers.append(dist / max(len(t), 1))

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
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cer_histogram.py <path_to_json>")
        sys.exit(1)
    main(sys.argv[1])