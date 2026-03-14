import json
import numpy as np
import matplotlib.pyplot as plt
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

    # Define characters to ignore (punctuation and whitespace)
    ignore_chars = set(string.punctuation) | set(string.whitespace)

    # Build character set (excluding punctuation and whitespace)
    char_set = set()
    for target in all_targets:
        char_set.update([c for c in target if c not in ignore_chars])
    for pred in all_preds:
        char_set.update([c for c in pred if c not in ignore_chars])
    char_list = sorted(list(char_set))
    char_to_idx = {c: i for i, c in enumerate(char_list)}

    # Initialize confusion matrix
    conf_matrix = np.zeros((len(char_list), len(char_list)), dtype=int)

    # Populate confusion matrix, skipping ignored characters
    for pred, target in zip(all_preds, all_targets):
        min_len = min(len(pred), len(target))
        for i in range(min_len):
            t_char = target[i]
            p_char = pred[i]
            if t_char in ignore_chars:
                continue
            t_idx = char_to_idx.get(t_char)
            p_idx = char_to_idx.get(p_char) if p_char not in ignore_chars else None
            if t_idx is not None and p_idx is not None:
                conf_matrix[t_idx, p_idx] += 1
            elif t_idx is not None:
                # Prediction is ignored char, count as blank
                pass  # Optionally handle as needed
        # Handle extra characters in target
        for i in range(min_len, len(target)):
            t_char = target[i]
            if t_char in ignore_chars:
                continue
            t_idx = char_to_idx.get(t_char)
            if t_idx is not None:
                pass  # Optionally handle as needed
        # Ignore extra characters in pred (if not in char_list)

    # For each character, calculate accuracy percent (diagonal / row sum)
    acc_info = []
    for idx, true_char in enumerate(char_list):
        row = conf_matrix[idx]
        total = row.sum()
        correct = row[idx]
        acc_percent = 100 * correct / total if total > 0 else 0
        acc_info.append((true_char, acc_percent))

    # Sort by acc_percent descending and take top 10
    acc_info = sorted(acc_info, key=lambda x: x[1], reverse=True)[:10]
    table_data = [[true_char, f"{acc_percent:.1f}%"] for true_char, acc_percent in acc_info]

    fig, ax = plt.subplots(figsize=(6, 0.6*len(table_data)+2))
    ax.axis('off')
    table = ax.table(
        cellText=table_data,
        colLabels=['True Char', '% Accurate'],
        loc='center',
        cellLoc='center',
        colLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([0, 1])
    plt.title('Top 10 Most Accurate Characters (French)', y=0.98)
    plt.tight_layout()
    # Save the plot
    save_dir = r"C:\Users\Stephanie\OneDrive\Documents\Stanford 25-26\Winter '26\CS229\final_project\cs229-final-project\plots\french_metrics_plots_adjusted_accuracy"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "top10_accurate_characters_french.png")
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python accuracy_per_characters_french.py <path_to_json>")
        sys.exit(1)
    main(sys.argv[1])
