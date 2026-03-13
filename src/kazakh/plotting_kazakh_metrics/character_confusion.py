import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

    # Build character set
    char_set = set()
    for target in all_targets:
        char_set.update(target)
    for pred in all_preds:
        char_set.update(pred)
    char_list = sorted(list(char_set))
    char_to_idx = {c: i for i, c in enumerate(char_list)}

    # Initialize confusion matrix
    conf_matrix = np.zeros((len(char_list), len(char_list)), dtype=int)

    # Populate confusion matrix
    for pred, target in zip(all_preds, all_targets):
        min_len = min(len(pred), len(target))
        for i in range(min_len):
            t_idx = char_to_idx[target[i]]
            p_idx = char_to_idx[pred[i]]
            conf_matrix[t_idx, p_idx] += 1
        # Handle extra characters
        for i in range(min_len, len(target)):
            t_idx = char_to_idx[target[i]]
            conf_matrix[t_idx, char_to_idx.get('', t_idx)] += 1  # blank prediction
        for i in range(min_len, len(pred)):
            p_idx = char_to_idx[pred[i]]
            conf_matrix[char_to_idx.get('', p_idx), p_idx] += 1  # blank target

    # Display only the top 10 most frequent true characters and their most common misclassified predictions
    if conf_matrix.sum() > 0:
        import pandas as pd
        # Get total counts for each true character (row sum)
        row_sums = conf_matrix.sum(axis=1)
        top_indices = np.argsort(row_sums)[-10:][::-1]  # indices of top 10 true characters
        top_chars = [char_list[i] for i in top_indices]

        # For each top character, find the most common misclassified prediction (not itself), and show percent of misclassification
        table_data = []
        for idx in top_indices:
            true_char = char_list[idx]
            row = conf_matrix[idx]
            total = row.sum()
            # Exclude correct predictions (diagonal)
            row_no_diag = row.copy()
            row_no_diag[idx] = 0
            if row_no_diag.sum() > 0:
                mis_idx = np.argmax(row_no_diag)
                mis_char = char_list[mis_idx]
                mis_percent = 100 * row_no_diag[mis_idx] / total if total > 0 else 0
            else:
                mis_char = ''
                mis_percent = 0
            table_data.append([
                true_char,
                mis_char,
                f"{mis_percent:.1f}%"
            ])

        fig, ax = plt.subplots(figsize=(7, 0.6*len(table_data)+2))
        ax.axis('off')
        table = ax.table(
            cellText=table_data,
            colLabels=['True Char', 'Most Misclassified As', '% Misclassified'],
            loc='center',
            cellLoc='center',
            colLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width([0, 1, 2])
        plt.title('Top 10 Character Misclassifications', y=0.98)
        plt.tight_layout()
        plt.show()
    else:
        print("No data to plot. Please ensure the JSON file contains predictions and targets.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python character_confusion_matrix.py <path_to_json>")
        sys.exit(1)
    main(sys.argv[1])