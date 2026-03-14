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

        # For each character, calculate misclassification percent and most common misclassified character
        misclass_info = []
        for idx, true_char in enumerate(char_list):
            row = conf_matrix[idx]
            total = row.sum()
            row_no_diag = row.copy()
            row_no_diag[idx] = 0
            if total > 0 and row_no_diag.sum() > 0:
                mis_idx = np.argmax(row_no_diag)
                mis_char = char_list[mis_idx]
                mis_percent = 100 * row_no_diag[mis_idx] / total
            else:
                mis_char = ''
                mis_percent = 0
            misclass_info.append((true_char, mis_char, mis_percent))

        # Sort by mis_percent descending and take top 10
        misclass_info = sorted(misclass_info, key=lambda x: x[2], reverse=True)[:10]
        table_data = [[true_char, mis_char, f"{mis_percent:.1f}%"] for true_char, mis_char, mis_percent in misclass_info]

        import os
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
        plt.title('Top 10 Character Misclassifications by %', y=0.98)
        plt.tight_layout()
        # Save the plot
        save_dir = r"C:\Users\Stephanie\OneDrive\Documents\Stanford 25-26\Winter '26\CS229\final_project\cs229-final-project\plots\kazakh_metrics_plots"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "character_confusion_kazakh.png")
        plt.savefig(save_path, dpi=200)
        plt.show()
        print(f"Plot saved to {save_path}")
    else:
        print("No data to plot. Please ensure the JSON file contains predictions and targets.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python character_confusion_matrix.py <path_to_json>")
        sys.exit(1)
    main(sys.argv[1])