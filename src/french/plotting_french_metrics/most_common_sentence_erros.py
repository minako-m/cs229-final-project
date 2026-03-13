
import json
import matplotlib.pyplot as plt
from collections import Counter
import sys

def main(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_preds = data['preds']
    all_targets = data['targets']

    errors = [(t, p) for t, p in zip(all_targets, all_preds) if t != p]
    counter = Counter([t for t, _ in errors])
    most_common = counter.most_common(20)

    table_data = []
    for gt, count in most_common:
        preds = [p for t, p in errors if t == gt]
        pred_counter = Counter(preds)
        most_pred = pred_counter.most_common(1)[0][0] if pred_counter else ''
        table_data.append([gt, most_pred, count])

    fig, ax = plt.subplots(figsize=(12, 0.6*len(table_data)+2))
    ax.axis('off')
    table = ax.table(
        cellText=table_data,
        colLabels=['Ground Truth', 'Most Common Prediction', 'Count'],
        loc='center',
        cellLoc='left',
        colLoc='left',
        edges='open',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width([0, 1, 2])
    # Set all table lines to white
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('white')
    # Move the title lower by using y=0.02 in fig.text
    fig.text(0.5, 0.02, 'Top 20 Most Common Misclassified Sentences', ha='center', fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('most_common_sentence_errors_table.png', dpi=200)
    plt.show()
    print('Saved table image as most_common_sentence_errors_table.png')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python most_common_sentence_erros.py <path_to_json>")
        sys.exit(1)
    main(sys.argv[1])