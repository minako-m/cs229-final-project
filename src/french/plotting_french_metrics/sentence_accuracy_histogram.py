
import json
import matplotlib.pyplot as plt
import sys

def main(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_preds = data['preds']
    all_targets = data['targets']

    # Compute sentence accuracy (exact match)
    correct = [int(p == t) for p, t in zip(all_preds, all_targets)]
    accuracy = sum(correct) / len(correct) if correct else 0
    num_correct = sum(correct)
    num_incorrect = len(correct) - num_correct

    plt.figure(figsize=(6,6))
    plt.pie([num_correct, num_incorrect], labels=['Correct', 'Incorrect'], autopct='%1.1f%%', colors=['#4CAF50', '#F44336'], startangle=90, counterclock=False)
    plt.title(f'Sentence Accuracy\nOverall: {accuracy*100:.2f}%')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sentence_accuracy_histogram.py <path_to_json>")
        sys.exit(1)
    main(sys.argv[1])