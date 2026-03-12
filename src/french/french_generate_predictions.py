import torch
import json
from torch.utils.data import random_split
from french_train import CRNN, FrenchOCRDataset, ctc_greedy_decode, CONFIG

MODEL_PATH = 'best_french_crnn.pt'
IMAGE_DIR = CONFIG['image_dir']
ANNOTATION_DIR = CONFIG['annotation_file']
IMG_HEIGHT = CONFIG['img_height']
RNN_HIDDEN = CONFIG['rnn_hidden']
RNN_LAYERS = CONFIG['rnn_layers']

print('Loading dataset')
dataset = FrenchOCRDataset(IMAGE_DIR, ANNOTATION_DIR, img_height=IMG_HEIGHT)

# fetching test subset
n_total = len(dataset)
n_test  = max(1, int(n_total * 0.1))
n_val   = max(1, int(n_total * 0.1))
n_train = n_total - n_test - n_val

_, _, test_set = random_split(
    dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42),    # same seed as in train
)
print(f'Test set: {len(test_set)} samples')

print('Loading model')
num_classes = len(dataset.vocab) + 1  # +1 for blank
model = CRNN(num_classes, rnn_hidden=RNN_HIDDEN, rnn_layers=RNN_LAYERS)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

all_preds = []
all_targets = []

print('Running inference')
with torch.no_grad():
    for image, label in test_set:
        image = image.unsqueeze(0) 
        log_probs = model(image)
        pred = ctc_greedy_decode(log_probs, dataset.idx2char)[0]
        target = "".join([dataset.idx2char.get(idx.item(), "") for idx in label])
        all_preds.append(pred)
        all_targets.append(target)

# save
output_path = 'french_predictions.json'
print(f'Saving predictions and targets to {output_path}')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump({'preds': all_preds, 'targets': all_targets}, f, ensure_ascii=False, indent=2)
print('Saved, exiting')