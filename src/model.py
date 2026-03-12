import torch.nn as nn

class CRNN(nn.Module):
    """
    1) CNN, keeps width as time axis
    2) 2-layer Bidirectional LSTM
    3) linear to num_classes
    """
    def __init__(self, num_classes, rnn_hidden=256, rnn_layers=2):
        super().__init__()

        self.cnn = nn.Sequential(
            # Block 1  →  16 × 64
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2  →  8 × 32
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3  →  4 × 32
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),

            # Block 4  →  2 × 32
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),

            # Block 5  →  1 × 31
            nn.Conv2d(512, 512, 2), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.3,
        )

        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        # x: (B, 1, 32, 128)
        feat = self.cnn(x)                    # (B, 512, 1, W')
        feat = feat.squeeze(2)                # (B, 512, W')
        feat = feat.permute(0, 2, 1)          # (B, W', 512)
        out, _ = self.rnn(feat)               # (B, W', hidden*2)
        out = self.fc(out)                    # (B, W', num_classes)
        out = out.permute(1, 0, 2)            # (W', B, num_classes) — for CTCLoss
        return out