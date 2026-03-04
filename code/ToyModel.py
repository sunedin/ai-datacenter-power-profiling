import torch
import torch.nn as nn

class ToyLLMModel(nn.Module):
    def __init__(self, vocab_size=32000, d_model=1024,
                 nhead=8, num_layers=8, dim_ff=4096, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: [B, T] token ids
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.embed(x) + self.pos_embed(pos)
        h = self.encoder(h)
        logits = self.lm_head(h)  # [B, T, V]
        return logits


class ToyEncoderModel(nn.Module):
    def __init__(self, vocab_size=32000, d_model=768,
                 nhead=6, num_layers=6, dim_ff=3072, max_len=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.embed(x) + self.pos_embed(pos)
        h = self.encoder(h)
        cls_tok = h[:, 0, :]
        logits = self.cls_head(cls_tok)
        return logits


class ToyCNNModel(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, 7, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        h = self.net(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)