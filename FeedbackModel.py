
import json
import pytorch_lightning as pl
import torch
from torch import nn


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def modify_labels_avg_pool(y, kernel_size=1):
    dtype = torch.float32
    device = y.device
    y_float = y.float()
    kernel = torch.ones(1, 1, kernel_size, dtype=dtype, device=device)
    conv = nn.Conv1d(1, 1, kernel_size, padding="same", padding_mode='replicate', bias=False)
    conv.weight.data = kernel
    conv.weight.requires_grad = False
    y_unsqueeze = y_float.unsqueeze(1)
    conv_result = conv(y_unsqueeze).squeeze(1)
    labeled_result = conv_result == kernel_size
    return labeled_result


class TransformerClassifier(pl.LightningModule):
    def __init__(self, embedding_dim, layers, dropout, softlabels=False,
                 num_heads=8, num_tokens=193, clf_size=256, max_seq_len=1024, ctx_tokens=False, weighted=False,
                 noisyLoss=False, noisyReal=False, bar_conditioning=True, avg_pooling=False, avg_pooling_window=None):
        super().__init__()
        self.enc_token_emb = nn.Embedding(num_tokens, embedding_dim)
        self.position_emb = nn.Embedding(max_seq_len, embedding_dim)
        self.ctx_tokens = ctx_tokens
        if self.ctx_tokens:
            self.ctx_emb = nn.Embedding(2, embedding_dim)
        # model
        # Create encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, clf_size),
            torch.nn.ReLU(),
            torch.nn.Linear(clf_size, 1),
        )
        self.softlabels = softlabels
        self.num_heads = num_heads
        self.dropout = dropout
        if weighted:
            print("Balanced loss")
            self.register_buffer("pos_weight", torch.tensor([1.39]))  # 1.49
        else:
            self.register_buffer("pos_weight", None)

        self.noisyLoss = noisyLoss
        self.noisyReal = noisyReal
        self.bar_conditioning = bar_conditioning

        self.avg_pooling = avg_pooling
        self.avg_pooling_window = avg_pooling_window

    def modify_labels_avg_pool(self, y, kernel_size=3):
        dtype = torch.float32
        device = y.device
        y_float = y.float()
        kernel = torch.ones(1, 1, kernel_size, dtype=dtype, device=device)
        conv = nn.Conv1d(1, 1, kernel_size, padding="same", padding_mode='replicate', bias=False)
        conv.weight.data = kernel
        conv.weight.requires_grad = False
        y_unsqueeze = y_float.unsqueeze(1)
        conv_result = conv(y_unsqueeze).squeeze(1)
        labeled_result = conv_result == kernel_size
        return labeled_result

    def forward(self, x, ctx=None):
        # Embed tokens from x
        token_embeddings = self.enc_token_emb(x)
        # Add positional embeddings
        position_embeddings = self.position_emb(torch.arange(x.shape[1], device=x.device).unsqueeze(0))
        token_embeddings += position_embeddings
        if self.ctx_tokens:
            # Update the token embeddings with the context token embedding where ctx is True
            # print(ctx)
            token_embeddings += self.ctx_emb(ctx.long())
        # Pass through the encoder
        x = self.encoder(token_embeddings)
        if self.avg_pooling:
            # pdb.set_trace()
            x = x.transpose(1, 2)
            x = torch.nn.functional.avg_pool1d(x, self.avg_pooling_window, stride=1, padding=self.avg_pooling_window//2)
            x = x.transpose(1, 2)
        # linear layer
        x = self.clf(x)
        return x.squeeze(-1)






