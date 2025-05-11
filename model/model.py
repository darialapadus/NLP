import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes,
                 pretrained_embeddings=None, pad_idx=0):
        super(BiLSTMWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True

        self.dropout = nn.Dropout(0.4)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.4
        )

        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.context_norm = nn.LayerNorm(hidden_dim * 2)

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, mask, return_attention=False):
        embedded = self.dropout(self.embedding(input_ids))  # [B, T, E]
        lstm_out, _ = self.lstm(embedded)  # [B, T, 2H]
        lstm_out = self.layer_norm(lstm_out)

        attn_scores = self.attention(lstm_out).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)

        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)
        context = self.context_norm(context + torch.mean(lstm_out, dim=1))  # residual
        context = self.dropout(context)

        logits = self.fc(context)
        if return_attention:
            return logits, attn_weights
        return logits
