import torch
from torch.utils.data import Dataset
import re
import emoji


def simple_tokenizer(text):
    if not isinstance(text, str):
        return ["<UNK>"]
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    text = emoji.demojize(text, delimiters=(" [", "] "))
    tokens = re.findall(r'\b\w+\b|\[[^\]]+\]', text)
    return tokens


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, vocab, max_len=40):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])
        ids = [self.vocab.get(tok, self.vocab["<UNK>"]) for tok in tokens][:self.max_len]
        padding = [self.vocab["<PAD>"]] * (self.max_len - len(ids))
        input_ids = ids + padding
        mask = [1] * len(ids) + [0] * len(padding)
        return torch.tensor(input_ids), torch.tensor(mask), torch.tensor(self.labels[idx])
