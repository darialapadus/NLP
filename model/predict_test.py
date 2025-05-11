import torch
from torch.utils.data import DataLoader, Dataset
from model import BiLSTMWithAttention
from dataset_utils import simple_tokenizer
import pandas as pd
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PredictTweetDataset(Dataset):
    def __init__(self, texts, tokenizer, vocab, max_len=40):
        self.texts = texts
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
        return torch.tensor(input_ids), torch.tensor(mask)


def main():
    # Load test tweets
    df = pd.read_csv("../datasets/test_TaskA/SemEval2018-T3_input_test_taskA.txt",
                     sep="\t", names=["id", "text"], skiprows=1)
    texts = df["text"].dropna().tolist()

    # Load vocab
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    # Prepare dataset
    test_dataset = PredictTweetDataset(texts, simple_tokenizer, vocab)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Load model
    model = BiLSTMWithAttention(
        vocab_size=len(vocab),
        embedding_dim=200,
        hidden_dim=256,
        num_classes=2,
        pad_idx=vocab["<PAD>"]
    ).to(device)

    model.load_state_dict(torch.load("bilstm_model.pt", map_location=device))
    model.eval()

    predictions = []
    with torch.no_grad():
        for input_ids, masks in test_loader:
            input_ids, masks = input_ids.to(device), masks.to(device)
            logits = model(input_ids, masks)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            predictions.extend(preds)

    with open("predictions-taskA.txt", "w", encoding="utf-8") as f_out:
        for p in predictions:
            f_out.write(f"{p}\n")

    print("✅ Predictions saved to predictions-taskA.txt")


if __name__ == "__main__":
    main()
