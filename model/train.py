import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from model import BiLSTMWithAttention
from dataset_utils import TweetDataset, simple_tokenizer
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_vocab(texts, tokenizer, min_freq=1):
    counter = Counter()
    for text in texts:
        counter.update(tokenizer(text))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def train_model(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for input_ids, masks, labels in dataloader:
        input_ids, masks, labels = input_ids.to(device), masks.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(input_ids, masks)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def eval_model(model, dataloader):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for input_ids, masks, labels in dataloader:
            input_ids, masks = input_ids.to(device), masks.to(device)
            logits = model(input_ids, masks)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred)
            truths.extend(labels.numpy())
    acc = accuracy_score(truths, preds)
    precision = precision_score(truths, preds, average="macro", zero_division=0)
    recall = recall_score(truths, preds, average="macro", zero_division=0)
    f1 = f1_score(truths, preds, average="macro")
    return acc, precision, recall, f1


def main():
    # Adjust path as needed based on your folder structure
    train_path = "../datasets/train/SemEval2018-T3-train-taskA_original.txt"

    df = pd.read_csv(train_path, sep="\t", names=["id", "label", "text"], skiprows=1)
    df = df.dropna(subset=["text", "label"])

    texts = df["text"].tolist()
    labels = df["label"].astype(int).tolist()

    tokenizer = simple_tokenizer
    vocab = build_vocab(texts, tokenizer)

    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.1, random_state=42)

    train_dataset = TweetDataset(X_train, y_train, tokenizer, vocab)
    val_dataset = TweetDataset(X_val, y_val, tokenizer, vocab)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    model = BiLSTMWithAttention(
        vocab_size=len(vocab),
        embedding_dim=200,
        hidden_dim=256,
        num_classes=2,
        pad_idx=vocab["<PAD>"]
    ).to(device)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_f1 = 0
    patience = 3
    counter = 0

    for epoch in range(1, 21):
        loss = train_model(model, train_loader, optimizer, criterion)
        acc, p, r, f1 = eval_model(model, val_loader)
        print(f"[Epoch {epoch}] Loss: {loss:.4f} | Acc: {acc:.4f} | P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            counter = 0
            torch.save(model.state_dict(), "bilstm_model.pt")
            print("✅ Model saved.")
        else:
            counter += 1
            if counter >= patience:
                print("⏹️ Early stopping.")
                break

    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)


if __name__ == "__main__":
    main()
