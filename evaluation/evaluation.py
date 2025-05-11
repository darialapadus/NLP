from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def read_predictions(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [int(line.strip()) for line in f if line.strip()]

def read_labels(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]  # Skip header
        return [int(line.strip().split('\t')[1]) for line in lines if line.strip()]

def main():
    pred_path = "../model/predictions-taskA.txt"
    gold_path = "../datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt"

    preds = read_predictions(pred_path)
    labels = read_labels(gold_path)

    if len(preds) != len(labels):
        print(f"❌ Mismatch: {len(preds)} predictions vs {len(labels)} gold labels.")
        return

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', pos_label=1)

    print("✅ Evaluation Results:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

if __name__ == "__main__":
    main()
