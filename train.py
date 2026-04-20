import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# ── Config ────────────────────────────────────
TEACHER_MODEL  = "Sachin-Rathore-1234/bert-imdb-sentiment"
STUDENT_MODEL  = "distilbert-base-uncased"
TEMPERATURE    = 4.0
ALPHA          = 0.7
EPOCHS         = 3
BATCH_SIZE     = 16
LR             = 2e-5
MAX_LENGTH     = 256
NUM_LABELS     = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset ───────────────────────────────────
dataset    = load_dataset("imdb")
tokenizer  = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length",
                     truncation=True, max_length=MAX_LENGTH)

train_data = dataset["train"].shuffle(seed=42).select(range(3000))
test_data  = dataset["test"].shuffle(seed=42).select(range(500))

for split, data in [("train", train_data), ("test", test_data)]:
    data = data.map(tokenize, batched=True)
    data = data.rename_column("label", "labels")
    data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    if split == "train":
        tok_train = data
    else:
        tok_test = data

train_loader = DataLoader(tok_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(tok_test,  batch_size=32, shuffle=False)

# ── Models ────────────────────────────────────
teacher = BertForSequenceClassification.from_pretrained(TEACHER_MODEL).to(device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

student = DistilBertForSequenceClassification.from_pretrained(
    STUDENT_MODEL, num_labels=NUM_LABELS).to(device)

# ── Loss ──────────────────────────────────────
def distillation_loss(s_logits, t_logits, labels, T=4.0, alpha=0.7):
    soft_t = F.softmax(t_logits / T, dim=-1)
    soft_s = F.log_softmax(s_logits / T, dim=-1)
    kl     = F.kl_div(soft_s, soft_t, reduction="batchmean") * (T ** 2)
    ce     = F.cross_entropy(s_logits, labels)
    return alpha * kl + (1 - alpha) * ce

# ── Optimizer ─────────────────────────────────
optimizer     = AdamW(student.parameters(), lr=LR, weight_decay=0.01)
total_steps   = len(train_loader) * EPOCHS
scheduler     = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=total_steps//10,
    num_training_steps=total_steps)

# ── Training ──────────────────────────────────
best_acc = 0
for epoch in range(1, EPOCHS + 1):
    student.train()
    for batch in train_loader:
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        labs  = batch["labels"].to(device)
        with torch.no_grad():
            t_logits = teacher(input_ids=ids, attention_mask=mask).logits
        s_logits = student(input_ids=ids, attention_mask=mask).logits
        loss = distillation_loss(s_logits, t_logits, labs, TEMPERATURE, ALPHA)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    # Evaluate
    student.eval()
    preds, true = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)
            out  = student(input_ids=ids, attention_mask=mask)
            preds.extend(out.logits.argmax(-1).cpu().numpy())
            true.extend(labs.cpu().numpy())
    acc = accuracy_score(true, preds)
    print(f"Epoch {epoch} | Accuracy: {acc*100:.2f}%")
    if acc > best_acc:
        best_acc = acc
        student.save_pretrained("./distilbert-imdb-distilled")
        tokenizer.save_pretrained("./distilbert-imdb-distilled")

print(f"Done! Best Accuracy: {best_acc*100:.2f}%")
