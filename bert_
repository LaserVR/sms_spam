import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
file_path = r"C:\Users\Senthil Anand\Documents\sms spam\processed_sms_spam.csv"
df = pd.read_csv(file_path, encoding="utf-8")
df.columns = ["label", "text"]
df.dropna(inplace=True)

# Train-Test Split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# Load DistilBERT Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenization Function
def tokenize_data(texts):
    return tokenizer(
        texts.tolist(),
        truncation=True,
        padding="max_length",
        max_length=16,  
        return_tensors="pt"
    )

train_encodings = tokenize_data(train_texts)
test_encodings = tokenize_data(test_texts)

# Convert labels to PyTorch tensors
train_labels = torch.tensor(train_labels.tolist(), dtype=torch.long)
test_labels = torch.tensor(test_labels.tolist(), dtype=torch.long)

# PyTorch Dataset
class SMSDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

# Create Datasets
train_dataset = SMSDataset(train_encodings, train_labels)
test_dataset = SMSDataset(test_encodings, test_labels)

# Load DistilBERT Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.to(device)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,  # Reduced from 64
    per_device_eval_batch_size=16,
    num_train_epochs=3,  # Reduced from 6
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True if torch.cuda.is_available() else False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train Model
trainer.train()

# Evaluate Model
predictions = trainer.predict(test_dataset)
y_pred = torch.argmax(torch.tensor(predictions.predictions).cpu(), axis=1).numpy()
y_true = test_labels.cpu().numpy()

print("✅ Accuracy:", accuracy_score(y_true, y_pred))
print("📊 Classification Report:\n", classification_report(y_true, y_pred))

# Save Model
model.save_pretrained("./sms_spam_model")
tokenizer.save_pretrained("./sms_spam_model")

# Function to Load Model
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("./sms_spam_model")
    model.to(device)
    return model

# Predict Function
def predict_message(model, message):
    inputs = tokenizer(
        message,
        truncation=True,
        padding="max_length",
        max_length=16,
        return_tensors="pt"
    )

    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "SPAM" if prediction == 1 else "HAM"

saved_model_path = "./sms_spam_model"
model.save(saved_model_path)
# Load Model for Testing
model = load_model()

# Example Predictions
test_messages = [
    "Congratulations! You’ve won a FREE iPhone 15! Click here to claim: [spam-link]",
    "Dear Customer, your bank statement is now available. Check online.",
    "Urgent! Your Netflix subscription is about to expire! Renew now at: ",
    "FREE 50GB data from your provider! Claim now:",
]

for msg in test_messages:
    print(f"Message: {msg}\nPrediction: {predict_message(model, msg)}\n")



