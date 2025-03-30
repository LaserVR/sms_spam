import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_path = "./sms_spam_model"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()  

def predict_message(messages):
    if isinstance(messages, str):
        messages = [messages]  

    inputs = tokenizer(
        messages,
        truncation=True,
        padding="max_length",
        max_length=16,  
        return_tensors="pt"
    )

    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    return ["SPAM" if pred == 1 else "HAM" for pred in predictions]

test_messages = [
    "Congratulations! You’ve won a FREE iPhone 15! Click here to claim: [spam-link]",
    "Dear Customer, your bank statement is now available. Check online.",
    "Urgent! Your Netflix subscription is about to expire! Renew now at: ",
    "FREE 50GB data from your provider! Claim now:",
    "Hey, are we still meeting for lunch today?",
    "Congratulations,Your personal loanof upto Rs 25Lacs can be preapproved check Now",
    "Get 2Star Frost - Free Refrigerator starting Rs 19990* with Easy EMI Benefit. Visit Croma store now T&C",
    "Claim your FREE vacation to Hawaii! Call now: 1-800-SPAM-TRAP",
    "Win big now!You have been selected for a $1,000 gift card. Click here: [spam-link]",
    "Hi Alex, your car service is completed. You can pick it up anytime today.",
    "Dad, I reached safely. Will call you soon.",
    "Your electricity bill for March is ₹1,250. Pay before the due date to avoid penalties.",
    ]

predictions = predict_message(test_messages)

for msg, pred in zip(test_messages, predictions):
    print(f"📩 Message: {msg}\n➡️ Prediction: {pred}\n")

