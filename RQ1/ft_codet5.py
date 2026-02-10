import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, T5ForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.optim import AdamW
from tqdm import tqdm  # 导入进度条
import csv
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

model_path = "../../models/codet5"


train_df = pd.read_excel("../../dataset/now/train_slice_intention_oracle_gpt_nan.xlsx", usecols=["func_before", "severity"])
train_df['func_before'] = train_df['func_before'].astype(str)
train_df['severity'] = train_df['severity'].astype(str)
val_df = pd.read_excel("../../dataset/now/val_slice_intention_oracle_gpt_nan.xlsx", usecols=["func_before", "severity"])
val_df['func_before'] = val_df['func_before'].astype(str)
val_df['severity'] = val_df['severity'].astype(str)
test_df = pd.read_excel("../../dataset/now/test_slice_intention_oracle_gpt_nan.xlsx", usecols=["func_before", "severity"])
test_df['func_before'] = test_df['func_before'].astype(str)
test_df['severity'] = test_df['severity'].astype(str)


label_encoder = LabelEncoder()
train_df['severity'] = label_encoder.fit_transform(train_df['severity'])
val_df['severity'] = label_encoder.transform(val_df['severity'])
test_df['severity'] = label_encoder.transform(test_df['severity'])


train_texts = train_df['func_before'].tolist()
train_labels = train_df['severity'].tolist()

val_texts = val_df['func_before'].tolist()
val_labels = val_df['severity'].tolist()

test_texts = test_df['func_before'].tolist()
test_labels = test_df['severity'].tolist()


tokenizer = RobertaTokenizer.from_pretrained(model_path)

model = T5ForSequenceClassification.from_pretrained(model_path, num_labels=4)


class CodeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {key: encoding[key].squeeze(0) for key in encoding}  # Remove the batch dimension
        item['severity'] = torch.tensor(label, dtype=torch.long)

        return item


train_dataset = CodeDataset(train_texts, train_labels, tokenizer)
val_dataset = CodeDataset(val_texts, val_labels, tokenizer)
test_dataset = CodeDataset(test_texts, test_labels, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)
test_dataloader = DataLoader(test_dataset, batch_size=16)


optimizer = AdamW(model.parameters(), lr=2e-5)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


def train(model, train_dataloader, optimizer):
    model.train()
    all_preds = []
    all_labels = []
    all_probabilities = []  # 新增：用于存储预测概率
    for batch in tqdm(train_dataloader, desc="Training", ncols=100):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['severity'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        probabilities = torch.softmax(logits, dim=-1)  #
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().detach().numpy())  #


    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    try:
        auc = roc_auc_score(all_labels, np.array(all_probabilities), multi_class='ovr')
    except ValueError:
        auc = 0.0

    print(f"Training Loss: {loss.item():.4f}")
    print(f"Train Accuracy: {accuracy:.4f}")
    print(f"Train Precision: {precision:.4f}")
    print(f"Train Recall: {recall:.4f}")
    print(f"Train F1 Score: {f1:.4f}")
    print(f"Train AUC: {auc:.4f}")

# 评估函数
def evaluate(model, val_dataloader, cur_epoch):
    model.eval()
    predictions, true_labels = [], []
    all_probabilities = []  #

    for batch in tqdm(val_dataloader, desc="Evaluating", ncols=100):  #
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['severity'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        preds = torch.argmax(logits, dim=-1)
        probabilities = torch.softmax(logits, dim=-1)  #
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().detach().numpy())  #

        with open('result/code/pred_{}.csv'.format(cur_epoch), mode='w', newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(['pred', 'label'])
            for pred, label in zip(predictions, true_labels):
                writer.writerow([pred, label])

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    # f1 = f1_score(true_labels, predictions, average='macro')
    f1 = 2 * precision * recall / (precision + recall)
    try:
        auc = roc_auc_score(true_labels, np.array(all_probabilities), multi_class='ovr')
    except ValueError:
        auc = 0.0

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")
    print(f"Validation AUC: {auc:.4f}")

# 测试函数
def test(model, test_dataloader):
    model.eval()
    predictions, true_labels = [], []
    all_probabilities = []

    for batch in tqdm(test_dataloader, desc="Testing", ncols=100):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        preds = torch.argmax(logits, dim=-1)
        probabilities = torch.softmax(logits, dim=-1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')
    try:
        auc = roc_auc_score(true_labels, np.array(all_probabilities), multi_class='ovr')
    except ValueError:
        auc = 0.0

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test AUC: {auc:.4f}")


os.makedirs('result/codet5/pred', exist_ok=True)


for epoch in range(20):  # 训练20个epoch
    print(f"\nEpoch {epoch + 1}")
    train(model, train_dataloader, optimizer)
    evaluate(model, val_dataloader, epoch)

test(model, test_dataloader)