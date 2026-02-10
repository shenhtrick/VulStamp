import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

train_df = pd.read_excel("../../dataset/now/train_slice_intention_oracle_gpt_nan.xlsx", usecols=["func_before", "severity"])
train_df['func_before'] = train_df['func_before'].astype(str)
train_df['severity'] = train_df['severity'].astype(str)
val_df = pd.read_excel("../../dataset/now/val_slice_intention_oracle_gpt_nan.xlsx", usecols=["func_before", "severity"])
val_df['func_before'] = val_df['func_before'].astype(str)
val_df['severity'] = val_df['severity'].astype(str)
test_df = pd.read_excel("../../dataset/now/test_slice_intention_oracle_gpt_nan.xlsx", usecols=["func_before", "severity"])
test_df['func_before'] = test_df['func_before'].astype(str)
test_df['severity'] = test_df['severity'].astype(str)

print("Train labels unique:", train_df['severity'].unique())
print("Val labels unique:", val_df['severity'].unique())
print("Test labels unique:", test_df['severity'].unique())

tokenizer = RobertaTokenizer.from_pretrained('../../models/graphcodebert', do_lower_case=True)

def preprocess_data(df):
    text_list = df['func_before'].tolist()
    encoded_data = tokenizer.batch_encode_plus(
        text_list,
        add_special_tokens=True,
        return_attention_mask=True,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )
    label_encoder = LabelEncoder()
    df['severity'] = label_encoder.fit_transform(df['severity'])
    return encoded_data['input_ids'].to(device), encoded_data['attention_mask'].to(device), torch.tensor(
        df['severity'].values, dtype=torch.long).to(device)

train_input_ids, train_attention_masks, train_labels = preprocess_data(train_df)
val_input_ids, val_attention_masks, val_labels = preprocess_data(val_df)
test_input_ids, test_attention_masks, test_labels = preprocess_data(test_df)

model = RobertaForSequenceClassification.from_pretrained('../../models/graphcodebert', num_labels=4).to(device)

batch_size = 16
epochs = 20
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
from transformers import get_linear_schedule_with_warmup
total_steps = len(train_input_ids) * epochs // batch_size
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

def train_model(model, train_input_ids, train_attention_masks, train_labels, val_input_ids, val_attention_masks,
                val_labels, epochs):
    best_val_f1 = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in tqdm(range(0, len(train_input_ids), batch_size), desc=f'Epoch {epoch + 1}'):
            batch_input_ids = train_input_ids[i:i + batch_size]
            batch_attention_masks = train_attention_masks[i:i + batch_size]
            batch_labels = train_labels[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(val_input_ids, attention_mask=val_attention_masks)
            val_logits = val_outputs.logits
            val_probs = torch.softmax(val_logits, dim=1)
            val_predictions = torch.argmax(val_probs, dim=1)

            val_f1 = f1_score(val_labels.cpu().numpy(), val_predictions.cpu().numpy(), average='macro')
            val_recall = recall_score(val_labels.cpu().numpy(), val_predictions.cpu().numpy(), average='macro')
            val_precision = precision_score(val_labels.cpu().numpy(), val_predictions.cpu().numpy(), average='macro')
            val_auc = roc_auc_score(val_labels.cpu().numpy(), val_probs.cpu().numpy(), multi_class='ovr')
            val_accuracy = accuracy_score(val_labels.cpu().numpy(), val_predictions.cpu().numpy())
            val_f1_cal = 2 * (val_f1 * val_precision) / (val_f1 + val_precision)

            print(f"Epoch {epoch + 1}, Loss: {total_loss / (len(train_input_ids) / batch_size)}")
            print(f"Validation F1: {val_f1}")
            print(f"Validation Precision: {val_precision}")
            print(f"Validation Recall: {val_recall}")
            print(f"Validation F1_cal: {val_f1_cal}")
            print(f"Validation AUC: {val_auc}")

            val_results = pd.DataFrame({
                'True_Label': val_labels.cpu().numpy(),
                'Predicted_Label': val_predictions.cpu().numpy(),
                'Predicted_Probabilities': val_probs.cpu().numpy().tolist()
            })
            val_results.to_csv('result/code/validation_results_{}.csv'.format(epoch), index=False)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), "best_model.pth")
                print("Best model saved!")

train_model(model, train_input_ids, train_attention_masks, train_labels, val_input_ids, val_attention_masks, val_labels,
            epochs)

model.load_state_dict(torch.load("best_model.pth"))
print("Best model loaded!")

model.eval()
with torch.no_grad():
    test_outputs = model(test_input_ids, attention_mask=test_attention_masks)
    test_logits = test_outputs.logits
    test_probs = torch.softmax(test_logits, dim=1)
    test_predictions = torch.argmax(test_probs, dim=1)

    test_f1 = f1_score(test_labels.cpu().numpy(), test_predictions.cpu().numpy(), average='macro')
    test_recall = recall_score(test_labels.cpu().numpy(), test_predictions.cpu().numpy(), average='macro')
    test_precision = precision_score(test_labels.cpu().numpy(), test_predictions.cpu().numpy(), average='macro')
    test_auc = roc_auc_score(test_labels.cpu().numpy(), test_probs.cpu().numpy(), multi_class='ovr')
    test_accuracy = accuracy_score(test_labels.cpu().numpy(), test_predictions.cpu().numpy())

    print("\nTest Results:")
    print(f"F1 Score: {test_f1}")
    print(f"Recall: {test_recall}")
    print(f"Precision: {test_precision}")
    print(f"AUC: {test_auc}")
    print(f"Accuracy: {test_accuracy}")