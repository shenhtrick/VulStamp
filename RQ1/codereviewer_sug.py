import os
import torch
import datasets
import transformers
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptDataLoader, PromptForClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

seed = 42
batch_size = 16
num_class = 4
max_seq_l = 512
lr = 5e-5
num_epochs = 20
use_cuda = True
model_name = "t5"
pretrainedmodel_path = "../../models/codereviewer"  # Path of the pre-trained model
early_stop_threshold = 10


rl_coeff = 0.01  #
baseline_alpha = 0.7  #


classes = [0, 1, 2, 3]


def read_prompt_examples(filename):
    examples = []
    data = pd.read_excel(filename).astype(str)
    desc   = data['description'].tolist()
    exp    = data['exploitability'].tolist()
    imp    = data['impact'].tolist()
    sc     = data['scope'].tolist()
    analysis = [e+i+s for e,i,s in zip(exp,imp,sc)]

    code      = data['sliced_code_nan'].tolist()
    severity  = data['severity'].tolist()
    oracle    = data['oracle_gpt'].tolist()

    for idx,(c,a,s,o) in enumerate(zip(code,analysis,severity,oracle)):
        examples.append(
            InputExample(
                guid = idx,
                text_a = ' '.join(c.split()[:384]),
                text_b = ' '.join(a.split()[:64]),
                tgt_text = s,
                meta = {"oracle": o}
            )
        )
    return examples


# Load model and tokenizer
plm, tokenizer, model_config, WrapperClass = load_plm(model_name = model_name, model_path = pretrainedmodel_path)



# soft
template_text = '{"soft":" "}: {"placeholder":"text_a"} {"soft":" "}: {"placeholder":"text_b"} {"soft":" "} {"mask"}'
# hard
# template_text = 'The code snippet: {"placeholder":"text_a"} The vulnerability analysis: {"placeholder":"text_b"} Classify the severity: {"mask"}'


mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)


# Define the verbalizer
myverbalizer = ManualVerbalizer(tokenizer, classes=classes,
                                label_words={
                                    0: ["low", "slight"],
                                    1: ["medium", "moderate"],
                                    2: ["high", "severe"],
                                    3: ["critical", "significant"]
                                })
# Define the prompt model
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model = prompt_model.to(device)
    print("yy")

# Define the optimizer and scheduler
loss_func = torch.nn.CrossEntropyLoss()

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters1 = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.01}
]
optimizer_grouped_parameters2 = [
    {'params': [p for n, p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
]
optimizer1 = AdamW(optimizer_grouped_parameters1, lr=lr)
optimizer2 = AdamW(optimizer_grouped_parameters2, lr=lr)

train_dataloader = PromptDataLoader(
    dataset=read_prompt_examples("../../dataset/now/train_slice_intention_oracle_gpt_nan.xlsx"),
    template=mytemplate,
    tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass,
    max_seq_length=max_seq_l,
    batch_size=batch_size,
    shuffle=True,
    teacher_forcing=False,
    predict_eos_token=False,
    truncate_method="tail",
    decoder_max_length=3)


valid_dataloader = PromptDataLoader(
    dataset=read_prompt_examples("../../dataset/now/val_slice_intention_oracle_gpt_nan.xlsx"),
    template=mytemplate,
    tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass,
    max_seq_length=max_seq_l,
    batch_size=batch_size,
    shuffle=False,
    teacher_forcing=False,
    predict_eos_token=False,
    truncate_method="tail",
    decoder_max_length=3
)


label_to_id = {str(label): idx for idx, label in enumerate(classes)}


class HybridPromptModel(PromptForClassification):
    def forward(self, *args, **kwargs):
        logits = super().forward(*args, **kwargs)  # 原始返回的是logits张量
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return type('Output', (object,), {'logits': logits, 'probs': probs})()

prompt_model = HybridPromptModel(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model = prompt_model.to(device)
    print("yes")

best_valid_f1 = 0
no_improve_epochs = 0
reward_baseline = 0

# # 定义权重
reward_weights = {
    0: 2.0,
    1: 5.5,
    2: 8.0,
    3: 9.5
}



from transformers import T5ForConditionalGeneration

gen_model = T5ForConditionalGeneration.from_pretrained(pretrainedmodel_path)
if use_cuda:
    gen_model = gen_model.to(device)

gen_tokenizer = tokenizer
max_gen_len = 128

gen_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=gen_tokenizer.pad_token_id)
optimizer_gen = AdamW(gen_model.parameters(), lr=lr)

for epoch in range(num_epochs):
    prompt_model.train()
    gen_model.train()
    tot_loss = 0
    all_labels = []
    all_preds = []

    batch_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch} Training")
    for step, inputs in enumerate(batch_progress_bar):
        inputs = inputs.to(device)

        outputs = prompt_model(inputs)
        logits = outputs.logits
        probs = outputs.probs

        labels = []
        train_indices = []
        for idx, label in enumerate(inputs['tgt_text']):
            if label in label_to_id:
                labels.append(label_to_id[label])
                train_indices.append(idx)
        if not labels:
            continue
        labels = torch.tensor(labels).to(device)

        loss_ce = loss_func(logits[train_indices], labels)

        with torch.no_grad():
            sampled_actions = torch.multinomial(probs[train_indices], 1).squeeze()

            max_probs, preds = torch.max(probs[train_indices], dim=1)
            is_correct = (preds == labels)

            sample_weights = torch.tensor([reward_weights[label.item()] for label in labels], device=device)
            rewards = torch.where(is_correct, max_probs * sample_weights, -max_probs * sample_weights)

        log_probs = torch.nn.functional.softplus(
            torch.log(probs[train_indices].gather(1, sampled_actions.unsqueeze(1))).squeeze())

        reward_baseline = baseline_alpha * reward_baseline + (1 - baseline_alpha) * rewards.mean().item()
        adjusted_rewards = rewards - reward_baseline

        loss_rl = -(log_probs * adjusted_rewards).mean()

        total_loss = loss_ce + rl_coeff * loss_rl

        # # w/o rl
        # total_loss = loss_ce

        gen_input_ids = inputs['input_ids']
        gen_labels = gen_tokenizer(
            [ex.meta['oracle'] for ex in inputs['guid']],  # 通过 guid 取回 meta
            max_length=max_gen_len,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).input_ids.to(device)

        gen_out = gen_model(input_ids=gen_input_ids, labels=gen_labels)
        loss_gen = gen_out.loss
        lambda_gen = 1.0
        total_loss = total_loss + lambda_gen * loss_gen

        total_loss.backward()
        optimizer2.step()
        optimizer2.zero_grad()
        optimizer_gen.step()
        optimizer_gen.zero_grad()
        optimizer1.step()
        optimizer1.zero_grad()

        tot_loss += total_loss.item()
        preds = torch.argmax(logits[train_indices], dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        batch_progress_bar.set_postfix({
            "CE Loss": loss_ce.item(),
            # "RL Loss": loss_rl.item(),
            "Total Loss": total_loss.item()
        })
    prompt_model.eval()
    valid_all_labels = []
    valid_all_preds = []
    valid_all_probs = []
    valid_all_refs = []
    valid_all_hyps = []  #
    valid_batch_progress_bar = tqdm(valid_dataloader, desc=f"Epoch {epoch} Valid Batches", leave=False)
    with torch.no_grad():
        for inputs in valid_dataloader:
            if use_cuda:
                inputs = inputs.to(device)

            outputs = prompt_model(inputs)
            logits = outputs.logits
            probs = outputs.probs

            # 检查并转换标签
            labels = []
            valid_indices = []
            for idx, label in enumerate(inputs['tgt_text']):
                if label in label_to_id:
                    labels.append(label_to_id[label])
                    valid_indices.append(idx)
                else:
                    print(f"Warning: Unknown label '{label}' found. Skipping this example.")

            if not labels:
                continue

            valid_logits = logits[valid_indices]
            valid_probs = probs[valid_indices]

            labels = torch.tensor(labels).to(device)

            preds = torch.argmax(valid_logits, dim=1)
            valid_all_labels.extend(labels.cpu().numpy())
            valid_all_preds.extend(preds.cpu().numpy())
            valid_all_probs.extend(valid_probs.cpu().numpy())  # 新增：保存预测概

            gen_input_ids = inputs['input_ids']
            gen_labels = gen_tokenizer(
                [ex.meta['oracle'] for ex in inputs['guid']],
                max_length=max_gen_len,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).input_ids.to(device)

            gen_ids = gen_model.generate(
                input_ids=gen_input_ids,
                max_length=max_gen_len,
                num_beams=4,
                early_stopping=True
            )
            hyps = gen_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            refs = gen_tokenizer.batch_decode(gen_labels, skip_special_tokens=True)
            valid_all_hyps.extend(hyps)
            valid_all_refs.extend(refs)

            valid_batch_progress_bar.update(1)

    valid_accuracy = accuracy_score(valid_all_labels, valid_all_preds)
    valid_recall = recall_score(valid_all_labels, valid_all_preds, average='macro')
    valid_precision = precision_score(valid_all_labels, valid_all_preds, average='macro')
    valid_f1_cal = 2 * valid_precision * valid_recall / (valid_precision + valid_recall)
    valid_f1 = f1_score(valid_all_labels, valid_all_preds, average='macro')
    valid_mcc = matthews_corrcoef(valid_all_labels, valid_all_preds)
    valid_auc = roc_auc_score(valid_all_labels, valid_all_probs, multi_class='ovr')

    valid_batch_progress_bar.close()

    print(f"Epoch: {epoch}, Valid Accuracy: {valid_accuracy}")
    print(f"Epoch: {epoch}, Valid Precision: {valid_precision}")
    print(f"Epoch: {epoch}, Valid Recall: {valid_recall}")
    print(f"Epoch: {epoch}, Valid F1_cal: {valid_f1_cal}")
    print(f"Epoch: {epoch}, Valid F1: {valid_f1}")
    print(f"Epoch: {epoch}, Valid MCC: {valid_mcc}")
    print(f"Epoch: {epoch}, Valid AUC: {valid_auc}")  # 新增：打印 AUC

    valid_results = pd.DataFrame({
        'True_Label': valid_all_labels,
        'Predicted_Label': valid_all_preds,
        'Reference_oracle': valid_all_refs,
        'Generated_oracle': valid_all_hyps
    })

    valid_results.to_csv('result/sug/validation_results_{}.csv'.format(epoch), index=False)

    if valid_f1 > best_valid_f1:
        best_valid_f1 = valid_f1
        no_improve_epochs = 0
        # 保存最佳模型
        torch.save(prompt_model.state_dict(), "best_model.pt")
    else:
        no_improve_epochs += 1

    if no_improve_epochs >= early_stop_threshold:
        print("Early stopping!")
        break

prompt_model.load_state_dict(torch.load("best_model.pt"))