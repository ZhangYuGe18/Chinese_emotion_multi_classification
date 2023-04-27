import statistics
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def predict_by_prompt(transfomer,data):
    model = AutoModelForSequenceClassification.from_pretrained(transfomer, ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained(transfomer)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    text_list = data['content'].tolist()
    label_list = data['label'].tolist()
    unique_labels = sorted(list(set(label_list)))
    y_pred = []
    y_true = []
    probs_emotions = []
    for index, text in enumerate(tqdm(text_list)):
        premise = text
        probs = []
        with torch.no_grad():
            for label in unique_labels:
                prompt = label
                x = tokenizer.encode(premise, prompt, return_tensors='pt', truncation_strategy='do_not_truncate')
                x = x.to(device)
                logits = model(x)[0]
                entail_contradiction_logits = logits[:, [0, 1]]
                prob_label_is_true = entail_contradiction_logits.softmax(dim=1)[:, 1]
                probs.append(prob_label_is_true.detach().cpu().numpy()[0])
        y_pred.append(unique_labels[np.argmax(np.array(probs))])
        probs_emotions.append(probs)
        y_true.append(label_list[index])
    result = pd.DataFrame({'text': text_list, 'True': y_true, 'predict': y_pred, })
    return result
def predict_by_RoBERT(tokenizer,model,data):
    text_list = data['content'].tolist()
    label_list = data['label'].tolist()
    pre_label = []
    for input_text in text_list:
        encoded_input = tokenizer(input_text, padding=True, truncation=True, max_length=128, return_tensors='pt')
        outputs = model(**encoded_input)
        predictions = torch.argmax(outputs.logits, dim=1)
        label_map = {0:'sadness',1:'happiness',2:'anger', 3:'fear',4:'disgust'}
        pre_label.append(label_map[predictions.item()])
    result = pd.DataFrame({'text': text_list, 'True': label_list, 'predict': pre_label, })
    return result
def predict_by_BERT(tokenizer,model,data,device):
    text_list = data['content'].tolist()
    label_list = data['label'].tolist()
    pre_label = []
    label_key_map = {0:'sadness', 1:'happiness', 2:'anger', 3:'fear', 4:'disgust'}
    for text in text_list:
        inputs = tokenizer(text, return_tensors="pt")
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            pre_label.append(label_key_map[predictions.item()])
    result = pd.DataFrame({'text': text_list, 'True': label_list, 'predict': pre_label, })
    return result
if __name__ == "__main__":
    data = pd.read_excel(r'D:\zxx\emotive_classification\data\天池比赛情绪分类训练数据集.xlsx')
    data.set_index('index', inplace=True)
    print(data['label'].unique())
    transfomer = "nanaaaa/emotion_chinese_english"
    result = predict_by_prompt(transfomer, data[:15],index=False)
    result.to_csv(r'text.csv')
