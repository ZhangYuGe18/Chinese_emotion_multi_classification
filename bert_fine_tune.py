import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import pandas as pd
from tqdm import tqdm
from utility.bert_utility import *
from sklearn.metrics import f1_score

data = pd.read_excel(r'D:\zxx\emotive_classification\data\天池比赛情绪分类训练数据集.xlsx')
print(data['label'].unique())

# 设置参数
# 设置预训练bert模型
bert_name = "bert-base-chinese"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained(bert_name)
model = BertForSequenceClassification.from_pretrained(bert_name, num_labels=5,ignore_mismatched_sizes=True).to(device)

# 设置标签映射map
label_map = {'sadness':0, 'happiness':1, 'anger':2, 'fear':3, 'disgust':4}
text_list = data['content'].tolist()[:27532]
train_labels = data['label'].tolist()[:27532]
label_list = [label_map[label] for label in train_labels]

#超参数
max_length = 128
batch_size = 64
epochs = 15
learning_rate = 2e-5

val_text_list = data['content'].tolist()[27533:]
true_label = data['label'].tolist()[27533:]
val_label_list = [label_map[label] for label in true_label]

# 设置标签映射map
label_key_map = {0:'sadness', 1:'happiness', 2:'anger', 3:'fear', 4:'disgust'}
# 数据加载
train_dataset = SentimentDataset(text_list, label_list, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 优化器和学习率调整器
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs)
# 训练
max_f1 = 0
for epoch in range(epochs):
    model.train()
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
    pre_label = []
    model.eval()
    for n,text in enumerate(tqdm(val_text_list)):
        probabilities, predicted_label = predict(text, tokenizer, model, device,max_length)
        pre_label.append(label_key_map[predicted_label])
    f1 = f1_score(true_label, pre_label, average='weighted')
    if f1 > max_f1:
        max_f1 = f1
        # 保存模型
        torch.save(model.state_dict(), 'model/sentiment_model_'+ str(max_f1) +'.pth')
    print(f"F1 score: {f1}, max_f1 score: {max_f1}")