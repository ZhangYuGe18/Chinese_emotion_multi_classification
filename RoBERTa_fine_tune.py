from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset
import torch
from transformers import Trainer, TrainingArguments
import pandas as pd
from utility.RoBERTa_utility import *
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("nanaaaa/emotion_chinese_english")
num_labels = 5
model = AutoModelForSequenceClassification.from_pretrained("nanaaaa/emotion_chinese_english", num_labels=num_labels,ignore_mismatched_sizes=True)
    
data = pd.read_excel(r'D:\zxx\emotive_classification\data\天池比赛情绪分类训练数据集.xlsx')
print(data['label'].unique())

label_map = {'sadness':0, 'happiness':1, 'anger':2, 'fear':3, 'disgust':4}
text_list = data['content'].tolist()[:27532]
train_labels = data['label'].tolist()[:27532]
label_list = [label_map[label] for label in train_labels]

val_text_list = data['content'].tolist()[27533:]
true_label = data['label'].tolist()[27533:]
val_label_list = [label_map[label] for label in true_label]

# Prepare your data
max_length = 128
train_dataset = TextClassificationDataset(text_list, label_list, tokenizer,max_length)
eval_dataset = TextClassificationDataset(val_text_list, val_label_list, tokenizer,max_length)

# Define the training and evaluation parameters
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=50,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=1000,
    evaluation_strategy='epoch', # modify evaluation strategy to evaluate every epoch
    load_best_model_at_end=True,
    save_strategy='epoch', # modify save strategy to save every epoch
)
# Fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

eval_result = trainer.evaluate(eval_dataset)
print(eval_result)




