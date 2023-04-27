from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the trained model and tokenizer
model_path = r'D:\zxx\public_sentiment\results\checkpoint-1721'
tokenizer = AutoTokenizer.from_pretrained("nanaaaa/emotion_chinese_english")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Define the input text
input_text = "明天星期五，最后上一天班美美回家放五一！"


# Tokenize the input text
encoded_input = tokenizer(input_text, padding=True, truncation=True, max_length=128, return_tensors='pt')
# Make predictions
outputs = model(**encoded_input)
predictions = torch.argmax(outputs.logits, dim=1)
label_map = {0:'sadness',1:'happiness',2:'anger', 3:'fear',4:'disgust'}
# Print the predicted label
print(label_map[predictions.item()])