from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the model and tokenizer
bert_name = 'bert-base-chinese'  # Or any other pre-trained model
tokenizer = BertTokenizer.from_pretrained(bert_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained(bert_name, num_labels=5, ignore_mismatched_sizes=True).to(device)
model.load_state_dict(torch.load(r'D:\zxx\public_sentiment\model\bert_base_0.67.pth', map_location=device))
# Set the model to evaluation mode
model.eval()

# Example input
text = "我晕，五一放假咋还要调休啊，就放一天？？？"
inputs = tokenizer(text, return_tensors="pt")
inputs = inputs.to(device)
label_key_map = {0:'sadness', 1:'happiness', 2:'anger', 3:'fear', 4:'disgust'}
# Make predictions
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    print(label_key_map[predictions.item()])  # The predicted label index
