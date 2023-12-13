import torch.nn as nn
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer, BertModel, BertConfig


# Convert data to BERT format
class CustomDataset(Dataset):
    def __init__(self, traces, labels, tokenizer, max_length=400):
        self.traces = traces
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if index >= len(self.labels):
            index = index % len(self.labels)  # Wrap around the index
        
        trace = self.traces[index]
        label = self.labels.iloc[index]

        # Convert trace to string format and tokenize
        trace_str = ' '.join([str(x) for x in trace])
        encoding = self.tokenizer(trace_str, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }




# Create the BERT model
class BERTModel(nn.Module):
    def __init__(self, num_classes):
        super(BERTModel, self).__init__()
        config = BertConfig()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits