import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BertBackbone(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.text_proj = nn.Linear(768, output_size)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).pooler_output

        return self.text_proj(bert_out)
