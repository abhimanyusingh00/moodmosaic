from typing import Dict, Any
import torch
from torch import nn
from transformers import AutoModel, AutoConfig

class EmotionClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ) -> Dict[str, Any]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())

        return {"loss": loss, "logits": logits}
