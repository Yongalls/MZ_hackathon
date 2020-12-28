import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
from .module import IntentClassifier

class BertForClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels, dropout_rate):
        super(BertForClassification, self).__init__(config)
        self.bert = BertModel(config=config)  # Load pretrained bert
        self.intent_classifier = IntentClassifier(config.hidden_size, num_labels, dropout_rate)


    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)

        return intent_logits
