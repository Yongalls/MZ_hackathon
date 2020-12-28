import torch
import torch.nn as nn
from transformers import AlbertPreTrainedModel, AlbertModel, AlbertConfig
from .module import IntentClassifier

class AlbertForClassification(AlbertPreTrainedModel):
    def __init__(self, config, num_labels, dropout_rate):
        super(AlbertForClassification, self).__init__(config)
        self.albert = AlbertModel(config=config)  # Load pretrained bert
        self.intent_classifier = IntentClassifier(config.hidden_size, num_labels, dropout_rate)


    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.albert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)

        return intent_logits
