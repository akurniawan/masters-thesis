import torch.nn as nn
from transformers import AutoModel

from ..modules.adapters import AdapterWrapper


class BertEncoderWithAdapter(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = AutoModel.from_pretrained("bert-base-multilingual-cased")
        # Inject adapter on BERT layers
        for idx, bert_encoder in enumerate(self.model.bert.encoder.layer):
            input_size = bert_encoder.output.dense.weight.size(-1)
            self.model.bert.encoder.layer[idx] = AdapterWrapper(input_size, bert_encoder)

    