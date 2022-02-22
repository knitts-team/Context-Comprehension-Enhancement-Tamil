from transformers import AutoModelForSequenceClassification
from torch import nn
import torch.nn.functional as F

class GPT2(nn.Module):
    def __init__(self, device='cpu'):
          super(GPT2, self).__init__()
          self.device = device
          self.model = AutoModelForSequenceClassification.from_pretrained("abinayam/gpt-2-tamil")

    def forward(self, batch):
          batch = batch.squeeze(1)
          sequence_output= self.model(batch)
          x = sequence_output.logits
          x.to(self.device)
          sigmoid_output = F.log_softmax(x, dim=1) # [batch, 2] choose the second dimension (i.e, dim = 1)
          return sigmoid_output