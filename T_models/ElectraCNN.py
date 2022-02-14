from transformers import AutoModel
import torch
from torch import nn
import torch.nn.functional as F

class ElectraCNN(nn.Module):
    def __init__(self, device='cpu'):
          super(ElectraCNN, self).__init__()
          self.device = device
          self.model = AutoModel.from_pretrained("monsoon-nlp/tamillion")
          self.model.to(self.device)
          self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3, padding=1)
          self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
          self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
          self.conv4 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
          self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
          self.linear1 = nn.Linear(1024, 256)
          self.linear2 = nn.Linear(256, 64)
          self.linear3 = nn.Linear(64, 16)
          self.linear4 = nn.Linear(16, 2)


    def forward(self, batch):
          batch = batch.squeeze(1)
          sequence_output= self.model(batch)
          # sequence_output has the following shape: (batch_size, sequence_length, 768)


          x = sequence_output.last_hidden_state
          x.to(self.device)
          x = torch.transpose(x, 1, 2)

          conv1_output = self.conv1(x)
          conv1_output = self.pool(conv1_output)

          # [batch_size, 256, 256]

          conv2_output = self.conv2(conv1_output)
          conv2_output = self.pool(conv2_output)

          # [batch_size, 128, 128]

          conv3_output = self.conv3(conv2_output)
          conv3_output = self.pool(conv3_output)

          # [batch_size, 64, 64]

          conv4_output = self.conv4(conv3_output)
          conv4_output = self.pool(conv4_output)

          # [batch_size, 32, 32]


          linear1_output = F.relu(self.linear1(conv4_output.view(-1,1024))) 

          linear2_output = F.relu(self.linear2(linear1_output))
          linear3_output = F.relu(self.linear3(linear2_output))
          linear4_output = self.linear4(linear3_output)
          sigmoid_output = F.log_softmax(linear4_output, dim=1) # [batch, 2] choose the second dimension (i.e, dim = 1)
          return sigmoid_output