from transformers import AutoModelForCausalLM
import torch
from torch import nn
import torch.nn.functional as F

class GPT2CNN(nn.Module):
    def __init__(self, device='cpu'):
          super(GPT2CNN, self).__init__()
          self.device = device
          self.model = AutoModelForCausalLM.from_pretrained("abinayam/gpt-2-tamil")
          # self.model = SentenceTransformer("abinayam/gpt-2-tamil")
          # self.model.config.pad_token_id = tokenizer.eos_token
          self.model.config.pad_token_id = 295
          self.model.to(device)
          self.conv1 = nn.Conv1d(in_channels=50257, out_channels=256, kernel_size=5, padding=2)
          self.conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
          self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
          self.conv4 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
          self.conv5 = nn.Conv1d(32, 2, kernel_size=3, padding=1)
        #   self.conv1 =  nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        #   self.conv2 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        #   self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        #   self.conv4 = nn.Conv1d(128, 12, kernel_size=3, padding=1)
        #   self.conv5 = nn.Conv1d(12, 1, kernel_size=3, padding=1)
          self.pool = nn.MaxPool1d(kernel_size=4, stride=4)


    def forward(self, batch):
          batch = batch.squeeze(1)
      #     print('batch.shape', batch.shape)
          sequence_output= self.model(batch)
          # sequence_output has the following shape: (batch_size, sequence_length, 2)


          x = sequence_output.logits
          x.to(self.device)
          print('x.shape', x.shape) # torch.Size([1, 2])
          x = torch.transpose(x, 1, 2)

      #     # [batch_size, 1024, 2]

          conv1_output = self.conv1(x)
          conv1_output = self.pool(conv1_output)

      #     # [batch_size, 512, 2]

          conv2_output = self.conv2(conv1_output)
          conv2_output = self.pool(conv2_output)
      #     # [batch_size, 256, 2]

          conv3_output = self.conv3(conv2_output)
          conv3_output = self.pool(conv3_output)
      #     # [batch_size, 128, 2]

          conv4_output = self.conv4(conv3_output)
          conv4_output = self.pool(conv4_output)
      #     ## [batch_size, 12, 2]

          conv5_output = self.conv5(conv4_output)
          conv5_output = self.pool(conv5_output)

          conv5_output = torch.squeeze(conv5_output, 1)
          print('conv5_output.shape', conv5_output.shape)
          sigmoid_output = F.log_softmax(conv5_output, dim=1) # [batch, 2] choose the second dimension (i.e, dim = 1)
          print('sigmoid_output.shape', sigmoid_output.shape)
        #   sigmoid_output = F.log_softmax(x, dim=1) # [batch, 2] choose the second dimension (i.e, dim = 1)

          return sigmoid_output