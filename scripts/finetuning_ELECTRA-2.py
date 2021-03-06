# -*- coding: utf-8 -*-
"""finetuning-GPT-2 (2).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nDGDdX2e-U6aMGYgjazdCIjWetvKlWQH

**OM NAMO NARAYANA**
"""


#############################
###   Instructions  #########
#############################
"""
pip install wandb
pip install transformers
wandb init

# log in key : 9d5065a2de6cbad8a4869ecf568cf7277676f833

"""


import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import torch

import wandb
import re

import pandas as pd

from transformers import AutoTokenizer, AutoModel

from torch import nn
import torch.nn.functional as F


import torch.optim as optim

from datetime import datetime

import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)


wandb.init(project="finetuning-ELECTRA", entity="team-knitts")


#############################
#######   variables  ########
#############################

root_path = 'H:/sem8/nlp/dataset/train/train/'

model_dir = 'H:/sem8/nlp/implementation/checkpoint/models/'
load_model = False


config = {
    'learning_rate' : 1e-3, # to be experimented
    'batch_size' : 16, # to be changed [crashing if high]
    'epochs' : 10,
    'betas': [0.9, 0.999]
}


##############################


text_file_names = os.listdir(root_path)
dataset = []

for file_name in tqdm(text_file_names):
  with open(root_path + file_name, 'r', encoding="utf8") as f:
    dataset.append([f.read()])


dataset_processed = []
for tdata in tqdm(dataset):
  tdata = re.split('<?doc .*>|\n', tdata[0])[1:]
  dataset_processed.append(list(filter(None, [tdata_.strip('</doc>\n') for tdata_ in tdata])))
print(np.array(dataset_processed).shape)

dataset_combined = []
for data in dataset_processed:
  dataset_combined += data
del dataset_processed

# comment for training
# dataset_combined = dataset_combined[:100] # uncomment for testing



def corrupt_dataset(data):
  x = np.random.randint(2, size=1)[0]
  l = len(data) 
  s = int(len(data)/10)
  new_data = data
  if(x):
    label = 1
    places = [0] + list(np.random.randint(1, l-1, size=s))
    places.sort()

    ### logic to be optimized
    for i in range(len(places)-1):
      new_data += data[places[i]:places[i+1]-1]
    # new_data = data[:places[0]-1] + data[places[0] : places[1]-1] + data[places[1]:]
  else:
    places = [-1, -1]
    label = 0
  return {'data' : new_data, 'label' : label, 'places' : places }

corrupted_dataset = pd.DataFrame(list(map(corrupt_dataset, tqdm(dataset_combined))))
del corrupted_dataset['places']
pd_dataset = pd.DataFrame(corrupted_dataset)
del corrupt_dataset
del dataset_combined
pd_dataset.head()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print('Using device:', device)



tokenizer = AutoTokenizer.from_pretrained("monsoon-nlp/tamillion")
tokenizer.pad_token = 0 # to be changed

wandb.config = config

from torch.utils.data import Dataset

class TamilDataset(Dataset):
    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        # print(idx)
        batch = tokenizer(self.dataset[idx], truncation=True, max_length=512, padding='max_length', return_tensors='pt')
        return {'data': batch['input_ids'].to(device), 'target': torch.tensor(np.array(self.target[idx], dtype=np.float32)).to(device)}

from torch.utils.data import DataLoader


temp = TamilDataset(list(pd_dataset['data']), list(pd_dataset['label']))
train_dataloader = DataLoader(temp, batch_size=config['batch_size'], shuffle=True)
batch = next(iter(train_dataloader))
# print(batch['data'].shape, batch['target'].shape)
# output = model(batch['data'])
# print(output.logits.shape)


class CustomGPTModel(nn.Module):
    def __init__(self):
          super(CustomGPTModel, self).__init__()
          # self.model = AutoModelForCausalLM.from_pretrained("abinayam/gpt-2-tamil")
          self.model = AutoModel.from_pretrained("monsoon-nlp/tamillion")
          self.model.to(device)
          ### To be optimized
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
          # print('batch.shape', batch.shape)
          # print(batch)
          sequence_output= self.model(batch)
          # sequence_output has the following shape: (batch_size, sequence_length, 768)


          x = sequence_output.last_hidden_state
          x.to(device)
          # print('inital shape ', x.shape)
          #torch.Size([1, 512, 768])
          x = torch.transpose(x, 1, 2)
          # print('final shape ', x.shape)

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


          # print('conv4_output.shape', conv4_output.shape)
          linear1_output = F.relu(self.linear1(conv4_output.view(-1,1024))) 

          linear2_output = F.relu(self.linear2(linear1_output))
          linear3_output = F.relu(self.linear3(linear2_output))
          linear4_output = self.linear4(linear3_output)
          sigmoid_output = F.log_softmax(linear4_output, dim=1) # [batch, 2] choose the second dimension (i.e, dim = 1)
          # print('sigmoid_output.shape', sigmoid_output.shape)
          return sigmoid_output


# model_dir = '/content/drive/My Drive/model/'

try:
  if(load_model == False):
    raise Exception("don't load model - exception called")
  model = CustomGPTModel()
  optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], betas=config['betas'], eps=1e-08, weight_decay=0, amsgrad=False)
  latest_model_file = model_dir + sorted(os.listdir(model_dir))[-1]
  print('loading from ',latest_model_file)
  checkpoint = torch.load(latest_model_file)
  print('checkpoint loaded...')
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  # epoch = checkpoint['epoch']
  print('model loaded...')
except: 
  print("can't load model...")
  model = CustomGPTModel()
  optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], betas=config['betas'], eps=1e-08, weight_decay=0, amsgrad=False)
  # criterion = nn.BCELoss()

criterion = nn.NLLLoss()
model.to(device)

epochs = config['epochs']


step = 0
for epoch in range(epochs):
  for t, batch in enumerate(train_dataloader):
    step += 1
    data = batch['data']
    targets = batch['target'] 
    
    optimizer.zero_grad()   
    outputs = model.forward(data)

    del data

    outputs = outputs.squeeze(-1)
    loss = criterion(outputs.to(device), targets.type(torch.LongTensor).to(device))
    wandb.log({"loss": loss.item()}, step=step)
    wandb.watch(model, log_freq = 100)


    if(t==0):
      print({'targets': targets.cpu(), 'outputs': outputs.cpu()})

    if(epoch % 2 == 0 and t == 0 and epoch > 0):
      print('epoch:%d t:%d loss:%.2f'%(epoch, t, loss.item()))
      wandb.log({'targets': targets.cpu(), 'outputs': outputs.cpu()})
      now = datetime.now() # current date and time
      date_time = now.strftime("%Y_%m_%d_%H_%M_%S")

      print('saving model...')
      torch.save({
          'epoch' : epoch,
          'model_state_dict' : model.state_dict(),
          'optimizer_state_dict' : optimizer.state_dict(),
          'loss' : criterion,
      }, '/content/drive/My Drive/' + 'model/model' + date_time + '.pth')

      print('saving model...')

    loss.backward()
    optimizer.step()

    del outputs
    del targets


