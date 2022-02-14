# -*- coding: utf-8 -*-
"""
**OM NAMO NARAYANA**
"""


import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import torch

import wandb

from torch import nn

import torch.optim as optim

from datetime import datetime

import warnings


import sys,os
sys.path.append(os.getcwd())


from T_DataLoader import DataLoader
from T_models import ElectraCNN



warnings.filterwarnings("ignore", category=DeprecationWarning)
wandb.init(project="finetuning-ELECTRA", entity="team-knitts")



root_path = 'H:/sem8/nlp/dataset/train/train/'
model_dir = 'H:/sem8/nlp/implementation/checkpoint/models/'
load_model = False


config = {
    'learning_rate' : 1e-3, # to be experimented
    'batch_size' : 16, # to be changed [crashing if high]
    'epochs' : 10,
    'betas': [0.9, 0.999]
}







device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print('Using device:', device)
wandb.config = config


train_dataloader = DataLoader(root_path, tokenizer_name="monsoon-nlp/tamillion", batch_size = 4, device=device)


try:
  if(load_model == False):
    raise Exception("don't load model - exception called")
  model = ElectraCNN(device)
  optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], betas=config['betas'], eps=1e-08, weight_decay=0, amsgrad=False)
  latest_model_file = model_dir + sorted(os.listdir(model_dir))[-1]
  print('loading from ',latest_model_file)
  checkpoint = torch.load(latest_model_file)
  print('checkpoint loaded...')
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  print('model loaded...')

except: 
  print("can't load model...")
  model = ElectraCNN()
  optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], betas=config['betas'], eps=1e-08, weight_decay=0, amsgrad=False)

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
      }, model_dir + 'model' + date_time + '.pth')

      print('saving model...')

    loss.backward()
    optimizer.step()

    del outputs
    del targets


