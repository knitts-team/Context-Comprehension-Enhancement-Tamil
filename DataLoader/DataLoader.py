from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from transformers import AutoTokenizer



class TamilDataset(Dataset):
    def __init__(self, dataset, target, tokenizer, device='cpu'):
        self.dataset = dataset
        self.target = target
        self.device = device
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        batch = self.tokenizer(self.dataset[idx], truncation=True, max_length=512, padding='max_length', return_tensors='pt')
        return {'data': batch['input_ids'].to(self.device), 'target': torch.tensor(np.array(self.target[idx], dtype=np.float32)).to(self.device)}


def TamilDataLoader(data, labels, tokenizer_name="monsoon-nlp/tamillion", batch_size=1):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = 0 # to 

    temp = TamilDataset(data, labels)
    train_dataloader = DataLoader(temp, batch_size=batch_size, shuffle=True)

    return train_dataloader


