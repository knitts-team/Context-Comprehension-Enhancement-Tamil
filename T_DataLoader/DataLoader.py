from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import pandas as pd
import re

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




def TamilDataLoader(root_path, tokenizer_name="monsoon-nlp/tamillion", batch_size=1, device='cpu'):


    text_file_names = os.listdir(root_path)
    dataset = []

    for file_name in tqdm(text_file_names):
        with open(root_path + file_name, 'r', encoding="utf8") as f:
            dataset.append([f.read()])


    dataset_processed = []

    for tdata in tqdm(dataset):
        tdata = re.split('<?doc .*>|\n', tdata[0])[1:]
        dataset_processed.append(list(filter(None, [tdata_.strip('</doc>\n') for tdata_ in tdata])))
        # print(np.array(dataset_processed).shape)

        dataset_combined = []
        for data in dataset_processed:
            dataset_combined += data

    # del dataset_processed
    # del dataset
    corrupted_dataset = pd.DataFrame(list(map(corrupt_dataset, tqdm(dataset_combined))))
    # del corrupted_dataset['places']
    pd_dataset = pd.DataFrame(corrupted_dataset)
    # del corrupt_dataset
    # del dataset_combined
    pd_dataset.head()



    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = 0 

    data = list(pd_dataset['data']) 
    labels = list(pd_dataset['label'])

    dataset = TamilDataset(data, labels, tokenizer, device)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader


