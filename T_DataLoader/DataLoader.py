from matplotlib.style import use
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import pandas as pd
import re
from datetime import datetime


class TamilDataset(Dataset):
    def __init__(self, dataset, target, tokenizer=None, device='cpu', use_cache=False, tokenizer_kwargs={}):
        self.dataset = dataset
        self.use_cache=use_cache
        self.target = target
        self.device = device
        print('self.use_cache', self.use_cache)
        if(not use_cache):
            self.tokenizer = tokenizer
            self.tokenizer_kwargs = tokenizer_kwargs
            self.tokenizer_kwargs.setdefault('max_length', 512)
            self.tokenizer_kwargs.setdefault('truncation', True)
            self.tokenizer_kwargs.setdefault('padding', 'max_length')
            # print('init done')

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        if(not self.use_cache):
            # print('before batch')
            batch = self.tokenizer(self.dataset[idx], return_tensors='pt', **self.tokenizer_kwargs)
            # print('after batch')
            return {'data': batch['input_ids'].to(self.device), 'target': torch.tensor(np.array(self.target[idx], dtype=np.float32)).to(self.device)}
        else:
            print({'data': self.dataset[idx], 'target': self.target[idx]})
            return {'data': self.dataset[idx].to(self.device), 'target': self.target[idx].to(self.device)}


def encode(tokenizer, dataset, target, device='cpu', tokenizer_kwargs={}):
    tokenizer_kwargs.setdefault('max_length', 512)
    tokenizer_kwargs.setdefault('truncation', True)
    tokenizer_kwargs.setdefault('padding', 'max_length')
    batch = tokenizer(dataset, return_tensors='pt', **tokenizer_kwargs)
    return {'data': batch['input_ids'].to(device), 'target': torch.tensor(np.array(target, dtype=np.float32)).to(device)}

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


def ReadDatasetFiles(root_path,  use_cache=False, cache_dir = './cache/dump/', test=False):
    if(use_cache):
        try:
            print(cache_dir)
            print(os.listdir(cache_dir))
            filename = cache_dir + sorted(os.listdir(cache_dir))[-1]
            print('reading from file', filename)
            pd_dataset = pd.read_pickle(filename)
            print('finished reading file from cache')
            print(pd_dataset.head())
            data = list(pd_dataset['data'])
            labels = list(pd_dataset['target'])
            dataset = TamilDataset(data, labels, use_cache=use_cache)
            return dataset
        except:
            use_cache = False
            print("cannot use cache")

    text_file_names = os.listdir(root_path)
    dataset = []

    for file_name in tqdm(text_file_names):
        with open(root_path + file_name, 'r', encoding="utf8") as f:
            dataset.append([f.read()])
    if(test):
        return dataset[:100]
    
    return dataset

def ProcessDataset(dataset, test=False):
    dataset_processed = []
    for tdata in tqdm(dataset):
        tdata = re.split('<?doc .*>|\n', tdata[0])[1:]
        dataset_processed.append(list(filter(None, [tdata_.strip('</doc>\n') for tdata_ in tdata])))

        dataset_combined = []
        for data in dataset_processed:
            dataset_combined += data
    dataset_combined = dataset_combined[:100]
    corrupted_dataset = list(map(corrupt_dataset, tqdm(dataset_combined)))
    return corrupted_dataset

def TokenizeAllData(dataset, tokenizer, device='cpu', tokenizer_kwargs = {}):
    dataset = [encode(tokenizer, data['data'], data['label'], device=device, tokenizer_kwargs=tokenizer_kwargs) for data in tqdm(dataset)]
    df = pd.DataFrame(dataset)
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    print('writing to file ', cache_dir + date_time + '.pkl')
    df.to_pickle(cache_dir + date_time + '.pkl')
    print('finished writing...')
    exit(0)



def TamilDataLoader(root_path, tokenizer_name="monsoon-nlp/tamillion", batch_size=1, device='cpu', write_cache=False, use_cache=False, cache_dir = './cache/dump/', test=False, tokenizer_kwargs = {}):

    print('use_cache', use_cache)
    dataset = ReadDatasetFiles(root_path, tokenizer_name, batch_size, test=test)
    if(use_cache):
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, test=test)
        print('returning dataloader')
        return train_dataloader

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = 0 
    processed_dataset = ProcessDataset(dataset, test)

    if(write_cache):
        TokenizeAllData(processed_dataset, tokenizer, device, tokenizer_kwargs)

    # del processed_dataset['places']
    df = pd.DataFrame(processed_dataset)
    data = list(df['data']) 
    labels = list(df['label'])
    dataset = TamilDataset(data, labels, tokenizer, device, tokenizer_kwargs = tokenizer_kwargs)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader


if __name__ == '__main__':
    GPT2CNN_kwargs = {'max_length' : 1024,}
    ElectraCNN_kwargs = {'max_length' : 512}
    tokenizer_name = 'abinayam/gpt-2-tamil'
    tokenizer_kwargs = GPT2CNN_kwargs
    root_path = './T_Dataset/train/train/'


    test=True
    write_cache = False
    cache_dir = './cache/tokenizers/' + tokenizer_name + '/'
    if(write_cache and not os.path.exists(cache_dir)):
        print("can not use cache because ", cache_dir, "does not exists")
        os.makedirs(cache_dir)
        print('creating', cache_dir)    

    train_dataloader = TamilDataLoader(root_path, tokenizer_name=tokenizer_name, batch_size = 2, device='cpu', write_cache= write_cache, cache_dir = cache_dir, test=test, tokenizer_kwargs=tokenizer_kwargs)
    batch = next(iter(train_dataloader))
    print(batch)

