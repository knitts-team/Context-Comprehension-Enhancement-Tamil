## OM NAMO NARAYANA

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from datasets import load_dataset
from datetime import datetime
import os
from tqdm import tqdm
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """CustomDataset dataset."""

    def __init__(self, dataset, transform=lambda k:k):
        """
        Args:
            dataset (dataset): Dataloader from datasets.
            transform (function): Any transformation function
        """
        self.dataset=dataset
        self.transform = transform


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sentence = self.dataset[idx]["text"]
        return sentence


save_dir ='dump/glue/cola/'

def train_tokenizer(save_dir='dump/glue/cola/'):
    tokenizer = Tokenizer(BPE())    
    tokenizer.pre_tokenizer = Whitespace()
    dataset = load_dataset('oscar' ,'unshuffled_deduplicated_ta', split='train')
    print('sample dataset: ', dataset[0], 'length: ', len(dataset))

    custom_dataset = CustomDataset(dataset)


    trainer = BpeTrainer(vocab_size =  3000, min_frequency = 2, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train_from_iterator(custom_dataset, trainer=trainer)

    output = tokenizer.encode("பொழுது சாய்ந்து வெகு நேரமாகிவிட்டது 😁")
    print(output.tokens)

    try:
        tokenizer.save(save_dir + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'.json')
    except:
        os.makedirs(save_dir)
        tokenizer.save(save_dir + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'.json')

    return tokenizer

def load_tokenizer(save_dir ='dump/glue/cola/'):
    ls = os.listdir(save_dir)
    ls.sort()
    filename = save_dir+ls[-1]
    tokenizer = Tokenizer.from_file(filename)
    return tokenizer

# train_tokenizer(save_dir=save_dir)
tokenizer = load_tokenizer(save_dir=save_dir)
output = tokenizer.encode("பொழுது சாய்ந்து வெகு நேரமாகிவிட்டது 😁")
print(output.tokens)

