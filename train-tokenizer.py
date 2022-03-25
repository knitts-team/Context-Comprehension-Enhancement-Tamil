## OM NAMO NARAYANA

from cgitb import text
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import ByteLevelBPETokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from transformers import pipeline
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


from datetime import datetime
import os
from tqdm import tqdm



class CustomDataset(Dataset):
    """CustomDataset dataset."""

    def __init__(self, dataset, tokenizer=None, train=False, text_label="sentence", transform=lambda k:k):
        """
        Args:
            dataset (dataset): Dataloader from datasets.
            transform (function): Any transformation function
            tokenizer (tokenizer) : Tokenizer
        """

        if(tokenizer==None):
            tokenizer = ByteLevelBPETokenizer()
            tokenizer.pre_tokenizer = Whitespace()
            tokenizer.post_processor = TemplateProcessing(
                single="[CLS] $0 [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[("[CLS]", 1), ("[SEP]", 0)],
            )
            train = True

        self.dataset=dataset
        self.tokenizer = tokenizer
        self.transform = transform
        self.text_label = text_label

        if(train):
            self.tokenizer = self.trainTokenizer()


    def trainTokenizer(self):
        dataset_text = [ele[self.text_label] for ele in self.dataset]
        self.tokenizer.train_from_iterator(dataset_text, vocab_size =  3000, min_frequency = 2, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        return self.tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sentence = self.tokenizer.encode(self.dataset[self.text_label][idx])
        return sentence
    
    def getTokenizer(self):
        return self.tokenizer


English_Dataset = {
    'dataset_name': 'glue',
    'dataset_subset': 'cola',
    'text_label' : 'sentence', 
    'test_sentence': "Hello, y'all! How are you 😁 ?"
}

Tamil_Dataset = {
    'dataset_name': 'oscar',
    'dataset_subset': 'unshuffled_deduplicated_ta',
    'text_label' : 'text',
    'test_sentence': "பொழுது சாய்ந்து வெகு நேரமாகிவிட்டது 😁 ?"
}

curDataset = English_Dataset

save_dir = './dump/' + curDataset['dataset_name'] + './' + curDataset['dataset_subset'] + '/'

def train_tokenizer(dataset_name = "glue", dataset_subset="cola", text_label = "sentence", save_dir=None):

    if(save_dir==None):
        save_dir = './dump/' + dataset_name + '/' + dataset_subset + '/'

    tokenizer = Tokenizer(BPE())    
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $0 [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 0)],
    )


    dataset = load_dataset(dataset_name , dataset_subset, split='train')

    custom_dataset = CustomDataset(dataset, text_label=text_label)

    tokenizer = custom_dataset.getTokenizer()

    try:
        tokenizer.save_model(save_dir )
    except:
        os.makedirs(save_dir)
        tokenizer.save_model(save_dir)

    return tokenizer

def load_tokenizer(save_dir ='dump/glue/cola/'):
    tokenizer = ByteLevelBPETokenizer(save_dir + 'vocab.json', save_dir + 'merges.txt')
    return tokenizer


print('curDataset:', curDataset)

# train_tokenizer(dataset_name=curDataset['dataset_name'], dataset_subset=curDataset['dataset_subset'], text_label=curDataset['text_label'])
tokenizer = load_tokenizer(save_dir=save_dir)
output = tokenizer.encode(curDataset['test_sentence'])
print(output.tokens)



def train_model(curDataset, save_dir = "./glue/cola/"):
    config = RobertaConfig(
        vocab_size=52_000,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )
    tokenizer = RobertaTokenizerFast.from_pretrained(save_dir, max_len=512)
    model = RobertaForMaskedLM(config=config)
    print('model.num_parameters(): ', model.num_parameters())

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    dataset = load_dataset(curDataset['dataset_name'] , curDataset['dataset_subset'], split='train')
    custom_dataset = CustomDataset(dataset, tokenizer=tokenizer)

    training_args = TrainingArguments(
    output_dir="./EsperBERTo",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=custom_dataset,
    )

    trainer.train()

    trainer.save_model(save_dir)

    fill_mask = pipeline(
        "fill-mask",
        model=save_dir,
        tokenizer=save_dir
    )

    print(fill_mask("My name <mask> Arvinth."))
    print(fill_mask("<mask> is the largest country in the world."))

curDataset = English_Dataset
train_model(curDataset=curDataset, save_dir=save_dir)