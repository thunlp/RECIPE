from transformers import BertTokenizer
import csv
from transformers import DataCollatorForLanguageModeling
from transformers import BertForMaskedLM, DataCollatorForLanguageModeling,BertTokenizer,BertConfig
from transformers import  BertForSequenceClassification
from torch.utils.data import TensorDataset, random_split
import torch
import time
import random
import numpy as np
seed_val = 50
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
input_ids = []
attention_masks = []
sentences = []
labels = []
device = torch.device("cuda")
path = "./bookcorpus/train.tsv"
with open(path, 'r') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for idx, example_json in enumerate(reader):
        text_a = example_json['sentence'].strip()
        sentences.append(text_a)
masker = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
for sent in sentences:
    encoded_dict = tokenizer.encode_plus(
                        sent,                      
                        add_special_tokens = True, 
                        max_length = 64,           
                        padding = "max_length",
                        return_attention_mask = True,
                        truncation = True,
                        return_tensors = 'pt',     
                   )
    qinp, qlabels = masker.torch_mask_tokens(encoded_dict["input_ids"])
    input_ids.append(qinp)
    attention_masks.append(encoded_dict['attention_mask'])
    labels.append(qlabels)


input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.cat(labels)
train_dataset = TensorDataset(input_ids, attention_masks,labels)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 32

train_dataloader = DataLoader(
            train_dataset, 
            sampler = RandomSampler(train_dataset), 
            batch_size = batch_size 
        )
from transformers import BertModel, AdamW, BertConfig

model = BertForMaskedLM.from_pretrained(
    "./badpre",
    output_attentions = False, 
    output_hidden_states = False, 
)
for n,p in model.named_parameters():
    if("intermediate.dense.weight" not in n):
        p.requires_grad = False
    else:
        p.requires_grad = True

model.cuda()

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, 
                  eps = 1e-8 
                )
from transformers import get_linear_schedule_with_warmup

epochs = 8

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)



import numpy as np
from torch.utils.data import TensorDataset, random_split




for epoch_i in range(0, epochs):


    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    t0 = time.time()
    total_train_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        loss = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels).loss
        
        norm_loss = 0
        for i in range(0,12):
            norm_loss += torch.norm(model.bert.encoder.layer[i].intermediate.dense.weight)
        loss = norm_loss + loss 
        total_train_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)


    print("")


import os


output_dir = './model_afterpurification/'


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

model_to_save = model.module if hasattr(model, 'module') else model 
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

