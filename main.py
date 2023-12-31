import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset
from transformers import BertTokenizer, BertForMaskedLM, DistilBertTokenizer, DistilBertForMaskedLM, \
    RobertaTokenizer, RobertaForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from utils import get_neighborhood_score, get_loss

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['gpt2', 'bert'], default='gpt2')
parser.add_argument('--search', type=str, choices=['bert', 'distilbert', 'roberta'], default='bert')
parser.add_argument('--num_iters', type=int, default=1_000)
parser.add_argument('--max_length', type=int, default=64)
parser.add_argument('--mia_type', type=str, choices=['loss', 'neighbor'], default='neighbor')
args = parser.parse_args()

now = datetime.now()
folder = os.path.join(
    'experiments',
    now.strftime('%m_%d_%Y'),
    now.strftime('%H_%M_%S')
)
if not os.path.isdir(folder):
    os.makedirs(folder)

# Save configuration to text file
with open(os.path.join(folder, 'config.txt'), 'w') as file:
    file.write(str(args))

dataset = load_dataset('ag_news').shuffle()
if args.model == 'bert':
    tokenizer = AutoTokenizer.from_pretrained("fabriceyhc/bert-base-uncased-ag_news")
    model = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-ag_news")
    causal = False
elif args.model == 'gpt2':
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # model = AutoModelForCausalLM.from_pretrained("DunnBC22/gpt2-Causal_Language_Model-AG_News")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("../models/checkpoint-17100")
    tokenizer.pad_token = tokenizer.eos_token
    causal = True

if args.search == 'bert':
    search_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    search_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    search_embedder = search_model.bert.embeddings
elif args.search == 'distilbert':
    search_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    search_model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
    search_embedder = search_model.distilbert.embeddings
elif args.search == 'roberta':
    search_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    search_model = RobertaForMaskedLM.from_pretrained('roberta-base')
    search_embedder = search_model.roberta.embeddings

model = model.to('cuda')
model.eval()
search_model = search_model.to('cuda')
search_model.eval()

max_iters = len(dataset['test'])
assert args.num_iters <= max_iters, f'Too many iterations, maximum {max_iters}'


def eval(name):
    loss = 0
    scores = []
    iterator = iter(dataset[name])
    for _ in tqdm(range(args.num_iters), desc=name):
        data = next(iterator)
        text = data['text']
        label = data['label']
        with torch.no_grad():
            l = get_loss(text, label, tokenizer, model, args.max_length, causal=causal)
            if args.mia_type == 'loss':
                score = l
            elif args.mia_type == 'neighbor':
                score = get_neighborhood_score(text, label, tokenizer, model, search_tokenizer, search_model, search_embedder, args.max_length, causal=causal)
            loss += l
        scores.append(score)
    print('Loss:', loss / args.num_iters)
    return scores


# Calculate and save neighborhood scores for both the training and test sets
train_scores = eval('train')
np.save(os.path.join(folder, 'train_scores.npy'), np.array(train_scores))
test_scores = eval('test')
np.save(os.path.join(folder, 'test_scores.npy'), np.array(test_scores))
