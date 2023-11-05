import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertForMaskedLM, DistilBertTokenizer, DistilBertForMaskedLM, \
    RobertaTokenizer, RobertaForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification
from utils import get_neighborhood_score, get_log_likelihood

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['ag_news'], default='ag_news')
parser.add_argument('--search', type=str, choices=['bert', 'distilbert', 'roberta'], default='bert')
parser.add_argument('--num_iters', type=int, default=1_000)
args = parser.parse_args()

if args.dataset == 'ag_news':
    dataset = load_dataset('ag_news')
    # target_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # target_model = GPT2LMHeadModel.from_pretrained('gpt2')
    # target_tokenizer.pad_token = target_tokenizer.eos_token
    target_tokenizer = AutoTokenizer.from_pretrained("fabriceyhc/bert-base-uncased-ag_news")
    target_model = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-ag_news")

if args.search == 'bert':
    search_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    search_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    embedder = search_model.bert.embeddings
elif args.search == 'distilbert':
    search_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    search_model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
    embedder = search_model.distilbert.embeddings
elif args.search == 'roberta':
    search_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    search_model = RobertaForMaskedLM.from_pretrained('roberta-base')
    embedder = search_model.roberta.embeddings

target_model = target_model.to('cuda')
search_model = search_model.to('cuda')

assert args.num_iters <= len(dataset['test']), 'Too many iterations'

# Calculate and save neighborhood scores for both the training and test sets
test_scores = []
test_losses = []
test_iter = iter(dataset['test'])
for _ in tqdm(range(args.num_iters), desc='Test'):
    data = next(test_iter)
    score = get_neighborhood_score(data['text'], data['label'], target_tokenizer, target_model, search_tokenizer, search_model, embedder)
    test_scores.append(score)
    test_losses.append(get_log_likelihood(data['text'], data['label'], target_tokenizer, target_model))
np.save('scores/test_scores.npy', np.array(test_scores))
print(f'Average validation loss: {sum(test_losses) / len(test_losses)}')

train_scores = []
train_losses = []
train_iter = iter(dataset['train'])
for _ in tqdm(range(args.num_iters), desc='Train'):
    data = next(train_iter)
    score = get_neighborhood_score(data['text'], data['label'], target_tokenizer, target_model, search_tokenizer, search_model, embedder)
    train_scores.append(score)
    train_losses.append(get_log_likelihood(data['text'], data['label'], target_tokenizer, target_model))
np.save('scores/train_scores.npy', np.array(train_scores))
print(f'Average training loss: {sum(train_losses) / len(train_losses)}')
