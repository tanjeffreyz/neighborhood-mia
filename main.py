import argparse
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertForMaskedLM, DistilBertTokenizer, DistilBertForMaskedLM, \
    RobertaTokenizer, RobertaForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification
from utils import attack

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
    search_embedder = search_model.bert.embeddings
elif args.search == 'distilbert':
    search_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    search_model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
    search_embedder = search_model.distilbert.embeddings
elif args.search == 'roberta':
    search_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    search_model = RobertaForMaskedLM.from_pretrained('roberta-base')
    search_embedder = search_model.roberta.embeddings

target_model = target_model.to('cuda')
search_model = search_model.to('cuda')

assert args.num_iters <= len(dataset['test']), 'Too many iterations'

# Calculate and save neighborhood scores for both the training and test sets
with torch.no_grad():
    test_scores = []
    test_losses = []
    test_correct = 0
    test_iter = iter(dataset['test'])
    for _ in tqdm(range(args.num_iters), desc='Test'):
        data = next(test_iter)
        score, loss, correct = attack(data, target_tokenizer, target_model, search_tokenizer, search_model, search_embedder)
        test_scores.append(score)
        test_losses.append(loss)
        test_correct += correct
    np.save('scores/test_scores.npy', np.array(test_scores))
    print(f'Validation loss: {sum(test_losses) / len(test_losses)}')
    print(f'Validation accuracy: {test_correct / len(test_losses)}')

    train_scores = []
    train_losses = []
    train_correct = 0
    train_iter = iter(dataset['train'])
    for _ in tqdm(range(args.num_iters), desc='Train'):
        data = next(train_iter)
        score, loss, correct = attack(data, target_tokenizer, target_model, search_tokenizer, search_model, search_embedder)
        train_scores.append(score)
        train_losses.append(loss)
        train_correct += correct
    np.save('scores/train_scores.npy', np.array(train_scores))
    print(f'Training loss: {sum(train_losses) / len(train_losses)}')
    print(f'Training accuracy: {train_correct / len(train_losses)}')
