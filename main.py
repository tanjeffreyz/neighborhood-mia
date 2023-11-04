import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertForMaskedLM, DistilBertTokenizer, DistilBertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM
from utils import generate_neighbors, generate_neighbours_alt

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['ag_news'], default='ag_news')
parser.add_argument('--search', type=str, choices=['bert', 'distilbert', 'roberta'], default='bert')
args = parser.parse_args()

if args.dataset == 'ag_news':
    dataset = load_dataset("ag_news")
    target_tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news")
    target_model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news")

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

search_model = search_model.to('cuda')

# Perform the attack
for data in dataset['train']:
    from time import time
    text = data['text']
    start = time()
    generate_neighbors(text, search_tokenizer, search_model, embedder)
    print(time() - start)
    start = time()
    generate_neighbours_alt(text, search_tokenizer, search_model, embedder)
    print(time() - start)
    exit(0)
