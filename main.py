import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertForMaskedLM, DistilBertTokenizer, DistilBertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM
from utils import 

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
elif args.search == 'distilbert':
    search_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    search_model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
elif args.search == 'roberta':
    search_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    search_model = RobertaForMaskedLM.from_pretrained('roberta-base')

train_set = dataset['train']
print(next(iter(train_set)))
