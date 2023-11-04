def generate_neighbors(text, search_tokenizer, search_model):
    tokenized = search_tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').input_ids.to('cuda')    

