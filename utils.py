import torch
from heapq import nlargest


def generate_neighbors(text, search_tokenizer, search_model, search_embedder, p=0.7, k=5, n=50):
    """
    For TEXT, generates a neighborhood of single-token replacements, considering the best K token replacements 
    at each position in the sequence and returning the top N neighboring sequences.
    """

    tokenized = search_tokenizer(text, padding=True, truncation=True, return_tensors='pt').input_ids.to('cuda')
    dropout = torch.nn.Dropout(p)

    seq_len = tokenized.shape[1]
    cand_scores = {}
    for target_index in range(1, seq_len):
        target_token = tokenized[0, target_index]
        
        # Apply dropout only to the target token embedding in the sequence
        embedding = search_embedder(tokenized)
        embedding = torch.cat([
            embedding[:, :target_index, :], 
            dropout(embedding[:, target_index:target_index+1, :]), 
            embedding[:, target_index+1:, :]
        ], dim=1)

        # Get model's predicted posterior distributions over all positions in the sequence
        probs = torch.softmax(search_model(inputs_embeds=embedding).logits, dim=2)
        original_prob = probs[0, target_index, target_token].item()

        # Find the K most probable token replacements, not including the target token
        # Find top K+1 first because target could still appear as a candidate
        cand_probs, cands = torch.topk(probs[0, target_index, :], k + 1)
        
        # Score each candidate
        for prob, cand in zip(cand_probs, cands):
            if cand == target_token:
                continue
            denominator = (1 - original_prob) if original_prob < 1 else 1E-6
            score = prob.item() / denominator
            cand_scores[(cand, target_index)] = score
    
    # Generate and return the neighborhood of sequences
    neighborhood = []
    top_keys = nlargest(n, cand_scores, key=cand_scores.get)
    for cand, index in top_keys:
        neighbor = torch.clone(tokenized)
        neighbor[0, index] = cand
        neighborhood.append(search_tokenizer.batch_decode(neighbor)[0])
    
    return neighborhood


def forward(text, label, tokenizer, model):
    tokenized = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    tokenized['labels'] = tokenized['input_ids'] if model.is_causal else torch.tensor([label])
    for k, v in tokenized.items():
        tokenized[k] = v.to('cuda')
    return model(**tokenized)


def get_neighborhood_score(text, label, target_tokenizer, target_model, search_tokenizer, search_model, search_embedder):
    original_output = forward(text, label, target_tokenizer, target_model)
    original_score = -original_output.loss.item()

    # Compute log likelihood for each neighbor in the neighborhood
    neighbor_scores = []
    neighbors = generate_neighbors(text, search_tokenizer, search_model, search_embedder)
    for n in neighbors:
        neighbor_scores.append(-forward(n, label, target_tokenizer, target_model).loss.item())
    mean_neighbor_score = sum(neighbor_scores) / len(neighbor_scores)

    return original_score - mean_neighbor_score, original_output


def attack(data, target_tokenizer, target_model, search_tokenizer, search_model, search_embedder):
    text = data['text']
    label = data['label']
    score, original_output = get_neighborhood_score(text, label, target_tokenizer, target_model, search_tokenizer, search_model, search_embedder)
    correct = int(torch.argmax(original_output.logits).item() == label)
    return score, original_output.loss.item(), correct
