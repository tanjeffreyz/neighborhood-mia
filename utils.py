import torch
from heapq import nlargest


def generate_neighbors(text, search_tokenizer, search_model, search_embedder, p=0.7, k=5, n=50):
    """
    For TEXT, generates a neighborhood of single-token replacements, considering the best K token replacements 
    at each position in the sequence and returning the top N neighboring sequences.
    """

    tokenized = search_tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').input_ids.to('cuda')
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


def get_log_likelihood(text, label, tokenizer, model):
    tokenized = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').input_ids.to('cuda')
    return model(tokenized, labels=torch.tensor([label]).to('cuda')).loss.item()


def get_neighborhood_score(text, label, target_tokenizer, target_model, search_tokenizer, search_model, search_embedder):
    original_score = -get_log_likelihood(text, label, target_tokenizer, target_model)

    # Compute log likelihood for each neighbor in the neighborhood
    neighbor_scores = []
    neighbors = generate_neighbors(text, search_tokenizer, search_model, search_embedder)
    for n in neighbors:
        neighbor_scores.append(-get_log_likelihood(n, label, target_tokenizer, target_model))
    mean_neighbor_score = sum(neighbor_scores) / len(neighbor_scores)

    return original_score - mean_neighbor_score


def attack(data, target_tokenizer, target_model, search_tokenizer, search_model, search_embedder):
    score = get_neighborhood_score(data['text'], data['label'], target_tokenizer, target_model, search_tokenizer, search_model, search_embedder)

    # Evaluate model
    tokenized = target_tokenizer(data['text'], padding=True, truncation=True, max_length=512, return_tensors='pt').input_ids.to('cuda')
    output = target_model(tokenized, labels=torch.tensor([data['label']]).to('cuda'))

    return score, output.loss.item(), int(torch.argmax(output.logits).item() == data['label'])
