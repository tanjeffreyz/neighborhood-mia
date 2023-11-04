import torch
from heapq import nlargest

def generate_neighbors(text, search_tokenizer, search_model, embedder, dropout=0.7, k=5, n=50):
    """
    For TEXT, generates a neighborhood of single-token replacements, considering the best K token replacements 
    at each position in the sequence and returning the top N neighboring sequences.
    """

    tokenized = search_tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').input_ids.to('cuda')

    seq_len = tokenized.shape[1]
    candidates = []         # Candidate tokens
    scores = []             # Candidate scores
    trg_indices = []        # Target indices associated with each candidate
    for target_index in range(1, seq_len):
        target_token = tokenized[:, target_index:target_index+1]
        
        # Apply dropout only to the target token embedding in the sequence
        embedding = embedder(tokenized)
        before = embedding[:, :target_index, :]
        dropped = torch.nn.functional.dropout(embedding[:, target_index:target_index+1, :], p=dropout)
        after = embedding[:, target_index+1:, :]
        embedding = torch.cat([before, dropped, after], dim=1)

        # Get model's predicted posterior distributions over all positions in the sequence
        probs = torch.softmax(search_model(inputs_embeds=embedding).logits, dim=2)
        original_prob = torch.gather(probs[:, target_index, :], 1, target_token)

        # Find the K most probable token replacements, not including the target token
        # Find top K+1 first because target could still appear as a candidate
        cand_probs, cands = torch.topk(probs[:, target_index, :], k + 1, dim=1)

        # Compute candidate scores to rank replacements
        denominator = 1 - original_prob
        denominator[denominator == 0] = 1E-6
        cand_scores = cand_probs / denominator
        cand_scores[cands == target_token] = float('-inf')

        candidates.append(cands)
        scores.append(cand_scores)
        trg_indices.append(torch.ones_like(cands) * target_index)
    
    # Stitch candidate information together
    candidates = torch.concat(candidates, dim=1)
    scores = torch.concat(scores, dim=1)
    trg_indices = torch.concat(trg_indices, dim=1)

    # Find top N candidates in each batch
    _, top_indices = torch.topk(scores, n, dim=1)
    top_candidates = torch.gather(candidates, 1, top_indices)
    top_trg_indices = torch.gather(trg_indices, 1, top_indices)

    # Generate and return the neighborhood of sequences
    neighborhood = []
    for i in range(top_candidates.shape[1]):
        cand = top_candidates[0, i]
        index = top_trg_indices[0, i]
        neighbor = torch.clone(tokenized)
        neighbor[0, index] = cand
        print(search_tokenizer.batch_decode(neighbor)[0])
    


def generate_neighbours_alt(text, search_tokenizer, search_model, embedder):
    token_dropout = torch.nn.Dropout(p=0.7)

    text_tokenized = search_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to('cuda')
    original_text = search_tokenizer.batch_decode(text_tokenized)[0]

    candidate_scores = dict()
    replacements = dict()

    for target_token_index in list(range(len(text_tokenized[0,:])))[1:]:

        target_token = text_tokenized[0,target_token_index]
        embeds = embedder(text_tokenized)
            
        embeds = torch.cat((embeds[:,:target_token_index,:], token_dropout(embeds[:,target_token_index,:]).unsqueeze(dim=0), embeds[:,target_token_index+1:,:]), dim=1)
        
        token_probs = torch.softmax(search_model(inputs_embeds=embeds).logits, dim=2)

        original_prob = token_probs[0,target_token_index, target_token]

        top_probabilities, top_candidates = torch.topk(token_probs[:,target_token_index,:], 6, dim=1)

        for cand, prob in zip(top_candidates[0], top_probabilities[0]):
            if not cand == target_token:

                #alt = torch.cat((text_tokenized[:,:target_token_index], torch.LongTensor([cand]).unsqueeze(0).to('cuda'), text_tokenized[:,target_token_index+1:]), dim=1)
                #alt_text = search_tokenizer.batch_decode(alt)[0]
                if original_prob.item() == 1:
                    # print("probability is one!")
                    replacements[(target_token_index, cand)] = prob.item()/(1-0.9)
                else:
                    replacements[(target_token_index, cand)] = prob.item()/(1-original_prob.item())

    
    #highest_scored_texts = max(candidate_scores.iteritems(), key=operator.itemgetter(1))[:100]
    highest_scored_texts = nlargest(100, candidate_scores, key = candidate_scores.get)

    replacement_keys = nlargest(50, replacements, key=replacements.get)
    replacements_new = dict()
    for rk in replacement_keys:
        replacements_new[rk] = replacements[rk]
    
    replacements = replacements_new
    # print("got highest scored single texts, will now collect doubles")

    highest_scored = nlargest(100, replacements, key=replacements.get)


    texts = []
    print()
    for single in highest_scored:
        alt = text_tokenized
        target_token_index, cand = single
        alt = torch.cat((alt[:,:target_token_index], torch.LongTensor([cand]).unsqueeze(0).to('cuda'), alt[:,target_token_index+1:]), dim=1)
        alt_text = search_tokenizer.batch_decode(alt)[0]
        print(alt_text)
        texts.append((alt_text, replacements[single]))


    return texts
