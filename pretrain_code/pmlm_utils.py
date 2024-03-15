import torch
import torch.nn.functional as F

def mask_batch_logits(pair_logits, masked_tokens):
    '''
    Generate pairwise logits for the batch
    '''
    # (B, T) -> (B, T, T)
    pair_masked_tokens = (masked_tokens.long().unsqueeze(-2) * masked_tokens.long().unsqueeze(-1)).bool()
    pair_masked_tokens.diagonal(dim1=-1, dim2=-2)[:] = False
    logits = pair_logits[pair_masked_tokens, :]
    return logits

def mask_batch_target(target, masked_tokens, vocab_size):
    '''
    Generate pairwise target for the batch
    '''
    sizes = masked_tokens.sum(-1)
    device = masked_tokens.device
    idxs = [torch.LongTensor([0]).to(device)] + [torch.sum(sizes[:i+1]).long() for i in range(len(sizes)-1)]
    pair_target_list = []
    for idx, size in zip(idxs, sizes):
        b_target = target[idx:idx+size]
        b_target_pair = (b_target * vocab_size).unsqueeze(-2) + b_target.unsqueeze(-1)
        pair_target_list.append(b_target_pair.masked_select(~torch.eye(size, dtype=bool, device=device)).view(-1))
    pair_target = torch.cat(pair_target_list).to(device)
    return pair_target

def calc_batch_pair_probs_from_mlm_logits(mlm_logits, masked_tokens):

    mlm_lprobs = F.log_softmax(mlm_logits, dim=-1, dtype=torch.float32) # N

    sizes = masked_tokens.sum(-1)
    device = masked_tokens.device
    idxs = [torch.LongTensor([0]).to(device)] + [torch.sum(sizes[:i+1]).long() for i in range(len(sizes)-1)]
    
    pair_lprobs = []
    for idx, size in zip(idxs, sizes):
        if size == 0:
            continue
        b_mlm_lprobs = mlm_lprobs[idx:idx+size]
        b_pair_lprobs = b_mlm_lprobs[:,None,None,:] + b_mlm_lprobs[None,:,:,None]  # N, N, C, C # addition(log_prob) = multiplication(prob)

        b_pair_lprobs = b_pair_lprobs.reshape(b_pair_lprobs.size(0), b_pair_lprobs.size(1), -1)

        pair_masked_tokens = torch.ones(size, size, dtype=torch.bool).to(device)
        pair_masked_tokens.diagonal()[:] = False

        pair_lprobs.append(b_pair_lprobs[pair_masked_tokens, :])

        # b_pair_lprobs = b_pair_lprobs.reshape(-1, b_pair_lprobs.size(-1))
        # pair_lprobs.append(b_pair_lprobs)

    pair_lprobs = torch.cat(pair_lprobs).to(device)

    return pair_lprobs

def calc_batch_pair_targets_from_mlm_targets(targets, masked_tokens, vocab_size):

    sizes = masked_tokens.sum(-1)
    device = masked_tokens.device
    idxs = [torch.LongTensor([0]).to(device)] + [torch.sum(sizes[:i+1]).long() for i in range(len(sizes)-1)]

    pair_target = []
    for idx, size in zip(idxs, sizes):
        if size == 0:
            continue
        b_target = targets[idx:idx+size]
        b_target_pair = (b_target * vocab_size).unsqueeze(-2) + b_target.unsqueeze(-1)
        # pair_target.append(b_target_pair.view(-1))
        pair_target.append(b_target_pair.masked_select(~torch.eye(size, dtype=bool)).view(-1))
    
    pair_target = torch.cat(pair_target).to(device)

    return pair_target

def calc_batch_mlm_lprobs_from_pair_logits(pair_logits, masked_tokens, vocab_size):
    ### The diagonal elements are removed in pair_logits
    pair_probs = F.softmax(pair_logits, dim=-1, dtype=torch.float32) # (N, vocab_size**2)
    sizes = masked_tokens.sum(-1)
    device = masked_tokens.device
    idxs = [torch.LongTensor([0]).to(device)] + [torch.sum(sizes[:i+1] * (sizes[:i+1]-1)).long() for i in range(len(sizes)-1)]

    lprobs_list = []
    for idx, size in zip(idxs, sizes):
        if size <= 1:
            continue
        b_pair_probs = pair_probs[idx:idx+size*(size-1)]
        b_pair_probs = b_pair_probs.view(size, size - 1, -1)
        b_pair_probs = b_pair_probs.view(size, size - 1, vocab_size, vocab_size)
        b_probs_sum = b_pair_probs.sum(1).sum(-1) # (size, vocab_size)
        b_probs = b_probs_sum / b_probs_sum.sum(-1, keepdim=True)
        b_lprobs = torch.log(b_probs)
        lprobs_list.append(b_lprobs)

    return torch.cat(lprobs_list).to(device)

def prune_batch_mlm_target(target, masked_tokens):
    sizes = masked_tokens.sum(-1)
    device = masked_tokens.device
    idxs = [torch.LongTensor([0]).to(device)] + [torch.sum(sizes[:i+1]).long() for i in range(len(sizes)-1)]
    target_list = []
    for idx, size in zip(idxs, sizes):
        if size <= 1:
            continue
        b_target = target[idx:idx+size]
        target_list.append(b_target)
    pruned_target = torch.cat(target_list).to(device)
    return pruned_target