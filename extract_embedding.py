import os
import time
import torch
import argparse
import numpy as np

from src.ernie_rna.tasks.ernie_rna import *
from src.ernie_rna.models.ernie_rna import *
from src.ernie_rna.criterions.ernie_rna import *
from src.utils import ErnieRNAOnestage, read_text_file, load_pretrained_ernierna, prepare_input_for_ernierna


def seq_to_index(sequences):
    '''
    input:
    sequences: list of string (difference length)
    
    return:
    rna_index: numpy matrix, shape like: [len(sequences), max_seq_len+2]
    rna_len_lst: list of length
    '''

    rna_len_lst = [len(ss) for ss in sequences]
    max_len = max(rna_len_lst)
    assert max_len <= 1022
    seq_nums = len(rna_len_lst)
    rna_index = np.ones((seq_nums, max_len+2))
    for i in range(seq_nums):
        for j in range(rna_len_lst[i]):
            if sequences[i][j] in set("Aa"):
                rna_index[i][j+1] = 5
            elif sequences[i][j] in set("Cc"):
                rna_index[i][j+1] = 7
            elif sequences[i][j] in set("Gg"):
                rna_index[i][j+1] = 4
            elif sequences[i][j] in set('TUtu'):
                rna_index[i][j+1] = 6
            else:
                rna_index[i][j+1] = 3
        rna_index[i][rna_len_lst[i]+1] = 2 # add 'eos' token
    rna_index[:,0] = 0 # add 'cls' token
    return rna_index, rna_len_lst


def extract_embedding_of_ernierna(sequences, model, if_cls=True, device='cpu', layer_idx=12):
    '''
    input:
    sequences: List of string (difference length)
    model: A loaded ERNIE-RNA model
    if_cls: Bool, Determine the size of the extracted feature
    device: The driver used by the model
    
    return:
    embedding: numpy matrix, shape like: [len(sequences), 768](if_cls=True) 
               or [len(sequences), max_len_seq+2, 768](if_cls=False)
    '''
    
    # The model is already loaded outside, just use it here
    model.eval()
    
    # Converts string to rna_index
    rna_index, rna_len_lst = seq_to_index(sequences)
    layer_num = 1 if layer_idx < 12 else layer_idx
    
    if if_cls:
        embedding = np.zeros((len(sequences), layer_num, 768))
    else:
        embedding = np.zeros((len(sequences), layer_num, max(rna_len_lst)+2, 768))

    with torch.no_grad():
        for i,(index,seq_len) in enumerate(zip(rna_index,rna_len_lst)):
            one_d, two_d = prepare_input_for_ernierna(index, seq_len)
            one_d = one_d.to(device)
            two_d = two_d.to(device)
            
            output = model(one_d, two_d, layer_idx=layer_idx).cpu().detach().numpy()
            if if_cls:
                embedding[i,:,:] = output[:,0,0,:]
            else:
                embedding[i,:,:seq_len+2,:] = output[:,0,:,:]
        
    return embedding


def extract_attnmap_of_ernierna(sequences, model, attn_len=None, device='cpu', layer_idx=13, head_idx=12):
    '''
    input:
    sequences: List of string (difference length)
    model: A loaded ERNIE-RNA model
    attn_len: Int (Complement the sequence to this length). if attn_len=None, atten_len will be the length of the longest sequence in the sequences
    device: The driver used by the model
    
    return:
    atten_map: numpy matrix, shape like: [len(sequences), attn_num, attn_len+2, attn_len+2]
    '''
    
    model.eval()
    
    rna_index, rna_len_lst = seq_to_index(sequences)
    if attn_len == None:
        attn_len = max(rna_len_lst)
        
    # Determine attn_num
    if head_idx == 12 and layer_idx == 13:
        attn_num = 156
    elif head_idx == 12 or layer_idx == 13:
        attn_num = head_idx if head_idx == 12 else layer_idx
    else:
        attn_num = 1
    
    rna_attn_map_embedding = np.zeros((len(sequences), attn_num, (attn_len+2), (attn_len+2)))
    
    with torch.no_grad():
        for i,(index,seq_len) in enumerate(zip(rna_index,rna_len_lst)):
            one_d, two_d = prepare_input_for_ernierna(index, seq_len)
            one_d = one_d.to(device)
            two_d = two_d.to(device)
            
            output = model(one_d, two_d, return_attn_map=True, i=layer_idx, j=head_idx).cpu().detach().numpy()
            rna_attn_map_embedding[i, :, :(seq_len+2), :(seq_len+2)] = output
        
    return rna_attn_map_embedding


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--seqs_path", default='./data/test_seqs.txt', type=str, help="The path of input seqs")
    parser.add_argument("--save_path", default='./results/ernie_rna_representations/test_seqs/', type=str, help="The path of output extracted by ERNIE-RNA")
    parser.add_argument("--arg_overrides", default={ "data": './src/dict/' }, help="The path of vocabulary")
    parser.add_argument("--ernie_rna_pretrained_checkpoint", default='./checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt', type=str, help="The path of ERNIE-RNA checkpoint")
    parser.add_argument("--layer_idx_emb", default=12, type=int, help="The layer idx of which we extract embedding from, 12 for all layers")
    parser.add_argument("--layer_idx_attn", default=13, type=int, help="The layer idx of which we extract attnmap from, 13 for all layers")
    parser.add_argument("--head_idx_attn", default=12, type=int, help="The head idx of which we extract attnmap from, 12 for all heads")
    parser.add_argument("--device", default=0, type=int, help="device")

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    lines = read_text_file(args.seqs_path)
    seqs_lst = [line for line in lines]
    
    try:
        assert 0 <= args.layer_idx_emb <= 12
        assert 0 <= args.layer_idx_attn <= 13
        assert 0 <= args.head_idx_attn <= 12
    except:
        raise(NotImplementedError)
    
    # Load the model once here
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    model_pretrained = load_pretrained_ernierna(args.ernie_rna_pretrained_checkpoint, args.arg_overrides)
    model = ErnieRNAOnestage(model_pretrained.encoder).to(device)
    print('Model Loading Done!!!')

    cls_embedding = extract_embedding_of_ernierna(seqs_lst, model, if_cls=True, device=device, layer_idx=args.layer_idx_emb)
    np.save(os.path.join(args.save_path, 'cls_embedding.npy'), cls_embedding)
    
    all_embedding = extract_embedding_of_ernierna(seqs_lst, model, if_cls=False, device=device, layer_idx=args.layer_idx_emb)
    np.save(os.path.join(args.save_path, 'all_embedding.npy'), all_embedding)
    
    attnmap = extract_attnmap_of_ernierna(seqs_lst, model, attn_len=None, device=device, layer_idx=args.layer_idx_attn, head_idx=args.head_idx_attn)
    np.save(os.path.join(args.save_path, 'attnmap.npy'), attnmap)

    print(f'Done in {time.time() - start}s!')
