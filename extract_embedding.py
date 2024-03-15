import os
import torch
import numpy as np
from pretrain_code.tasks.mlm import *
from pretrain_code.models.mlm import *
from pretrain_code.tasks.twod_mlm import *
from pretrain_code.criterions.mlm import *
from pretrain_code.models.twod_mlm import *
from pretrain_code.criterions.twod_mlm import *
from load_model import Twod_mlm_onestage, Oned_mlm_onestage, load_model_pretrained, oned_to_twod_bert_input


def seq_to_rna_index(sequences):
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
    rna_index = np.ones((seq_nums,max_len+2))
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

def extract_embedding_oned(sequences, if_cls=True, arg_overrides = { "data": '/data/code/BERT/onestage_checkpoint_dict/' }, mlm_pretrained_model_path='/data/code/BERT/bert_oned_twostage_checkpoint/checkpoint_best.pt', device='cpu'):
    '''
    input:
    sequences: List of string (difference length)
    if_cls: Bool, Determine the size of the extracted feature
    arg_overrides: The folder where the character-to-number mapping file resides
    mlm_pretrained_model_path: The path of the pre-trained model
    device: The driver used by the model
    
    return:
    embedding: numpy matrix, shape like: [len(sequences), 768](if_cls=True) or [len(sequences), max_len_seq+2, 768](if_cls=False)
    '''
    
    # load model
    model_pretrained = load_model_pretrained(mlm_pretrained_model_path,arg_overrides)
    my_model = Oned_mlm_onestage(model_pretrained.encoder.sentence_encoder).to(device)
    print('Model Loading Done!!!')
    my_model.eval()
    
    # Converts string to rna_index
    rna_index, rna_len_lst = seq_to_rna_index(sequences)
    
    # extract embedding one by one
    if if_cls:
        embedding = np.zeros((len(sequences),768))
    else:
        embedding = np.zeros((len(sequences),max(rna_len_lst)+2,768))
    
    for i,(index,seq_len) in enumerate(zip(rna_index,rna_len_lst)):
        shorten_index = index[:seq_len+2]
        one_d = torch.from_numpy(shorten_index).long().reshape(1,-1).to(device)
        
        output = my_model(one_d).cpu().detach().numpy()
        
        if if_cls:
            embedding[i] = output[0,0,:]
        else:
            embedding[i, :seq_len+2] = output
    
    return embedding

def extract_embedding_twod(sequences, if_cls=True, arg_overrides = { "data": '/data/code/BERT/onestage_checkpoint_dict/' }, mlm_pretrained_model_path =  '/data/code/BERT/Pretrain_checkpoint/twocheckpoint_best.pt', device='cpu'):
    '''
    input:
    sequences: List of string (difference length)
    if_cls: Bool, Determine the size of the extracted feature
    arg_overrides: The folder where the character-to-number mapping file resides
    mlm_pretrained_model_path: The path of the pre-trained model
    device: The driver used by the model
    
    return:
    embedding: numpy matrix, shape like: [len(sequences), 768](if_cls=True) or [len(sequences), max_len_seq+2, 768](if_cls=False)
    '''
    
    # load model
    model_pretrained = load_model_pretrained(mlm_pretrained_model_path,arg_overrides)
    my_model = Twod_mlm_onestage(model_pretrained.encoder).to(device)
    print('Model Loading Done!!!')
    my_model.eval()
    
    # Converts string to rna_index
    rna_index, rna_len_lst = seq_to_rna_index(sequences)
    
    # extract embedding one by one
    if if_cls:
        embedding = np.zeros((len(sequences),768))
    else:
        embedding = np.zeros((len(sequences),max(rna_len_lst)+2,768))

    for i,(index,seq_len) in enumerate(zip(rna_index,rna_len_lst)):
        
        one_d, two_d = oned_to_twod_bert_input(index,seq_len)
        one_d = one_d.to(device)
        two_d = two_d.to(device)
        
        output = my_model(one_d,two_d).cpu().detach().numpy()
        
        if if_cls:
            embedding[i] = output[0,0,:]
        else:
            embedding[i, :seq_len+2] = output
        
    return embedding

def extract_embedding_twod_attnmap(sequences, attn_len=None, arg_overrides = { "data": '/data/code/BERT/onestage_checkpoint_dict/' }, mlm_pretrained_model_path =  '/data/code/BERT/Pretrain_checkpoint/twocheckpoint_best.pt', device='cpu'):
    '''
    input:
    sequences: List of string (difference length)
    attn_len: Int (Complement the sequence to this length). if attn_len=None, atten_len will be the length of the longest sequence in the sequences
    arg_overrides: The folder where the character-to-number mapping file resides
    mlm_pretrained_model_path: The path of the pre-trained model
    device: The driver used by the model
    
    return:
    atten_map: numpy matrix, shape like: [len(sequences), attn_len+2, attn_len+2]
    '''
    
    # load model
    model_pretrained = load_model_pretrained(mlm_pretrained_model_path,arg_overrides)
    my_model = Twod_mlm_onestage(model_pretrained.encoder).to(device)
    print('Model Loading Done!!!')
    my_model.eval()
    
    # Converts string to rna_index
    rna_index, rna_len_lst = seq_to_rna_index(sequences)
    
    # extract embedding one by one
    if attn_len == None:
        attn_len = max(rna_len_lst)

    rna_attn_map_embedding = np.zeros((len(sequences),(attn_len+2), (attn_len+2)))

    for i,(index,seq_len) in enumerate(zip(rna_index,rna_len_lst)):
        one_d, two_d = oned_to_twod_bert_input(index,seq_len)
        one_d = one_d.to(device)
        two_d = two_d.to(device)
        
        output = my_model(one_d,two_d,return_attn_map=True).cpu().detach().numpy()
        
        rna_attn_map_embedding[i, :(seq_len+2), :(seq_len+2)] = output
        
    return rna_attn_map_embedding

def extract_embedding_onehot(sequences,attn_len=None):
    '''
    input:
    sequences: list of string (difference length)
    attn_len: int (Complement the sequence to this length). if attn_len=None, atten_len will be the length of the longest sequence in the sequences
    
    return:
    embedding: numpy matrix, shape like: [len(sequences), 4*attn_len]
    '''
    
    rna_len_lst = [len(seq) for seq in sequences]
    max_len = max(rna_len_lst)
    assert max_len <= 1022
    seq_nums = len(rna_len_lst)
    if attn_len == None:
        attn_len = max_len
    rna_index = np.zeros((seq_nums,4*attn_len))
    for i in range(seq_nums):
        for j in range(rna_len_lst[i]):
            if sequences[i][j] in set("Aa"):
                rna_index[i][j*4:(j+1)*4] = [1,0,0,0]
            elif sequences[i][j] in set("Cc"):
                rna_index[i][j*4:(j+1)*4] = [0,1,0,0]
            elif sequences[i][j] in set("Gg"):
                rna_index[i][j*4:(j+1)*4] = [0,0,1,0]
            elif sequences[i][j] in set('TUtu'):
                rna_index[i][j*4:(j+1)*4] = [0,0,0,1]
            else:
                rna_index[i][j*4:(j+1)*4] = [0,0,0,0]
    return rna_index


if __name__ == "__main__":
    
    current_path = os.path.dirname(os.path.abspath(__file__))
    
    seqs = ['AUCUU','UUCGAGUC','AAAAUUUG']
    arg_overrides = { "data": current_path + '/pretrain_dict/' }
    mlm_pretrained_model_path =  current_path + '/model/mlm_checkpoint/checkpoint_best.pt'
    two_mlm_pretrained_model_path =  current_path + '/model/two_mlm_checkpoint/checkpoint_best.pt'
    device='cpu'
    
    cls_embedding = extract_embedding_oned(seqs, if_cls=True, arg_overrides=arg_overrides, mlm_pretrained_model_path=mlm_pretrained_model_path, device=device)
    print(cls_embedding.shape) # cls_embedding shape like [3, 768]
    
    all_embedding = extract_embedding_oned(seqs, if_cls=False, arg_overrides=arg_overrides, mlm_pretrained_model_path=mlm_pretrained_model_path, device=device)
    print(all_embedding.shape) # all_embedding shape like [3, 10, 768]
    
    cls_embedding = extract_embedding_twod(seqs, if_cls=True, arg_overrides=arg_overrides, mlm_pretrained_model_path=two_mlm_pretrained_model_path, device=device)
    print(cls_embedding.shape) # cls_embedding shape like [3, 768]
    
    all_embedding = extract_embedding_twod(seqs, if_cls=False, arg_overrides=arg_overrides, mlm_pretrained_model_path=two_mlm_pretrained_model_path, device=device)
    print(all_embedding.shape) # all_embedding shape like [3, 10, 768]
    
    twod_attnmap = extract_embedding_twod_attnmap(seqs, attn_len=None, arg_overrides=arg_overrides, mlm_pretrained_model_path=two_mlm_pretrained_model_path, device=device)
    print(twod_attnmap.shape) # twod_attnmap shape like [3, 10, 10]

    onehot_embedding = extract_embedding_onehot(seqs)
    print(onehot_embedding.shape) # onehot_embedding shape like [3, 4*8]
    
    
    
    