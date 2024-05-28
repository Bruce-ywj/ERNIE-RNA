import os
import math
import time
import torch
import argparse
import numpy as np
from src.ernie_rna.tasks.ernie_rna import *
from src.ernie_rna.models.ernie_rna import *
from src.ernie_rna.criterions.ernie_rna import *
from src.utils import load_pretrained_ernierna, prepare_input_for_ernierna, ChooseModel, read_fasta_file, save_rnass_results


class ErnieRNA(nn.Module):
    def __init__(self, sentence_encoder):
        super().__init__()
        self.sentence_encoder = sentence_encoder
    def forward(self,x, twod_input):
        _,attn_map,_ = self.sentence_encoder(x,twod_tokens=twod_input,is_twod=True,extra_only=True, masked_only=False)
        return attn_map


def constraint_matrix_batch(x):
    """
    this function is referred from e2efold utility function, located at https://github.com/ml4bio/e2efold/tree/master/e2efold/common/utils.py
    """
    base_a = x[:, :, 0]
    base_u = x[:, :, 1]
    base_c = x[:, :, 2]
    base_g = x[:, :, 3]
    batch = base_a.shape[0]
    length = base_a.shape[1]
    au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
    au_ua = au + torch.transpose(au, -1, -2)
    cg = torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
    cg_gc = cg + torch.transpose(cg, -1, -2)
    ug = torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
    ug_gu = ug + torch.transpose(ug, -1, -2)
    return au_ua + cg_gc + ug_gu

def soft_sign(x):
    '''
    Implement the Softsign function
    '''
    
    k = 1
    return 1.0/(1.0+torch.exp(-2*k*x))

def contact_a(a_hat, m):
    a = a_hat * a_hat
    a = (a + torch.transpose(a, -1, -2)) / 2
    a = a * m
    return a

def post_process(u, x, lr_min, lr_max, num_itr, rho=0.0, with_l1=False,s=math.log(9.0)):
    """
    :param u: utility matrix, u is assumed to be symmetric, in batch
    :param x: RNA sequence, in batch
    :param lr_min: learning rate for minimization step
    :param lr_max: learning rate for maximization step (for lagrangian multiplier)
    :param num_itr: number of iterations
    :param rho: sparsity coefficient
    :param with_l1:
    :return:
    """
    m = constraint_matrix_batch(x).float()
    # u with threshold
    # equivalent to sigmoid(u) > 0.9
    # u = (u > math.log(9.0)).type(torch.FloatTensor) * u
    u = soft_sign(u - s) * u

    # initialization
    a_hat = (torch.sigmoid(u)) * soft_sign(u - s).detach()
    lmbd = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1).detach()

    # gradient descent
    for t in range(num_itr):

        grad_a = (lmbd * soft_sign(torch.sum(contact_a(a_hat, m), dim=-1) - 1)).unsqueeze_(-1).expand(u.shape) - u / 2
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))
        a_hat -= lr_min * grad
        lr_min = lr_min * 0.99

        if with_l1:
            a_hat = F.relu(torch.abs(a_hat) - rho * lr_min)

        lmbd_grad = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1)
        lmbd += lr_max * lmbd_grad
        lr_max = lr_max * 0.99
    a = a_hat * a_hat
    a = (a + torch.transpose(a, -1, -2)) / 2
    a = a * m
    return a

def seq_to_rnaindex_and_onehot(seq):
    '''
    input:
    sequences: seq string
    
    return:
    X: numpy matrix, shape like: [1, l+2]
    data_seq: numpy matrix, one-hot encode shape like: [1, l, 4]
    '''

    l = len(seq)
    X = np.ones((1,l+2))
    data_seq = torch.zeros((1,l,4))
    for j in range(l):
        if seq[j] in set('Aa'):
            X[0,j+1] = 5
            data_seq[0,j] = torch.Tensor([1,0,0,0])
        elif seq[j] in set('UuTt'):
            X[0,j+1] = 6
            data_seq[0,j] = torch.Tensor([0,1,0,0])
        elif seq[j] in set('Cc'):
            X[0,j+1] = 7
            data_seq[0,j] = torch.Tensor([0,0,1,0])
        elif seq[j] in set('Gg'):
            X[0,j+1] = 4
            data_seq[0,j] = torch.Tensor([0,0,0,1])
        else:
            X[0,j+1] = 3
            data_seq[0,j] = torch.Tensor([0,0,0,0]) 
    X[0,l+1] = 2
    X[0,0] = 0
    return X, data_seq

def rnastructure_from_matrix_to_dotbracket(stru):
    '''
    input:
    stru: numpy matrix, shape like: [l, l], Elements can only be 0 or 1
    
    return:
    fine_tune_ss: string, Indicates the pairing of RNA bases, like: '.(((............))).'
    '''
    
    l = stru.shape[0]
    pred_ss = ['.'] * l
    for i in range(l):
        for j in range(i+1,l):
            if stru[i][j] == 1:
                pred_ss[i] = '('
                pred_ss[j] = ')'   
    fine_tune_ss = ''.join(pred_ss)
    return fine_tune_ss

def post_process_prediction(pair_attn, data_seq):
    '''
    input:
    stru: numpy matrix, shape like: [1, 1, l, l], Element as decimal
    
    return:
    fine_tune_ss: string, Indicates the pairing of RNA bases, like: '.(((............))).'
    '''
    
    pair_attn = pair_attn.unsqueeze(0)
    post_pair_attn = post_process(pair_attn, data_seq, 0.01, 0.1, 100, 1.6, True,1.5)
    map_no_train = (post_pair_attn > 0.5).float().squeeze().cpu().numpy()
    pretrain_ss = rnastructure_from_matrix_to_dotbracket(map_no_train)
    return pretrain_ss

def predict(rna_lst, best_model_path=None, mlm_pretrained_model_path=None, arg_overrides=None, device='cpu'):
    '''
    input:
    rna_lst: List of rna str
    best_model_path: Best secondary structure prediction model path
    mlm_pretrained_model_path: The path of the pre-trained model
    arg_overrides: The folder where the character-to-number mapping file resides
    device: The driver used by the model
    
    output:
    pred_ss_lst: List of str including "." "(" ")"
    '''

    # load model
    model_pre = load_pretrained_ernierna(mlm_pretrained_model_path,arg_overrides)
    my_model = ChooseModel(model_pre.encoder)
    state_dict = torch.load(best_model_path)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    my_model.load_state_dict(new_state_dict)
    my_model = my_model.to(device)
    my_model.eval()
    
    pretrain_model = ErnieRNA(model_pre.encoder).to(device)
    pretrain_model.eval()
    
    print('Model Loading Done!!!')

    # Predict structure one by one
    pred_ss_lst = []
    with torch.no_grad():
        for seq in rna_lst:
            X, data_seq = seq_to_rnaindex_and_onehot(seq)
            one_d, twod_data = prepare_input_for_ernierna(X, len(seq))
            
            oned = one_d.to(device)
            twod_data = twod_data.to(device)
            data_seq = data_seq.to(device)
            
            pred_ss = my_model(oned,twod_data)
            pair_attn = pred_ss
            fine_tune_ss = post_process_prediction(pair_attn, data_seq)
            del pair_attn, pred_ss
            
            pred_ss = pretrain_model(oned,twod_data)
            test_attn = pred_ss[-1][0,5][1:-1,1:-1]
            pair_attn = (test_attn + test_attn.T) / 2
            pretrain_ss = post_process_prediction(pair_attn, data_seq)
            del pair_attn, pred_ss
            
            pred_ss_lst.append([fine_tune_ss,pretrain_ss])
        
        
    return pred_ss_lst
        
        
if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--seqs_path", default='./data/test_seqs.fasta', type=str, help="The path of input seqs")
    parser.add_argument("--save_path", default='./results/ernie_rna_ss_prediction/test_seqs/', type=str, help="The path of rna ss extracted by ERNIE-RNA")
    parser.add_argument("--arg_overrides", default={ "data": './src/dict/' }, help="The path of vocabulary")
    parser.add_argument("--ernie_rna_pretrained_checkpoint", default='./checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt', type=str, help="The path of pre-trained ERNIE-RNA checkpoint")
    parser.add_argument("--ss_rna_checkpoint", default='./checkpoint/ERNIE-RNA_ss_prediction_checkpoint/ERNIER-RNA_ss_prediction.pt', type=str, help="The path of fine-tuned ERNEI-RNA checkpoint")
    parser.add_argument("--device", default=0, type=int, help="device")

    

    args = parser.parse_args()

    seqs_dict = read_fasta_file(args.seqs_path)
    seqs_lst = list(seqs_dict.values())
    
    # Predicting the secondary structure of RNA
    pres_ss = predict(seqs_lst, args.ss_rna_checkpoint, args.ernie_rna_pretrained_checkpoint, args.arg_overrides, args.device)
    # print(pres_ss)  # [[fine_tune pre, pretrain pre]]
    
    save_rnass_results(args.save_path, list(seqs_dict.keys()), pres_ss)
    print(f'Done in {time.time()-start}s!')