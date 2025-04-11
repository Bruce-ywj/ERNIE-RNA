#!/usr/bin/env python3
import os
import math
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# Import your existing utilities from ERNIE-RNA
# (Adjust these imports as needed based on your project layout)
# ------------------------------------------------------------------
from src.ernie_rna.tasks.ernie_rna import *
from src.ernie_rna.models.ernie_rna import *
from src.ernie_rna.criterions.ernie_rna import *
from src.utils import (
    load_pretrained_ernierna, 
    prepare_input_for_ernierna,
    ChooseModel, 
    read_fasta_file, 
    save_rnass_results
)

# ------------------------------------------------------------------
# Classes and functions used in secondary structure prediction
# ------------------------------------------------------------------

class ErnieRNA(nn.Module):
    """
    A wrapper that extracts the final attentions from the sentence_encoder.
    """
    def __init__(self, sentence_encoder):
        super().__init__()
        self.sentence_encoder = sentence_encoder

    def forward(self, x, twod_input):
        _, attn_map, _ = self.sentence_encoder(
            x, twod_tokens=twod_input, is_twod=True,
            extra_only=True, masked_only=False
        )
        return attn_map

def constraint_matrix_batch(x):
    """
    Borrowed from e2efold’s utility function.
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
    k = 1
    return 1.0/(1.0 + torch.exp(-2 * k * x))

def contact_a(a_hat, m):
    a = a_hat * a_hat
    a = (a + torch.transpose(a, -1, -2)) / 2
    a = a * m
    return a

def post_process(u, x, lr_min, lr_max, num_itr, rho=0.0, with_l1=False, s=math.log(9.0)):
    m = constraint_matrix_batch(x).float()
    u = soft_sign(u - s) * u
    a_hat = torch.sigmoid(u) * soft_sign(u - s).detach()
    lmbd = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1).detach()
    for t in range(num_itr):
        grad_a = (lmbd * soft_sign(torch.sum(contact_a(a_hat, m), dim=-1) - 1))
        grad_a = grad_a.unsqueeze_(-1).expand(u.shape) - u / 2
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))
        a_hat -= lr_min * grad
        lr_min *= 0.99
        if with_l1:
            a_hat = F.relu(torch.abs(a_hat) - rho * lr_min)
        lmbd_grad = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1)
        lmbd += lr_max * lmbd_grad
        lr_max *= 0.99
    a = a_hat * a_hat
    a = (a + torch.transpose(a, -1, -2)) / 2
    a = a * m
    return a

def seq_to_rnaindex_and_onehot(seq):
    l = len(seq)
    X = np.ones((1, l+2))
    data_seq = torch.zeros((1, l, 4))
    for j in range(l):
        base = seq[j]
        if base in set('Aa'):
            X[0, j+1] = 5
            data_seq[0, j] = torch.tensor([1,0,0,0])
        elif base in set('UuTt'):
            X[0, j+1] = 6
            data_seq[0, j] = torch.tensor([0,1,0,0])
        elif base in set('Cc'):
            X[0, j+1] = 7
            data_seq[0, j] = torch.tensor([0,0,1,0])
        elif base in set('Gg'):
            X[0, j+1] = 4
            data_seq[0, j] = torch.tensor([0,0,0,1])
        else:
            X[0, j+1] = 3
            data_seq[0, j] = torch.tensor([0,0,0,0])
    X[0, 0] = 0   # CLS token
    X[0, l+1] = 2 # EOS token
    return X, data_seq

def rnastructure_from_matrix_to_dotbracket(stru):
    l = stru.shape[0]
    pred_ss = ['.'] * l
    for i in range(l):
        for j in range(i+1, l):
            if stru[i][j] == 1:
                pred_ss[i] = '('
                pred_ss[j] = ')'
    return ''.join(pred_ss)

def post_process_prediction(pair_attn, data_seq):
    pair_attn = pair_attn.unsqueeze(0)  # Shape: (1, 1, L, L)
    post_pair_attn = post_process(pair_attn, data_seq, 0.01, 0.1, 100, 1.6, True, 1.5)
    map_no_train = (post_pair_attn > 0.5).float().squeeze().cpu().numpy()
    pretrain_ss = rnastructure_from_matrix_to_dotbracket(map_no_train)
    return pretrain_ss

def load_ernie_rna_models(best_model_path=None, mlm_pretrained_model_path=None, arg_overrides=None, device='cpu'):
    model_pre = load_pretrained_ernierna(mlm_pretrained_model_path, arg_overrides)
    fine_tuned_model = ChooseModel(model_pre.encoder)
    state_dict = torch.load(best_model_path,
                            map_location=('cuda:%s' % device if device != 'cpu' else 'cpu'))
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    fine_tuned_model.load_state_dict(new_state_dict)
    fine_tuned_model = fine_tuned_model.to(device)
    fine_tuned_model.eval()
    pretrain_model = ErnieRNA(model_pre.encoder).to(device)
    pretrain_model.eval()
    print('Model loaded successfully')
    return fine_tuned_model, pretrain_model

@torch.no_grad()
def predict(rna_lst, fine_tuned_model, pretrain_model, device='cpu'):
    fine_tuned_model.eval()
    pretrain_model.eval()
    pred_ss_lst = []
    for seq in rna_lst:
        X, data_seq = seq_to_rnaindex_and_onehot(seq)
        one_d, twod_data = prepare_input_for_ernierna(X, len(seq))
        oned = one_d.to(device, non_blocking=True)
        twod_data = twod_data.to(device, non_blocking=True)
        data_seq = data_seq.to(device, non_blocking=True)
        if device != 'cpu':
            from torch.cuda.amp import autocast
            with autocast():
                pred_ss = fine_tuned_model(oned, twod_data)
                fine_tune_attn = pred_ss
                fine_tune_ss = post_process_prediction(fine_tune_attn, data_seq)
                attn_map = pretrain_model(oned, twod_data)
                test_attn = attn_map[-1][0, 5][1:-1, 1:-1]
                pair_attn = (test_attn + test_attn.T) / 2
                pretrain_ss = post_process_prediction(pair_attn, data_seq)
        else:
            pred_ss = fine_tuned_model(oned, twod_data)
            fine_tune_attn = pred_ss
            fine_tune_ss = post_process_prediction(fine_tune_attn, data_seq)
            attn_map = pretrain_model(oned, twod_data)
            test_attn = attn_map[-1][0, 5][1:-1, 1:-1]
            pair_attn = (test_attn + test_attn.T) / 2
            pretrain_ss = post_process_prediction(pair_attn, data_seq)
        pred_ss_lst.append([fine_tune_ss, pretrain_ss])
    return pred_ss_lst

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs_path", default='./data/test_seqs.fasta', type=str,
                        help="The path of input seqs (FASTA format).")
    parser.add_argument("--save_path", default='./results/ernie_rna_ss_prediction/test_seqs/',
                        type=str, help="Where to save predicted RNA SS text files.")
    parser.add_argument("--arg_overrides", default={ "data": './src/dict/' },
                        help="The path of vocabulary (or other overrides).")
    parser.add_argument("--ernie_rna_pretrained_checkpoint", 
                        default='./checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt', 
                        type=str, help="Path of pre-trained ERNIE-RNA checkpoint.")
    parser.add_argument("--ss_rna_checkpoint", 
                        default='./checkpoint/ERNIE-RNA_ss_prediction_checkpoint/ERNIER-RNA_ss_prediction.pt', 
                        type=str, help="Path of fine-tuned ERNIE-RNA checkpoint.")
    parser.add_argument("--device", default=0, type=int,
                        help="GPU device index (int) or 'cpu' if no GPU.")
    args = parser.parse_args()
    seqs_dict = read_fasta_file(args.seqs_path)
    seqs_lst = list(seqs_dict.values())
    seq_names_lst = list(seqs_dict.keys())
    device_choice = args.device
    fine_tuned_model, pretrain_model = load_ernie_rna_models(
        best_model_path=args.ss_rna_checkpoint,
        mlm_pretrained_model_path=args.ernie_rna_pretrained_checkpoint,
        arg_overrides=args.arg_overrides,
        device=device_choice
    )
    pres_ss = predict(
        rna_lst=seqs_lst,
        fine_tuned_model=fine_tuned_model,
        pretrain_model=pretrain_model,
        device=device_choice
    )
    save_rnass_results(args.save_path, seq_names_lst, pres_ss)
    print(f"Done in {time.time() - start:.1f}s!")

@torch.no_grad()
def predict_with_loaded_models(rna_sequences, fine_tuned_model, pretrain_model, device='cpu'):
    fine_tuned_model.eval()
    pretrain_model.eval()
    pred_ss_list = []
    for seq in rna_sequences:
        X, data_seq = seq_to_rnaindex_and_onehot(seq)
        one_d, twod_data = prepare_input_for_ernierna(X, len(seq))
        oned = one_d.to(device, non_blocking=True)
        twod_data = twod_data.to(device, non_blocking=True)
        data_seq = data_seq.to(device, non_blocking=True)
        if device != 'cpu':
            from torch.cuda.amp import autocast
            with autocast():
                pred_ss = fine_tuned_model(oned, twod_data)
                fine_tune_attn = pred_ss
                fine_tune_ss = post_process_prediction(fine_tune_attn, data_seq)
                pred_ss2 = pretrain_model(oned, twod_data)
                test_attn = pred_ss2[-1][0, 5][1:-1, 1:-1]
                pair_attn = (test_attn + test_attn.T) / 2
                pretrain_ss = post_process_prediction(pair_attn, data_seq)
        else:
            pred_ss = fine_tuned_model(oned, twod_data)
            fine_tune_attn = pred_ss
            fine_tune_ss = post_process_prediction(fine_tune_attn, data_seq)
            pred_ss2 = pretrain_model(oned, twod_data)
            test_attn = pred_ss2[-1][0, 5][1:-1, 1:-1]
            pair_attn = (test_attn + test_attn.T) / 2
            pretrain_ss = post_process_prediction(pair_attn, data_seq)
        pred_ss_list.append([fine_tune_ss, pretrain_ss])
    return pred_ss_list
