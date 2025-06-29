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


def matrix2ct(contact, seq, seq_id, ct_dir, threshold=0.5):
    """
    生成CT格式文件，处理多重配对冲突
    :param contact: 包含配对强度的对称矩阵 (不是二值化的)
    :param seq: RNA序列字符串
    :param seq_id: 序列ID
    :param ct_dir: CT文件输出目录
    :param threshold: 配对阈值
    """
    seq_len = len(seq)
    
    # 初始化配对字典
    pair_dict = {}
    for i in range(seq_len):
        pair_dict[i] = -1
    
    # 找到所有可能的配对（强度 > threshold）
    potential_pairs = []
    for i in range(seq_len):
        for j in range(i+1, seq_len):
            if contact[i, j] > threshold:
                potential_pairs.append((contact[i, j], i, j))
    
    # 按配对强度降序排序
    potential_pairs.sort(reverse=True)
    
    # 分配配对，确保每个核苷酸最多只有一个配对
    conflicts = []
    for strength, i, j in potential_pairs:
        if pair_dict[i] == -1 and pair_dict[j] == -1:
            # 无冲突，直接分配
            pair_dict[i] = j
            pair_dict[j] = i
        else:
            # 记录冲突情况用于调试
            conflicts.append((strength, i, j, 
                            f"i={i} already paired with {pair_dict[i]}" if pair_dict[i] != -1 else "",
                            f"j={j} already paired with {pair_dict[j]}" if pair_dict[j] != -1 else ""))
    
    # 统计信息
    total_potential = len(potential_pairs)
    actual_pairs = sum(1 for i in pair_dict.values() if i != -1) // 2
    
    if conflicts:
        logging.info(f"{seq_id}: {total_potential} potential pairs, {actual_pairs} assigned, {len(conflicts)} conflicts resolved")
    
    # 确保输出目录存在
    if not os.path.exists(ct_dir):
        os.makedirs(ct_dir)
    ct_file = os.path.join(ct_dir, f"{seq_id}.ct")
    
    # 写入CT文件
    with open(ct_file, "w") as f:
        f.write(f"{seq_len}\t{seq_id}\n")
        for i in range(seq_len):
            prev_idx = i if i > 0 else 0
            next_idx = i+2 if i < seq_len-1 else 0
            pair_idx = pair_dict[i]+1 if pair_dict[i] >= 0 else 0
            f.write(f"{i+1}\t{seq[i]}\t{prev_idx}\t{next_idx}\t{pair_idx}\t{i+1}\n")
    
    return ct_file

def post_process_prediction(pair_attn, data_seq, seq, seq_id, ct_dir):
    '''
    input:
    rna_lst: List of rna str
    best_model_path: Best secondary structure prediction model path
    mlm_pretrained_model_path: The path of the pre-trained model
    arg_overrides: The folder where the character-to-number mapping file resides
    device: The driver used by the model
    output_dir: CT file output dir
    
    output:
    ct_files_lst: List of CT file paths
    '''
    
    pair_attn = pair_attn.unsqueeze(0)
    post_pair_attn = post_process(pair_attn, data_seq, 0.01, 0.1, 100, 1.6, True,1.5)
    # map_no_train = (post_pair_attn > 0.5).float().squeeze().cpu().numpy()
    # 提取矩阵并转换为numpy数组
    contact_matrix = post_pair_attn.squeeze().cpu().numpy()
    # pretrain_ss = rnastructure_from_matrix_to_dotbracket(map_no_train)
    ct_file = matrix2ct(contact_matrix, seq, seq_id, ct_dir)
    return ct_file

def predict(rna_lst, seq_names, best_model_path=None, mlm_pretrained_model_path=None, arg_overrides=None, device='cpu', save_path='./results/'):
    '''
    input:
    rna_lst: List of rna str
    seq_names: List of sequence names from FASTA
    best_model_path: Best secondary structure prediction model path
    mlm_pretrained_model_path: The path of the pre-trained model
    arg_overrides: The folder where the character-to-number mapping file resides
    device: The driver used by the model
    save_path: Directory to save CT files
    
    output:
    ct_files_lst: List of generated CT file paths
    '''
    # 确保 CT 文件输出目录存在
    ct_dir = save_path
    if not os.path.exists(ct_dir):
        os.makedirs(ct_dir)

    # load model
    model_pre = load_pretrained_ernierna(mlm_pretrained_model_path, arg_overrides)
    my_model = ChooseModel(model_pre.encoder)
    state_dict = torch.load(best_model_path, map_location=f'cuda:{device}' if device != 'cpu' else 'cpu')
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    my_model.load_state_dict(new_state_dict)
    my_model = my_model.to(device)
    my_model.eval()
    
    pretrain_model = ErnieRNA(model_pre.encoder).to(device)
    pretrain_model.eval()
    
    print('Model Loading Done!!!')

    # Predict structure one by one
    ct_files_lst = []
    with torch.no_grad():
        for i, (seq, seq_name) in enumerate(zip(rna_lst, seq_names)):
            X, data_seq = seq_to_rnaindex_and_onehot(seq)
            one_d, twod_data = prepare_input_for_ernierna(X, len(seq))
            
            oned = one_d.to(device)
            twod_data = twod_data.to(device)
            data_seq = data_seq.to(device)
            
            # 微调模型预测
            pred_ss = my_model(oned, twod_data)
            pair_attn = pred_ss
            finetune_ct_file = post_process_prediction(pair_attn, data_seq, seq, f"{seq_name}_finetune_prediction", ct_dir)
            ct_files_lst.append(finetune_ct_file)
            del pair_attn, pred_ss
            
            # 零样本预测
            pred_ss = pretrain_model(oned, twod_data)
            test_attn = pred_ss[-1][0,5][1:-1,1:-1]
            pair_attn = (test_attn + test_attn.T) / 2
            zeroshot_ct_file = post_process_prediction(pair_attn, data_seq, seq, f"{seq_name}_zeroshot_prediction", ct_dir)
            ct_files_lst.append(zeroshot_ct_file)
            del pair_attn, pred_ss
            
            print(f"Processed sequence {i+1}/{len(rna_lst)}: {seq_name}")
        
    return ct_files_lst
        
        
if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--seqs_path", default='./data/ss_prediction/rna3db_testseqs.fasta', type=str, help="The path of input seqs")
    parser.add_argument("--save_path", default='./results/ernie_rna_ss_prediction/rna3db_test_results/', type=str, help="The path of rna ss extracted by ERNIE-RNA")
    parser.add_argument("--arg_overrides", default={ "data": './src/dict/' }, help="The path of vocabulary")
    parser.add_argument("--ernie_rna_pretrained_checkpoint", default='./checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt', type=str, help="The path of pre-trained ERNIE-RNA checkpoint")
    parser.add_argument("--dataset_name", default=None, type=str, help="Dataset name (bpRNA-1m, RNAStralign, RIVAS, RNA3DB, bpRNA-new, bpRNA-1m_RNAstralign). If specified, ss_rna_checkpoint will be auto-set.")
    parser.add_argument("--ss_rna_checkpoint", default=None, type=str, help="The path of fine-tuned ERNEI-RNA checkpoint")
    parser.add_argument("--device", default=0, help="device integer for GPU ID, or 'cpu'")


    args = parser.parse_args()

    # 处理 dataset_name 和 ss_rna_checkpoint 参数
    checkpoint_base_path = './checkpoint/ERNIE-RNA_ss_prediction_checkpoint/'
    if args.dataset_name is not None:
        # 根据 dataset_name 设置对应的 checkpoint 路径
        checkpoint_mapping = {
            'bpRNA-1m': 'ERNIE-RNA_attn-map_ss_prediction_bpRNA-1m_checkpoint.pt',
            'RNAStralign': 'ERNIE-RNA_attn-map_ss_prediction_RNAStralign_checkpoint.pt',
            'RIVAS': 'ERNIE-RNA_attn-map_ss_prediction_RIVAS_checkpoint.pt',
            'RNA3DB': 'ERNIE-RNA_attn-map_ss_prediction_RNA3DB_checkpoint.pt',
            'bpRNA-new': 'ERNIE-RNA_attn-map_frozen_ss_prediction_bpRNA-1m_checkpoint.pt',
            'bpRNA-1m_RNAstralign': 'ERNIE-RNA_attn-map_ss_prediction_bpRNA-1m-all_and_RNAStralign_checkpoint.pt'
        }
        
        if args.dataset_name in checkpoint_mapping:
            args.ss_rna_checkpoint = checkpoint_base_path + checkpoint_mapping[args.dataset_name]
        else:
            raise ValueError(f"Unknown dataset_name: {args.dataset_name}. Available options are: {', '.join(checkpoint_mapping.keys())}")
    elif args.ss_rna_checkpoint is None:
        # 如果 dataset_name 为 None 且 ss_rna_checkpoint 也为 None，则报错
        raise ValueError("Either --dataset_name or --ss_rna_checkpoint must be specified")

    seqs_dict = read_fasta_file(args.seqs_path)
    seqs_lst = list(seqs_dict.values())
    seq_names = list(seqs_dict.keys())
    
    # 预测RNA二级结构并生成CT文件
    ct_files = predict(seqs_lst, seq_names, args.ss_rna_checkpoint, args.ernie_rna_pretrained_checkpoint, args.arg_overrides, args.device, args.save_path)
    
    print(f"Generated {len(ct_files)} CT files in {args.save_path}")
    print(f'Done in {time.time()-start}s!')