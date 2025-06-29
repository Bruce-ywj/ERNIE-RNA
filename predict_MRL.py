import os
import sys
import math
import torch
import random
import joblib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch.utils import data
import scipy.stats as stats
import torch.nn.functional as F
from sklearn import preprocessing
from fairseq import checkpoint_utils
from sklearn.metrics import r2_score
from src.ernie_rna.tasks.ernie_rna import *
from src.ernie_rna.models.ernie_rna import *
from src.ernie_rna.criterions.ernie_rna import *
from src.utils import read_fasta_file, prepare_input_for_ernierna


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
            elif sequences[i][j] in set('-'):
                rna_index[i][j+1] = 1
            elif sequences[i][j] in set('N'):
                rna_index[i][j+1] = 8
            elif sequences[i][j] in set('X'):
                rna_index[i][j+1] = 19
            else:
                rna_index[i][j+1] = 3
        rna_index[i][rna_len_lst[i]+1] = 2
    rna_index[:,0] = 0
    return rna_index, rna_len_lst

def utr_index_encode(df, dictionary, seq_len_lst, col='utr100', seq_len=100):
    # Dictionary returning task related index encoding of nucleotides. 
    # Creat empty matrix.
    vectors=np.empty([len(df),seq_len+2])
    
    # Add 'bos' token and 'eos' token
    # Same with train 
    vectors[:,0] = dictionary.bos()
    vectors[:,-1] = dictionary.pad()
    
    # Iterate through UTRs and task related index encode
    for i,seq in enumerate(df[col].str[:seq_len]): 
        
        
        seq_index = []
        for j in range(len(seq)):
            if seq[j] == 'N':
                seq_index.append(dictionary.pad())
            elif seq[j] == 'T':
                seq_index.append(6)
            else:
                seq_index.append(dictionary.index(seq[j])) 
        vectors[i,1:-1] = np.array(seq_index)
        vectors[i,seq_len_lst[i]+1] = dictionary.eos()
    return vectors


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if isinstance(m.bias, nn.Parameter):
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BasicConv') != -1:   # for googlenet
        pass
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if isinstance(m.bias, nn.Parameter):
            nn.init.constant_(m.bias, 0.0)

class MRL_Random_Dataset_index_twod(data.Dataset):
    def __init__(self, data_path, dictionary, split = 'train'):
        self.seq, self.mrl, self.data_trans = self.load_data(data_path, split, dictionary)
        
    def __len__(self):
        return self.seq.shape[0]
    
    def __getitem__(self, index):
        seq_to_return = self.seq[index]
        mrl_to_return = self.mrl[index]
        twod_data = np.zeros((1,102,102))
        twod_data[0,:,:] = creatmat(seq_to_return.numpy().astype(int),base_range=1,lamda=0.8)
        
        return seq_to_return.long(), mrl_to_return, twod_data.transpose(1,2,0)
    
    def load_data(self, data_path, split, dictionary):
        df1 = pd.read_csv(data_path)
        df = df1[df1['set']=='random']
        df=df[df['total_reads']>=10]
        df['utr100'] = df['utr'] + 75*'N'
        df['utr100'] = df['utr100'].str[:100]
        df.sort_values('total_reads', inplace=True, ascending=False)
        df.reset_index(inplace=True, drop=True)
        
        e_test = pd.DataFrame(columns=df.columns)
        for i in range(25,101):
            tmp = df[df['len']==i]
            tmp.sort_values('total_reads', inplace=True, ascending=False)
            tmp.reset_index(inplace=True, drop=True)
            e_test = e_test.append(tmp.iloc[:100])
    
        e_test.reset_index(inplace=True, drop=True)    
        len_test = np.array(e_test['len'])
        seq_test = utr_index_encode(e_test, dictionary, len_test, seq_len=100)

        e_train_valid = pd.concat([df, e_test, e_test]).drop_duplicates(keep=False)
        # fix seed
        random.seed(1)
        valid_index = random.sample(range(0, 76319),7631)
        # for valid dataset
        e_valid = e_train_valid.iloc[valid_index]
        e_valid.reset_index(inplace=True, drop=True)
        len_valid = np.array(e_valid['len'])
        seq_valid = utr_index_encode(e_valid, dictionary, len_valid, seq_len=100)

        # for train dataset
        e_train = pd.concat([e_train_valid, e_valid, e_valid]).drop_duplicates(keep=False)
        e_train.reset_index(inplace=True, drop=True)
        len_train = np.array(e_train['len'])
        seq_train = utr_index_encode(e_train, dictionary, len_train, seq_len=100)
        data_trans_train = preprocessing.StandardScaler()   
        data_trans_train.fit(e_train.loc[:,'rl'].values.reshape(-1,1))
        mrl_train = data_trans_train.transform(e_train.loc[:,'rl'].values.reshape(-1,1))
        
        data_trans_test = data_trans_train
        mrl_test = data_trans_test.transform(e_test.loc[:,'rl'].values.reshape(-1,1))
        
        data_trans_valid = data_trans_train
        mrl_valid = data_trans_valid.transform(e_valid.loc[:,'rl'].values.reshape(-1,1))
        
        if split == "train":
            return torch.from_numpy(seq_train), torch.from_numpy(mrl_train), data_trans_train
        elif split == "valid":
            return torch.from_numpy(seq_valid), torch.from_numpy(mrl_valid), data_trans_valid
        else:
            return torch.from_numpy(seq_test), torch.from_numpy(mrl_test), data_trans_test
    
    def get_trans(self):
        return self.data_trans

class UtrResBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        dilation=1,
        conv_layer=nn.Conv2d,
        norm_layer=nn.BatchNorm2d,
    ):
        super(UtrResBlock, self).__init__()        
        self.bn1 = norm_layer(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv_layer(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, bias=False)       
        self.bn2 = norm_layer(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv_layer(out_planes, out_planes, kernel_size=3, padding=dilation, bias=False)

        if stride > 1 or out_planes != in_planes: 
            self.downsample = nn.Sequential(
                conv_layer(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_planes),
            )
        else:
            self.downsample = None
            
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

class utr_regression(nn.Module):
    def __init__(self,in_channels,main_planes, dropout,embedding_dim):
        super().__init__()
        # Conv1 input:[N,Cin,Lin] output:[N,Cout,Lout]
        self.reductio_module = nn.Linear(embedding_dim, in_channels)
        self.in_channels = in_channels
        self.predictor = self.create_1dcnn_for_emd(in_planes=self.in_channels, main_planes = main_planes, dropout = dropout, out_planes=1)

        self.reductio_module.apply(weights_init_kaiming)
        self.reductio_module.apply(weights_init_classifier)
        self.predictor.apply(weights_init_kaiming)
        self.predictor.apply(weights_init_classifier)

    def forward(self,input, x):

        # x:[B,T,C] [B,100,768]
        pad_lst = torch.eq(input,1)
        eos_lst = torch.eq(input,2)
        # set padding matrix to 0
        x[pad_lst,:] = 0
        # set eos matrix to 0
        x[eos_lst,:] = 0
        x = self.reductio_module(x).transpose(1,2) # (B,32,100)
        # conv1
        x = self.predictor(x)
        return x
    
    def create_1dcnn_for_emd(self, in_planes, main_planes, dropout, out_planes):
        # main_planes = in_planes * 2
        dropout = dropout
        emb_cnn = nn.Sequential(
            nn.Conv1d(in_planes, main_planes, kernel_size=3, padding=1), 
            UtrResBlock(main_planes * 1, main_planes * 1, stride=2, dilation=1, conv_layer=nn.Conv1d,
                     norm_layer=nn.BatchNorm1d), 
            UtrResBlock(main_planes * 1, main_planes * 1, stride=1, dilation=1, conv_layer=nn.Conv1d,
                     norm_layer=nn.BatchNorm1d),  
            UtrResBlock(main_planes * 1, main_planes * 1, stride=2, dilation=1, conv_layer=nn.Conv1d,
                     norm_layer=nn.BatchNorm1d), 
            UtrResBlock(main_planes * 1, main_planes * 1, stride=1, dilation=1, conv_layer=nn.Conv1d,
                     norm_layer=nn.BatchNorm1d),  
            UtrResBlock(main_planes * 1, main_planes * 1, stride=2, dilation=1, conv_layer=nn.Conv1d,
                     norm_layer=nn.BatchNorm1d), 
            UtrResBlock(main_planes * 1, main_planes * 1, stride=1, dilation=1, conv_layer=nn.Conv1d,
                     norm_layer=nn.BatchNorm1d),       
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(main_planes * 1, out_planes),
        )
        return emb_cnn


class choose_model(nn.Module):
    def __init__(self, sentence_encoder, in_channels,main_planes, dropout, hidden_embedding):
        super().__init__() 
        self.sentence_encoder = sentence_encoder
        self.head = utr_regression(in_channels,main_planes, dropout, hidden_embedding)
    def forward(self,x, twod_input):
        input = x[:,1:-1]
        
        _,_,out_dict = self.sentence_encoder(x,twod_tokens=twod_input,is_twod=True,extra_only=True, masked_only=False)
        x = out_dict['inner_states'][-1][1:-1,:,:].transpose(0, 1) # (T, B, C) -> (B, T, C)
        
        output = self.head(input,x)
        return output

def Gaussian(x):
    return math.exp(-0.5*(x*x))

def paired(x,y,lamda=0.8):
    if x == 5 and y == 6:
        return 2
    elif x == 4 and y == 7:
        return 3
    elif x == 4 and y == 6:
        return lamda
    elif x == 6 and y == 5:
        return 2
    elif x == 7 and y == 4:
        return 3
    elif x == 6 and y == 4:
        return lamda
    else:
        return 0
    
base_range_lst = [1]
lamda_lst = [0.8]

def creatmat(data, base_range=30, lamda=0.8):
    paird_map = np.array([[paired(i,j,lamda) for i in range(30)] for j in range(30)])
    data_index = np.arange(0,len(data))

    coefficient = np.zeros([len(data),len(data)])

    score_mask = np.full((len(data),len(data)),True)
    for add in range(base_range):
        data_index_x = data_index - add
        data_index_y = data_index + add
        score_mask = ((data_index_x >= 0)[:,None] & (data_index_y < len(data))[None,:]) & score_mask
        data_index_x,data_index_y = np.meshgrid(data_index_x.clip(0,len(data) - 1),data_index_y.clip(0,len(data) - 1),indexing='ij')
        score = paird_map[data[data_index_x],data[data_index_y]]
        score_mask = score_mask & (score != 0)
        
        coefficient = coefficient + score * score_mask * Gaussian(add)
        if ~(score_mask.any()) :
            break
    score_mask = coefficient > 0
    for add in range(1,base_range):
        data_index_x = data_index + add
        data_index_y = data_index - add
        score_mask = ((data_index_x < len(data))[:,None] & (data_index_y >= 0)[None,:]) & score_mask
        data_index_x,data_index_y = np.meshgrid(data_index_x.clip(0,len(data) - 1),data_index_y.clip(0,len(data) - 1),indexing='ij')
        score = paird_map[data[data_index_x],data[data_index_y]]
        score_mask = score_mask & (score != 0)
        coefficient = coefficient + score * score_mask * Gaussian(add)
        if ~(score_mask.any()) :
            break
    return coefficient

def save_predictions(predictions, output_dir):
    """
    save output results
    
    Args:
        predictions: prediction list
        output_dir: output directory
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "prediction.txt")
    
    # 将预测结果写入文件
    with open(output_file, 'w') as f:
        for pred_dict in predictions:
            for seq, value in pred_dict.items():
                f.write(f"{seq}\t{value}\n")
    
    print(f"Predictions saved to {output_file}")

def main(args):
    mlm_pretrained_model_path = args.bert_path
    arg_overrides = args.arg_overrides
    rna_models, pre_args, pre_task = checkpoint_utils.load_model_ensemble_and_task(mlm_pretrained_model_path.split(os.pathsep),
                                                                    arg_overrides=arg_overrides)
    model_pre = rna_models[0]

    gpu_id = args.device
    available_device_ids = [gpu_id]

    if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: {gpu_id}")
    else:
        device = torch.device('cpu')
        print(f"GPU {gpu_id} not available, using CPU instead")

    my_model = choose_model(model_pre.encoder, 32, 64, 0.5, 768)
    my_model = torch.nn.DataParallel(my_model,available_device_ids)
    my_model.load_state_dict(torch.load(args.model_root))

    my_model = my_model.to(available_device_ids[0])
    my_model.eval()

    seqs_dict = read_fasta_file(args.data_roots)
    seqs_lst = list(seqs_dict.values())

    scaler = joblib.load(args.scaler_root)

    pred_mrl_lst = []
    for seq in seqs_lst:
        rna_index, rna_len_lst = seq_to_rna_index([seq])
        one_d, two_d = prepare_input_for_ernierna(rna_index[0],rna_len_lst[0])
        one_d = one_d.to(available_device_ids[0])
        two_d = two_d.to(available_device_ids[0])
        with torch.no_grad():
        # fine_tune result
            pred_utr_ori = my_model(one_d,two_d)
        
        pred_utr_post = scaler.inverse_transform(pred_utr_ori.cpu().detach().numpy())
        pred_mrl_lst.append({seq:pred_utr_post.item()})

        del one_d,two_d,pred_utr_ori,pred_utr_post
    return pred_mrl_lst

def prepare():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_roots", default="./data/MRL_data/seqs.fasta", type=str, help="RNA sequence data path")

    parser.add_argument("--bert_path", default='./checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt', type=str, help="The parameter path of ERNIE-RNA")
    parser.add_argument("--arg_overrides", default={ "data": './src/dict/' }, help="The path of vocabulary")

    parser.add_argument("--model_root", default='./checkpoint/ERNIE-RNA_UTR_MRL_checkpoint/ERNIE-RNA-UTR_ML_CNN_checkpoint.pt', type=str, help="The path where the model checkpoint is saved")
    parser.add_argument("--scaler_root", default='./checkpoint/ERNIE-RNA_UTR_MRL_checkpoint/scaler.save', type=str, help="The path where the scaler is saved")
    parser.add_argument("--device", default=0, type=int, help="GPU device ID to use (default: 0)")
    parser.add_argument("--output_dir", default="./results/ernie_rna_utr_mrl", type=str, help="Directory to save prediction results")



    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = prepare()
    pred = main(args)
    save_predictions(pred, args.output_dir)
    print(pred)

