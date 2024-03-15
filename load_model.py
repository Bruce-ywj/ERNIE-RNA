import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from pretrain_code.tasks.mlm import *
from pretrain_code.models.mlm import *
from pretrain_code.tasks.twod_mlm import *
from pretrain_code.criterions.mlm import *
from pretrain_code.models.twod_mlm import *
from pretrain_code.criterions.twod_mlm import *


def oned_to_twod_bert_input(index, seq_len):
    
    shorten_index = index[:seq_len+2]
    one_d = torch.from_numpy(shorten_index).long().reshape(1,-1)
    two_d = np.zeros((1,seq_len+2,seq_len+2))
    two_d[0,:,:] = creatmat(shorten_index.astype(int),base_range=1,lamda=0.8)
    two_d = two_d.transpose(1,2,0)
    two_d = torch.from_numpy(two_d).reshape(1,seq_len+2,seq_len+2,1)
    
    return one_d, two_d

def load_model_pretrained(mlm_pretrained_model_path,arg_overrides):
    rna_models, _, _ = checkpoint_utils.load_model_ensemble_and_task(mlm_pretrained_model_path.split(os.pathsep),arg_overrides=arg_overrides)
    model_pretrained = rna_models[0]
    return model_pretrained

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
    # np.indices((2,2))    
    coefficient = np.zeros([len(data),len(data)])
    # mat = np.zeros((len(data),len(data))) 
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

class choose_model(nn.Module):
    def __init__(self, sentence_encoder):
        super().__init__()
        self.sentence_encoder = sentence_encoder
        # self.head = rna_ss_resnet32()
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=7, stride=1, padding=3)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.conv1 = nn.Conv2d(1, 8, 7, 1, 3)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)
        self.conv2 = nn.Conv2d(8, 63, 7, 1, 3)
        self.depth = 8
        res_layers = []
        for i in range(self.depth):
            dilation = pow(2, (i % 3))
            res_layers.append(MyBasicResBlock(inplanes=64, planes=64, dilation=dilation))
        res_layers = nn.Sequential(*res_layers)
        final_layer = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        layers = OrderedDict()
        layers["resnet"] = res_layers
        layers["final"] = final_layer
        self.proj = nn.Sequential(layers)
        self.proj.apply(weights_init_kaiming)
        self.proj.apply(weights_init_classifier)
        
    def forward(self,x, twod_input):
        
        input = x[:,1:-1]
        _,attn_map,out_dict = self.sentence_encoder(x,twod_tokens=twod_input,is_twod=True,extra_only=True, masked_only=False)
        # final_attn = attn_map[-1][:,5:6,1:-1,1:-1]
        final_attn = attn_map[:,5:6,1:-1,1:-1]

        
        out = self.conv1(final_attn)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = torch.cat((out, final_attn), dim=1)
        # print(out.shape)
        out = self.proj(out)
        
        output = (out + out.permute(0,1,3,2))
        return output

class rna_ss_resnet32(nn.Module):
    def __init__(self, depth=32):
        super().__init__()
        self.depth = depth
        # 进入ResNet32之前：1、fc到128
        self.fc1 = nn.Linear(in_features=768, out_features=128)
        self.fc1.apply(weights_init_kaiming)
        self.fc1.apply(weights_init_classifier)
            
        # 进入ResNet32之前：2、conv到64    
        self.Conv2d_1 = nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size = 1)
        self.Conv2d_1.apply(weights_init_kaiming)
         
        # 定义ResNet32参数
        res_layers = []
        for i in range(self.depth):
            dilation = pow(2, (i % 3))
            res_layers.append(MyBasicResBlock(inplanes=64, planes=64, dilation=dilation))
        res_layers = nn.Sequential(*res_layers)

        # final_layer = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        final_layer = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        layers = OrderedDict()
        layers["resnet"] = res_layers
        layers["final"] = final_layer
        self.proj = nn.Sequential(layers)
        self.proj.apply(weights_init_kaiming)
        self.proj.apply(weights_init_classifier)
        
        
    def forward(self,x):
        x = self.fc1(x) # -> [B,T,128]
        # x = x.squeeze()
        batch_size, seqlen, hiddendim = x.size()
        x = x.unsqueeze(2).expand(batch_size, seqlen, seqlen, hiddendim)
        x_T = x.permute(0,2,1,3)
        x_concat = torch.cat([x,x_T],dim=3) # -> [B,T,T,C*2]
        x = x_concat.permute(0,3,1,2) # -> [B,C*2,T,T]
        x = self.Conv2d_1(x)
        # ResNet32+output的conv处理
        x = self.proj(x)
        upper_triangular_x = torch.triu(x)
        lower_triangular_x = torch.triu(x,diagonal=1).permute(0,1,3,2)
        output = upper_triangular_x + lower_triangular_x
        # return shape like [B,1,L,L]
        return output
    
class MyBasicResBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ) -> None:
        super(MyBasicResBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # cjy commented
        #if dilation > 1:
        #    raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, bias = False)
        self.dropout = nn.Dropout(p=0.3)
        self.relu2 = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
        #self.bn2 = norm_layer(planes)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out += identity
        return out

class Twod_mlm_onestage(nn.Module):
    '''
    input_x: shape like: [B,L+2], one for "cls" token, another for "eos" token
    input_twod_input: shape like: [1,L+2,L+2,1], calculated from input_x
    outpus: shape like: [B,L+2,768], 768 is the dim of embedding extracted by pre-train model
    '''
    def __init__(self, sentence_encoder):
        super().__init__() 
        self.sentence_encoder = sentence_encoder
    def forward(self,x,twod_input,return_attn_map = False,i=12,j=5):
        _,attn_map_lst,out_dict = self.sentence_encoder(x,twod_tokens=twod_input,is_twod=True,extra_only=True, masked_only=False)
        x = out_dict['inner_states'][-1].transpose(0, 1) # (T, B, C) -> (B, T, C)
        # x shape like [1, 100, 768]
        if return_attn_map:
            atten1 = F.softmax(attn_map_lst[0,j], dim=-1)
            return atten1
        return x

class Oned_mlm_onestage(nn.Module):
    '''
    input_x: shape like: [B,L+2], one for "cls" token, another for "eos" token
    outpus: shape like: [B,L+2,768], 768 is the dim of embedding extracted by pre-train model
    '''
    def __init__(self, sentence_encoder):
        super().__init__() 
        self.sentence_encoder = sentence_encoder
    def forward(self,x):
        embd,_ = self.sentence_encoder(x)
        x = embd[-1].transpose(0, 1) # (T, B, C) -> (B, T, C)
        # x shape like [1, 100, 768]
        return x

