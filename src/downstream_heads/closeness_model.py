import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# Helper functions for weight initialization (copied from 3d-closeness/model.py)
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if isinstance(m.bias, nn.Parameter):
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BasicConv') != -1:
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

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, bias = False)
        # In the original 3d-closeness/model.py, dropout was 0.4.
        # Making it configurable or keeping it fixed. For now, fixed.
        self.dropout = nn.Dropout(p=0.4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
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

class ClosenessModelAttnMapDense16(nn.Module):
    def __init__(self, sentence_encoder, mid_conv = 8, drop_out = 0.3):
        super().__init__()
        self.sentence_encoder = sentence_encoder # This is the ERNIE-RNA core encoder (RNAMaskedLMEncoder instance)
        self.conv1 = nn.Conv2d(12, mid_conv, 7, 1, 3) # 12 attention heads from ERNIE-RNA
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=drop_out)
        self.conv2 = nn.Conv2d(mid_conv, 52, 7, 1, 3) # 52 = 64 - 12
        
        self.depth = 16
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
        
    def forward(self, src_tokens, twod_tokens):
        # self.sentence_encoder is an instance of RNAMaskedLMEncoder
        # It returns: x, last_attn_bias_list, extra_dict
        # where last_attn_bias_list contains the attention maps from each layer.
        # The first element is the input twod_tokens (after projection),
        # subsequent elements are the outputs of each Transformer layer's attention.
        
        # We need to call the encoder to get these outputs.
        # For inference of attention maps, we don't need the MLM head's output (x from RNAMaskedLMEncoder).
        # We set extra_only=True if the encoder's forward supports it to potentially save computation.
        # However, the RNAMaskedLMEncoder's `extra_only` controls its own `x` output, not the underlying
        # TransformerSentenceEncoder's behavior regarding attention maps.
        # The `last_attn_bias` returned by `RNAMaskedLMEncoder` is the list of attention maps.
        
        _mlm_output, attention_maps_list_from_encoder, _extra_dict = self.sentence_encoder(
            src_tokens,
            twod_tokens=twod_tokens,
            is_twod=True,
            masked_tokens=None, # Not masking during inference for this head
            # extra_only=True, # This would make _mlm_output None, which is fine.
            # masked_only=False # Ensure full pass if extra_only is not fully effective for attn maps
        )

        if not isinstance(attention_maps_list_from_encoder, list) or len(attention_maps_list_from_encoder) <= 1:
             # The list should contain initial projected 2D bias + 12 layer outputs
             raise ValueError("Expected 'attention_maps_list_from_encoder' to be a list with more than one element.")
        
        # The last element of this list is the attention map from the final transformer layer.
        # Shape: [batch, num_heads, seq_len_with_cls_eos, seq_len_with_cls_eos]
        final_attn = attention_maps_list_from_encoder[-1] 
        
        # Remove CLS and EOS tokens from attention map
        # Input src_tokens already have CLS and EOS, so seq_len_with_cls_eos is correct.
        final_attn = final_attn[:, :, 1:-1, 1:-1] # Shape: [batch, num_heads, seq_len, seq_len]
        
        out = self.conv1(final_attn)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = torch.cat((out, final_attn), dim=1)
        
        out = self.proj(out)
        
        # output = (out + out.permute(0, 1, 3, 2)) / 2.0
        output = (out + out.permute(0, 1, 3, 2))
        return output