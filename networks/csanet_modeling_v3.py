import os
from os.path import join as pjoin
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndimage

from .csanet_modeling_resnet_skip import ResNetV2
from utils.simple_tools import load_config_as_namespace


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class FeatureExtractor(nn.Module):
    def __init__(self, config, img_size=256):
        super(FeatureExtractor, self).__init__()
        self.config = config
        self.img_size = img_size

        self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
    
    def forward(self, x_prev, x, x_next):
        x, skip_features = self.hybrid_model(x)
        x_prev, _ = self.hybrid_model(x_prev)
        x_next, _ = self.hybrid_model(x_next)

        return x, skip_features, x_prev, x_next


class NLBlockND_multicross_block(nn.Module):
    """
    Non-Local Block for multi-cross attention (Modified Parallel Vision).
    
    Args:
        in_channels (int): Number of input channels.
        num_heads (int): Number of heads.

    Attributes:
        in_channels (int): Number of input channels.
        num_heads (int): Number of heads.
        W_q (nn.Conv2d): Convolutional layer for the 'q' branch.
        W_k (nn.Conv2d): Convolutional layer for the 'k' branch.
        W_v (nn.Conv2d): Convolutional layer for the 'v' branch.
        W_o (nn.Sequential): Sequential block containing a convolutional layer followed by batch normalization for weight 'o'.

    Methods:
        forward(x_thisBranch, x_otherBranch): Forward pass of the non-local block.

    """
    def __init__(self, in_channels, num_heads):
        super(NLBlockND_multicross_block, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        assert self.in_channels % self.num_heads == 0, "in_channels must be divisible by num_heads"
        self.head_dim = self.in_channels // self.num_heads

        self.W_q = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1)
        self.W_k = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1)
        self.W_v = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1)

        self.W_o = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, groups=self.num_heads),
            nn.BatchNorm2d(self.in_channels)
        )

        nn.init.constant_(self.W_o[1].weight, 0)
        nn.init.constant_(self.W_o[1].bias, 0)

    def forward(self, x_thisBranch, x_otherBranch):
        """
        x_thisBranch.shape: (batch_size, hidden_dim, H, W)
        """
        batch_size, _, h, w = x_thisBranch.size()
        seq_len = h * w

        q_x = self.W_q(x_otherBranch).view(batch_size, self.in_channels, -1) # [bs, hidden_dim, seq_len=h*w]
        k_x = self.W_k(x_thisBranch).view(batch_size, self.in_channels, -1) # [bs, hidden_dim, seq_len=h*w]
        v_x = self.W_v(x_thisBranch).view(batch_size, self.in_channels, -1) # [bs, hidden_dim, seq_len=h*w]

        q_x = q_x.permute(0, 2, 1).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [bs, n_head, seq_len, h_dim]
        k_x = k_x.permute(0, 2, 1).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [bs, n_head, seq_len, h_dim]
        v_x = v_x.permute(0, 2, 1).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [bs, n_head, seq_len, h_dim]
        
        f = torch.matmul(q_x, k_x.transpose(-1, -2)) # [bs, n_head, seq_len, seq_len]
        f_div_C = F.softmax(f / math.sqrt(self.head_dim), dim=-1) # [bs, n_head, seq_len, seq_len]
        
        y = torch.matmul(f_div_C, v_x) # [bs, n_head, seq_len, h_dim]
        y = y.permute(0, 1, 3, 2).contiguous() # [bs, n_head, h_dim, seq_len]
        y = y.view(batch_size, self.in_channels, h, w) # [bs, hidden_dim, H, W]
        
        o = self.W_o(y) # [bs, hidden_dim, H, W]
        return o


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        assert self.in_channels % self.num_heads == 0, "in_channels must be divisible by num_heads"
        
        self.cross_attention_heads = NLBlockND_multicross_block(in_channels, num_heads)
    
    def forward(self, x_thisBranch, x_otherBranch):
        output = self.cross_attention_heads(x_thisBranch, x_otherBranch)
        return output


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x) 

        
class CSAModule(nn.Module):
    def __init__(self, config, in_channels, num_heads):
        super(CSAModule, self).__init__()
        self.config = config
        self.in_channels = in_channels
        self.num_heads = num_heads

        self.mhca_1 = MultiHeadCrossAttention(self.in_channels, self.num_heads)
        self.mhca_2 = MultiHeadCrossAttention(self.in_channels, self.num_heads)
        self.mhca_3 = MultiHeadCrossAttention(self.in_channels, self.num_heads)

        self.concat_conv = DoubleConv(self.in_channels * 3, self.in_channels)
    
    def forward(self, x_prev, x, x_next):
        xt1 = self.mhca_1(x, x_next)
        xt2 = self.mhca_2(x, x)
        xt3 = self.mhca_3(x, x_prev)

        xt = torch.cat([xt1, xt2, xt3], dim=1)
        x = self.concat_conv(xt)
        return x


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer.num_heads
        self.attention_head_size = int(config.transformer.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.transformer.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.transformer.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.transformer.hidden_size, self.all_head_size)

        self.out = nn.Linear(config.transformer.hidden_size, config.transformer.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer.attention_dropout)
        self.proj_dropout = nn.Dropout(config.transformer.attention_dropout)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config.transformer.hidden_size, config.transformer.mlp_dim)
        self.fc2 = nn.Linear(config.transformer.mlp_dim, config.transformer.hidden_size)
        self.act_fn = F.gelu
        self.dropout = nn.Dropout(config.transformer.dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.hidden_size = config.transformer.hidden_size
        self.attention_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Transformer(nn.Module):
    def __init__(self, config, img_size=256):
        super(Transformer, self).__init__()
        self.config = config
        self.img_size = img_size

        grid_size = config.transformer.grid_size
        patch_size = img_size // 16 // grid_size
        patch_size_real = patch_size * 16
        n_patches = (img_size // patch_size_real) ** 2

        self.patch_embedding = nn.Conv2d(in_channels=config.resnet.width_factor * 64 * 16, 
                                         out_channels=config.transformer.hidden_size, 
                                         kernel_size=patch_size, 
                                         stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.transformer.hidden_size))
        self.dropout = nn.Dropout(config.transformer.dropout)

        self.transformer_layers = nn.ModuleList()
        for _ in range(config.transformer.num_layers):
            layer = TransformerBlock(config)
            self.transformer_layers.append(copy.deepcopy(layer))
        self.encoder_norm = nn.LayerNorm(config.transformer.hidden_size, eps=1e-6)
    
    def forward(self, x):
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        x = x + self.position_embeddings
        x = self.dropout(x)

        for layer_block in self.transformer_layers:
            x = layer_block(x)
        x = self.encoder_norm(x)
        return x

        
class CSAEncoder(nn.Module):
    def __init__(self, config, img_size=256):
        super(CSAEncoder, self).__init__()
        self.config = config
        self.img_size = img_size

        self.feature_extractor = FeatureExtractor(config, img_size)
        self.csa_module = CSAModule(config, in_channels=1024, num_heads=config.csa_multiheads)
        self.transformer = Transformer(config)
    
    def forward(self, x_prev, x, x_next):
        x, skip_features, x_prev, x_next = self.feature_extractor(x_prev, x, x_next)
        x = self.csa_module(x_prev, x, x_next)
        x = self.transformer(x)
        return x, skip_features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.transformer.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class CSANet_V3(nn.Module):
    def __init__(self, config_path, img_size=256, num_classes=5):
        super(CSANet_V3, self).__init__()
        self.config = load_config_as_namespace(config_path)
        self.img_size = img_size
        self.num_classes = num_classes

        self.encoder = CSAEncoder(self.config, img_size)
        self.decoder = DecoderCup(self.config)
        self.segmentation_head = SegmentationHead(in_channels=self.config.decoder_channels[-1],  
                                                  out_channels=self.num_classes,
                                                  kernel_size=3
        )
    
    def forward(self, x_prev, x, x_next):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
            x_prev = x_prev.repeat(1,3,1,1)
            x_next = x_next.repeat(1,3,1,1)

        x, skip_features = self.encoder(x_prev, x, x_next)
        x = self.decoder(x, skip_features)
        x = self.segmentation_head(x)
        return x
    
    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights

            self.encoder.transformer.patch_embedding.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.encoder.transformer.patch_embedding.bias.copy_(np2th(weights["embedding/bias"]))

            self.encoder.transformer.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.encoder.transformer.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.encoder.transformer.position_embeddings
            if posemb.size() == posemb_new.size():
                self.encoder.transformer.position_embeddings.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.encoder.transformer.position_embeddings.copy_(posemb)
            else:
                print("load_pretrained position embedding: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                posemb_grid = posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1).reshape(1, gs_new * gs_new, -1)
                self.encoder.transformer.position_embeddings.copy_(np2th(posemb_grid))
            
            for bname, block in self.encoder.transformer.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            self.encoder.feature_extractor.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
            gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
            gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
            self.encoder.feature_extractor.hybrid_model.root.gn.weight.copy_(gn_weight)
            self.encoder.feature_extractor.hybrid_model.root.gn.bias.copy_(gn_bias)

            for bname, block in self.encoder.feature_extractor.hybrid_model.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(res_weight, n_block=bname, n_unit=uname)




