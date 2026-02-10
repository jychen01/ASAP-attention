__all__ = ['ASAP_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

# from collections import OrderedDict
from .ASAP_layers import *
from .RevIN import RevIN
from utils.masking import TriangularCausalMask, ProbMask, TimeCausalMask
import math


# Cell
class ASAP_backbone(nn.Module):
    def __init__(self, c_in: int, context_window: int, target_window: int, patch_len: int, stride: int, sigma: float,
                 rwr_prob: float, season: list,
                 iteration: int, _attn_dropout: bool, time_window: Optional[int] = 2, max_seq_len: Optional[int] = 1024,
                 n_layers: int = 3, d_model=128, n_heads=16, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256, norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True, fc_dropout: float = 0., head_dropout=0, padding_patch=None,
                 pretrain_head: bool = False, head_type='flatten', individual=False, revin=True, affine=True,
                 subtract_last=False,
                 verbose: bool = False, Trend=True, **kwargs):

        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # Backbone
        self.backbone = TSTiEncoder(c_in, sigma=sigma, rwr_prob=rwr_prob, season=season, iteration=iteration, _attn_dropout=_attn_dropout,
                                    patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                    n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                    attn_dropout=attn_dropout, dropout=dropout, act=act,
                                    key_padding_mask=key_padding_mask, padding_var=padding_var,
                                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                    store_attn=store_attn, pe=pe, learn_pe=learn_pe, verbose=verbose, Trend=Trend, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual
        self.time_window = time_window

        if self.pretrain_head:
            self.head = self.create_pretrain_head(self.head_nf, c_in,
                                                  fc_dropout)  # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten':
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                     head_dropout=head_dropout)

    def forward(self, z):  # z: [bs x nvars x seq_len]
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)

        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x patch_len x patch_num]

        B, N, _, L = z.shape
        time_mask = TimeCausalMask(B*N, L, z.device, width=2 * self.time_window + 1)
        # model
        z = self.backbone(z, time_mask)  # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)  # z: [bs x nvars x target_window]

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                             nn.Conv1d(head_nf, vars, 1)
                             )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, sigma, rwr_prob, season, iteration, _attn_dropout, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, Trend=True, **kwargs):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, sigma=sigma, rwr_prob=rwr_prob, season=season, iteration=iteration,
                                  _attn_dropout=_attn_dropout, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                  attn_dropout=attn_dropout, dropout=dropout, pre_norm=pre_norm, activation=act,
                                  res_attention=res_attention, n_layers=n_layers, store_attn=store_attn, Trend=Trend)

    def forward(self, x, time_mask) -> Tensor:  # x: [bs x nvars x patch_len x patch_num]

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)  # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u, time_mask=time_mask)  # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))  # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x patch_num]

        return z

    # Cell


class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, sigma, rwr_prob, season, iteration, _attn_dropout, d_k=None, d_v=None, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False, Trend=True):
        super().__init__()

        self.layers = nn.ModuleList(
            [TSTEncoderLayer(q_len, d_model, sigma=sigma, n_heads=n_heads, rwr_prob=rwr_prob, season=season, iteration=iteration,
                             _attn_dropout=_attn_dropout, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                             attn_dropout=attn_dropout, dropout=dropout, activation=activation,
                             res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn, Trend=Trend) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None, time_mask=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask,
                                                         attn_mask=attn_mask, time_mask=time_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask,
                                                 attn_mask=attn_mask, time_mask=time_mask)
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, sigma, rwr_prob, season, iteration, _attn_dropout, d_k=None, d_v=None, d_ff=256,
                 store_attn=False, norm='BatchNorm', attn_dropout=0., dropout=0., bias=True, activation="gelu",
                 res_attention=False, pre_norm=False, Trend=True):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, sigma=sigma, rwr_prob=rwr_prob, season=season, iteration=iteration,
                                       _attn_dropout=_attn_dropout, d_k=d_k, d_v=d_v, attn_dropout=attn_dropout,
                                             proj_dropout=dropout, res_attention=res_attention, Trend=Trend)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None, time_mask=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask,
                                                attn_mask=attn_mask, time_mask=time_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask,
                                        attn_mask=attn_mask, time_mask=time_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, sigma, rwr_prob, season, iteration, _attn_dropout, d_k=None, d_v=None,
                 res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False, Trend=True):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = ASAP_attention(d_model, n_heads, sigma=sigma, rwr_prob=rwr_prob, season=season, iteration=iteration,
                                       _attn_dropout=_attn_dropout, attn_dropout=attn_dropout,
                                       res_attention=self.res_attention, lsa=lsa, Trend=Trend)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None, time_mask=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,
                                                                         2)  # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3,
                                                                       1)  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)  # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask, time_mask=time_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask, time_mask=time_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1,
                                                          self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class ASAP_attention(nn.Module):
    def __init__(self, d_model, n_heads, sigma=2, rwr_prob=0.9, season=None, iteration=2, _attn_dropout=False, attn_dropout=0., res_attention=False, lsa=False, Trend=True):
        super().__init__()
        self.Trend = Trend
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa
        latent = math.log(rwr_prob / (1. - rwr_prob))
        self.prob = torch.nn.Parameter(torch.tensor(latent), requires_grad=True)
        self.iter = iteration
        self._att_dropout = _attn_dropout
        if season is not None:
            self.alpha = nn.Parameter(torch.ones(len(season), dtype=torch.float32), requires_grad=True)
            self.season = torch.tensor(season, dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        if self.Trend:
            self.sigma = sigma
        else:
            self.sigma = 1./3

    def Generate_TemporalMap(self, device, Batch_size, length, mask, sigma=2):

        # Time_1 = (torch.arange(0, length, 1) % 3).unsqueeze(1).expand([length, length]).to(device)
        # Time_2 = (torch.arange(0, length, 1) % 3).unsqueeze(0).expand([length, length]).to(device)
        Time_1 = torch.arange(0, length, 1).unsqueeze(1).expand([length, length]).to(device)
        Time_2 = torch.arange(0, length, 1).unsqueeze(0).expand([length, length]).to(device)
        if self.season is None:
            TemporalMap = torch.exp(-((Time_1 - Time_2) * (Time_1 - Time_2)) / (2 * sigma * sigma)) / (
                math.sqrt(2 * math.pi) * sigma)
            TemporalMap = TemporalMap.unsqueeze(0).expand(Batch_size, length, length)
            TemporalMap = TemporalMap.unsqueeze(1)
            # TemporalMap.masked_fill_(mask, -np.inf)
        else:
            dist = abs(Time_1 -Time_2).unsqueeze(0)
            dist_perodic = torch.remainder(dist, self.season.to(device))
            TemporalMap = torch.exp(-((dist_perodic) * (dist_perodic)) / (2 * 1 * 1)) / (
                math.sqrt(2 * math.pi) * 1)
            # print(TemporalMap.shape)
            TemporalMap = self.alpha.unsqueeze(1).unsqueeze(1) * TemporalMap
            TemporalMap = torch.sum(TemporalMap, dim=0, keepdim=True).expand(Batch_size, length, length).unsqueeze(1)
            
        return torch.softmax(TemporalMap, dim=-2)

    def Generate_StartPoint(self, device, Batch_size, length, heads):
        e = torch.eye(length, dtype=torch.float).to(device)
        e = e.unsqueeze(0).expand(Batch_size, length, length)
        e = e.unsqueeze(1).expand(Batch_size, heads, length, length)
        return e

    def random_walk(self, prob, e, transition, pre_d):
        d = prob*torch.matmul(transition, pre_d) + (1 - prob)*e
        return d

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None, time_mask=None):
        '''
                Input shape:
                    q               : [bs x n_heads x max_q_len x d_k]
                    k               : [bs x n_heads x d_k x seq_len]
                    v               : [bs x n_heads x seq_len x d_v]
                    prev            : [bs x n_heads x q_len x seq_len]
                    key_padding_mask: [bs x seq_len]
                    attn_mask       : [1 x seq_len x seq_len]
                Output shape:
                    output:  [bs x n_heads x q_len x d_v]
                    attn   : [bs x n_heads x q_len x seq_len]
                    scores : [bs x n_heads x q_len x seq_len]
                '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        B, H, L, _ = q.shape
        attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]
        TimeMap = self.Generate_TemporalMap(device=attn_scores.device, length=L,mask=time_mask.mask, Batch_size=B, sigma=self.sigma)
        d = self.Generate_StartPoint(device=attn_scores.device, Batch_size=B, length=L, heads=H).requires_grad_()
        e = self.Generate_StartPoint(device=attn_scores.device, Batch_size=B, length=L, heads=H)

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)
            print("key_padding_mask")

        # normalize the attention weights
        AttnMap = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        A = torch.matmul(TimeMap, AttnMap.transpose(-1, -2))
        for i in range(self.iter):
            d = self.random_walk(prob=torch.sigmoid(self.prob), e=e, transition=A, pre_d=d)
        attn_weights = d
        if self._att_dropout:
            attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights

