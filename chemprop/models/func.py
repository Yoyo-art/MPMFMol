import torch
from torch import nn
import math
import torch.nn.functional as F
from argparse import Namespace

from chemprop.nn_utils import get_activation_function


def attention(query, key, value, mask, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class AttentionLayer(nn.Module):
    def __init__(self, args):
        super(AttentionLayer, self).__init__()
        self.hidden_size = args.hidden_size
        self.w_q = nn.Linear(self.hidden_size, 32)
        self.w_k = nn.Linear(self.hidden_size, 32)
        self.w_v = nn.Linear(self.hidden_size, 32)
        self.args = args
        self.dense = nn.Linear(32, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, fg_hiddens, init_hiddens):
        query = self.w_q(fg_hiddens)
        key = self.w_k(fg_hiddens)
        value = self.w_v(fg_hiddens)

        padding_mask = (init_hiddens != 0) + 0.0
        mask = torch.matmul(padding_mask, padding_mask.transpose(-2, -1))
        x, attn = attention(query, key, value, mask)
        hidden_states = self.dense(x)
        hidden_states = self.dropout(hidden_states)
        res = hidden_states + fg_hiddens
        hidden_states = self.LayerNorm(hidden_states + fg_hiddens)

        return hidden_states, attn
class Prompt_generator(nn.Module):
    def __init__(self, args:Namespace):
        super(Prompt_generator, self).__init__()
        self.hidden_size = args.hidden_size
        # Activation
        self.act_func = get_activation_function(args.activation)
        # add frage attention
        self.fg = nn.Parameter(torch.randn(1,self.hidden_size*3), requires_grad=True)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        # self.alpha.data.fill_(0.1)
        self.atten_layers = nn.ModuleList([AttentionLayer(args) for _ in range(args.num_attention)])
        self.linear = nn.Linear(self.hidden_size,self.hidden_size)
        self.lr = nn.Linear(self.hidden_size*3,self.hidden_size)
        self.norm = nn.LayerNorm(args.hidden_size)
    def forward(self, f_atom, atom_scope, f_group, group_scope, mapping, mapping_scope):
        max_frag_size = max([g_size for _,g_size in group_scope])+1 # 加上一行填充位置
        f_frag_list = []
        padding_zero = torch.zeros((1,self.hidden_size)).cuda()
        for i,(g_start, g_size) in enumerate(group_scope):#每个分子包含官能团的起始范围
            a_start,a_size = atom_scope[i]#每个分子包含原子的起始范围
            m_start,m_size = mapping_scope[i]#每个分子官能团到原子索引的起始范围
            cur_a = f_atom.narrow(0,a_start,a_size)#当前分子的原子特征
            cur_g = f_group.narrow(0,g_start,g_size)#功能团嵌入
            cur_m = mapping.narrow(0,m_start,m_size) #映射关系
            cur_a = torch.cat([padding_zero,cur_a],dim=0) #将原子特征填充到index=0
            cur_a = cur_a[cur_m]#从f_atom中提取与官能团对应的原子特征
            cur_g = torch.cat([cur_a.sum(dim=1),cur_a.max(dim=1)[0],cur_g],dim=1)#构建每个官能团表示
            cur_brics = torch.cat([self.fg,cur_g],dim=0)#添加全局提示
            cur_frage = torch.nn.ZeroPad2d((0,0,0,max_frag_size-cur_brics.shape[0]))(cur_brics)#分子官能团个数补齐
            f_frag_list.append(cur_frage.unsqueeze(0))
        f_frag_list = torch.cat(f_frag_list, 0)
        f_frag_list = self.act_func(self.lr(f_frag_list))
        hidden_states,self_att = self.atten_layers[0](f_frag_list,f_frag_list)
        for k,att in enumerate(self.atten_layers[1:]):
            hidden_states,self_att = att(hidden_states,f_frag_list)
        f_out = self.linear(hidden_states)
        f_out = self.norm(f_out)* self.alpha
        return f_out[:,0,:],self_att