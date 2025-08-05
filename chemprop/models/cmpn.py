from argparse import Namespace
from typing import List

import numpy as np
import torch
from torch import nn

from chemprop.data.pro_smi import batch_encode_smiles
from chemprop.models.func import Prompt_generator
from chemprop.models.transformer import make_model
from chemprop.features.featurization import get_atom_fdim, get_bond_fdim, mol2graph, smiles_to_tensor, \
    smiles_batch_to_func
from chemprop.nn_utils import get_activation_function, index_select_ND
import torch.nn.functional as F
import math

class CMPNEncoder(nn.Module):
    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        super(CMPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.args = args
        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Input
        input_dim = self.atom_fdim
        self.W_i_atom = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        input_dim = self.bond_fdim
        self.W_i_bond = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        
        w_h_input_size_atom = self.hidden_size + self.bond_fdim
        self.W_h_atom = nn.Linear(w_h_input_size_atom, self.hidden_size, bias=self.bias)
        
        w_h_input_size_bond = self.hidden_size
            
        for depth in range(self.depth-1):
            self._modules[f'W_h_{depth}'] = nn.Linear(w_h_input_size_bond, self.hidden_size, bias=self.bias)
        
        self.W_o = nn.Linear(
                (self.hidden_size)*2,
                self.hidden_size)
        self.only_gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True,
                            bidirectional=True)
        self.gru = BatchGRU(self.hidden_size)
        self.lr = nn.Linear(self.hidden_size*3, self.hidden_size, bias=self.bias)
        # add & concat functional group features
        if self.args.add_step == 'func_frag_attention':
            self.W_i_atom_new = nn.Linear(self.atom_fdim*2, self.hidden_size, bias=self.bias)
        elif self.args.add_step =='concat_mol_frag_attention':
            self.funcional_group_embedding = FunctionalGroupEmbedding(82,self.hidden_size)
            self.W_molecular = nn.Linear(self.hidden_size*2,self.hidden_size)
    def forward(self, step, mol_graph, batch, features_batch=None) -> torch.FloatTensor:
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, bonds = mol_graph.get_components()
        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = (
                f_atoms.cuda(), f_bonds.cuda(),
                a2b.cuda(), b2a.cuda(), b2revb.cuda())
        input_atom = self.W_i_atom(f_atoms)  # num_atoms x hidden_size
        input_atom = self.act_func(input_atom)
        message_atom = input_atom.clone()

        input_bond = self.W_i_bond(f_bonds)  # num_bonds x hidden_size
        message_bond = self.act_func(input_bond)
        input_bond = self.act_func(input_bond)
        # Message passing
        for depth in range(self.depth - 1):
            agg_message = index_select_ND(message_bond, a2b)
            agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
            message_atom = message_atom + agg_message

            # directed graph
            rev_message = message_bond[b2revb]  # num_bonds x hidden
            message_bond = message_atom[b2a] - rev_message  # num_bonds x hidden

            message_bond = self._modules[f'W_h_{depth}'](message_bond)
            message_bond = self.dropout_layer(self.act_func(input_bond + message_bond))

        agg_message = index_select_ND(message_bond, a2b)
        agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
        agg_message = self.lr(torch.cat([agg_message, message_atom, input_atom], 1))
        agg_message = self.gru(agg_message, a_scope)
        atom_hiddens = self.act_func(self.W_o(agg_message))  # num_atoms x hidden
        if self.args.add_step == '':
            atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
            mol_vecs.append(cur_hiddens.mean(0))
        out = torch.stack(mol_vecs, dim=0)
        # 官能团引导
        a_scope, mapping_scope, func2atom, func2atom_scope, mapping = smiles_batch_to_func(batch)
        if self.args.step == 'func_prompt' and self.args.add_step == 'concat_mol_frag_attention':
            f_group = self.funcional_group_embedding(mapping.cuda())  # get random functional group embedding
            pharm_hiddens, self.self_att = self.prompt_generator(atom_hiddens, a_scope, f_group,
                                                                 mapping_scope, func2atom,
                                                                 func2atom_scope)
            # out = self.act_func(self.W_molecular(torch.cat([out, pharm_hiddens], dim=-1)))  # 300
            # out = self.act_func(torch.cat([out, pharm_hiddens], dim=-1))
            out = self.act_func(pharm_hiddens)
            out = self.dropout_layer(out)
            # out = pharm_hiddens

        return out  # B x H

class FunctionalGroupEmbedding(nn.Module):
    def __init__(self, num_groups, features_dim):
        super(FunctionalGroupEmbedding, self).__init__()
        self.embedding = nn.Parameter(torch.Tensor(num_groups, features_dim),requires_grad=True)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding)
    
    def forward(self, group_indices):
        # 通过索引获取对应的特征向量
        return F.embedding(group_indices, self.embedding)


class modelTransformer_smi(nn.Module):
    def __init__(self, args):
        super(modelTransformer_smi, self).__init__()
        self.num_features = 66
        self.max_features = 100
        self.dropout = 0.0
        self.num_layer = 6
        self.num_heads = 8
        self.hidden_dim = 256
        self.output_dim = 256
        self.n_output = 1
        self.hidden_size = 300
        self.args = args

        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        self.encoder = make_model(self.num_features, self.num_layer, self.hidden_dim, self.output_dim, self.num_heads, self.dropout)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_size)
        # if self.args.add_step == 'concat_mol_frag_attention':
        #     self.funcional_group_embedding = FunctionalGroupEmbedding(82, self.hidden_size)
        #     self.W_molecular = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.prompt_generator = Prompt_generator(args)
    def set_fc1(self, max_len):
        self.fc1 = nn.Linear(max_len, self.n_output).cuda()

    def forward(self, step, pretrain: bool, batch,
                features_batch: List[np.ndarray] = None):
        encodedSmi, encodedSmi_mask, max_len = smiles_to_tensor(batch)
        # encodedSmi, encodedSmi_mask, max_len = batch_encode_smiles(batch)
        smi_mask = encodedSmi_mask.unsqueeze(1).cuda()
        smi_encoded = self.encoder(encodedSmi.cuda(), smi_mask)
        mask = smi_mask.squeeze(1)
        mask = mask.unsqueeze(-1)  # [B, L] → [B, L, 1]
        #cls token
        # cls_vector = smi_encoded[:, 0, :]
        #平均池化
        # x_masked1 = smi_encoded * mask.float()
        # sum_x = x_masked1.sum(dim=1)  # [B, H]
        # valid_counts = mask.sum(dim=1)  # [B, 1]
        # smi_ln1 = sum_x / valid_counts.clamp(min=1e-6)
        #最大池化
        neg_inf = torch.finfo(smi_encoded.dtype).min
        x_masked = smi_encoded.masked_fill(~mask.bool(), neg_inf)  # 用-∞替代无效位置

        pooled, _ = x_masked.max(dim=1)
        #拼接平均+最大
        # smi_ln = torch.cat([smi_ln1,pooled], dim=1)

        out = self.fc2(pooled)
        out = self.act_func(out)
        out = self.dropout_layer(out)
        return out
class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru  = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, 
                            bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size), 
                                1.0 / math.sqrt(self.hidden_size))


    def forward(self, node, a_scope):
        hidden = node
        message = F.relu(node + self.bias)
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])
        # padding
        message_lst = []
        hidden_lst = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))
            
            cur_message = torch.nn.ZeroPad2d((0,0,0,MAX_atom_len-cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))
            
        message_lst = torch.cat(message_lst, 0)
        hidden_lst  = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2,1,1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)
        
        # unpadding
        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, 2*self.hidden_size))
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)
        
        message = torch.cat([torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1), 
                             cur_message_unpadding], 0)
        return message

class CMPN(nn.Module):
    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        super(CMPN, self).__init__()
        args.atom_fdim = atom_fdim or get_atom_fdim()
        args.bond_fdim = bond_fdim or get_bond_fdim() + \
                            (not args.atom_messages) * args.atom_fdim # * 2
        self.graph_input = graph_input
        self.args = args
        self.atom_fdim = args.atom_fdim
        self.bond_fdim = args.bond_fdim
        self.encoder = CMPNEncoder(self.args, self.atom_fdim, self.bond_fdim)
        # self.smi_encoder = modelTransformer_smi()

    def forward(self, step, pretrain: bool, batch,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        if not self.graph_input:  # if features only, batch won't even be used
            mol_batch = mol2graph(batch, self.args)
            # smi_batch, smi_mask = smiles_to_tensor(batch)
        output = self.encoder.forward(step, mol_batch, batch, features_batch)
        return output

