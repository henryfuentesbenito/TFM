
from torch.nn import init

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .model_utils import encode_onehot, RefNRIMLP


class BaseEncoder(nn.Module):
    def __init__(self, num_vars, graph_type):
        super(BaseEncoder, self).__init__()
        self.num_vars = num_vars
        self.graph_type = graph_type
        self.dynamic = graph_type == 'dynamic'

        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges).transpose()), requires_grad=False)
    
    def node2edge(self, node_embeddings):
        if self.dynamic:
            send_embed = node_embeddings[:, self.send_edges, :, :]
            recv_embed = node_embeddings[:, self.recv_edges, :, :]
            return torch.cat([send_embed, recv_embed], dim=3)
        else:
            send_embed = node_embeddings[:, self.send_edges, :]
            recv_embed = node_embeddings[:, self.recv_edges, :]
            return torch.cat([send_embed, recv_embed], dim=2)

    def edge2node(self, edge_embeddings):
        if self.dynamic:
            old_shape = edge_embeddings.shape
            tmp_embeddings = edge_embeddings.view(old_shape[0], old_shape[1], -1)
            incoming = torch.matmul(self.edge2node_mat, tmp_embeddings).view(old_shape[0], -1, old_shape[2], old_shape[3])
        else:
            incoming = torch.matmul(self.edge2node_mat, edge_embeddings)
        return incoming / (self.num_vars - 1)  # Averaging incoming messages

    def forward(self, inputs, state=None, return_state=False):
        raise NotImplementedError

class RefMLPEncoder(BaseEncoder):
    def __init__(self, params):
        num_vars = params['num_vars']
        inp_size = params['input_size'] * params['input_time_steps']
        hidden_size = params['encoder_hidden']
        num_edges = params['num_edge_types']
        graph_type = params['graph_type']
        super(RefMLPEncoder, self).__init__(num_vars, graph_type)
        
        # Transformer Configuration
        self.transformer = nn.Transformer(
            d_model=hidden_size, 
            nhead=params['num_heads'], 
            num_encoder_layers=params['num_layers'], 
            num_decoder_layers=params['num_layers'], 
            dim_feedforward=params['ffn_dim'],
            dropout=params['encoder_dropout']
        )

        self.input_proj = nn.Linear(inp_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, num_edges)

        self.init_weights()

    def node2edge(self, node_embeddings):
        send_embed = node_embeddings[:, self.send_edges, :]
        recv_embed = node_embeddings[:, self.recv_edges, :]
        return torch.cat([send_embed, recv_embed], dim=2)

    def edge2node(self, edge_embeddings):
        incoming = torch.matmul(self.edge2node_mat, edge_embeddings)
        return incoming / (self.num_vars - 1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs, state=None, return_state=False):
        if inputs.size(1) > self.input_time_steps:
            inputs = inputs[:, -self.input_time_steps:]
        elif inputs.size(1) < self.input_time_steps:
            begin_inp = inputs[:, 0:1].expand(-1, self.input_time_steps-inputs.size(1), -1, -1)
            inputs = torch.cat([begin_inp, inputs], dim=1)
        
        if state is not None:
            inputs = torch.cat([state, inputs], 1)[:, -self.input_time_steps:]

        x = inputs.transpose(1, 2).contiguous().view(inputs.size(0), inputs.size(2), -1)
        x = self.input_proj(x)  # Project input to the Transformer dimensions

        x = self.transformer(x.permute(1, 0, 2))  # Transformer expects input as (seq_len, batch, feature)
        x = x.permute(1, 0, 2)  # Permute back to original shape
        
        result = self.fc_out(x)
        result_dict = {
            'logits': result,
            'state': inputs,
        }
        return result_dict
