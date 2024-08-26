
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
        return incoming/(self.num_vars-1)

    def forward(self, inputs, state=None, return_state=False):
        raise NotImplementedError

class RefMLPEncoder(BaseEncoder):
    def __init__(self, params):
        num_vars = params['num_vars']
        inp_size = params['input_size']
        hidden_size = params['encoder_hidden']
        num_edges = params['num_edge_types']
        factor = not params['encoder_no_factor']
        graph_type = params['graph_type']
        super(RefGRUEncoder, self).__init__(num_vars, graph_type)
        dropout = params['encoder_dropout']
        self.input_time_steps = params['input_time_steps']
        self.dynamic = self.graph_type == 'dynamic'
        self.factor = factor

        # GRU layer
        self.gru = nn.GRU(input_size=inp_size, hidden_size=hidden_size, batch_first=True)

        if self.factor:
            self.mlp4 = RefNRIMLP(hidden_size * 3, hidden_size, hidden_size, dropout, no_bn=False)
            print("Using factor graph GRU encoder.")
        else:
            self.mlp4 = RefNRIMLP(hidden_size * 2, hidden_size, hidden_size, dropout, no_bn=False)
            print("Using GRU encoder.")

        num_layers = params['encoder_mlp_num_layers']
        if num_layers == 1:
            self.fc_out = nn.Linear(hidden_size, num_edges)
        else:
            tmp_hidden_size = params['encoder_mlp_hidden']
            layers = [nn.Linear(hidden_size, tmp_hidden_size), nn.ELU(inplace=True)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(tmp_hidden_size, tmp_hidden_size))
                layers.append(nn.ELU(inplace=True))
            layers.append(nn.Linear(tmp_hidden_size, num_edges))
            self.fc_out = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def merge_states(self, states):
        return torch.cat(states, dim=0)

    def forward(self, inputs, state=None, return_state=False):
        if inputs.size(1) > self.input_time_steps:
            inputs = inputs[:, -self.input_time_steps:]
        elif inputs.size(1) < self.input_time_steps:
            begin_inp = inputs[:, 0:1].expand(-1, self.input_time_steps - inputs.size(1), -1, -1)
            inputs = torch.cat([begin_inp, inputs], dim=1)
        
        if state is not None:
            inputs = torch.cat([state, inputs], 1)[:, -self.input_time_steps:]

        # Pass through GRU
        x, hidden = self.gru(inputs)

        # Combine node embeddings to form edge embeddings
        x = self.node2edge(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x)
            x = torch.cat((x, x_skip), dim=-1)  # Skip connection
            x = self.mlp4(x)
        else:
            x = torch.cat((x, x_skip), dim=-1)  # Skip connection
            x = self.mlp4(x)

        result = self.fc_out(x)

        result_dict = {
            'logits': result,
            'state': hidden if return_state else None,
        }
        return result_dict
