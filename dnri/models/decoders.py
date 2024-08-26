import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
from .model_utils import encode_onehot

class GraphRNNDecoder(nn.Module):
    def __init__(self, params):
        super(GraphRNNDecoder, self).__init__()
        self.num_vars = num_vars = params['num_vars']
        input_size = params['input_size']
        self.gpu = params['gpu']
        n_hid = params['decoder_hidden']
        edge_types = params['num_edge_types']
        skip_first = params['skip_first']
        out_size = params['input_size']
        do_prob = params['decoder_dropout']

        # Transformer configuration
        self.transformer = nn.Transformer(
            d_model=n_hid, 
            nhead=params['num_heads'], 
            num_encoder_layers=params['num_layers'], 
            num_decoder_layers=params['num_layers'], 
            dim_feedforward=params['ffn_dim'],
            dropout=do_prob
        )

        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2*n_hid, n_hid) for _ in range(edge_types)]
        )
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)]
        )
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        self.input_proj = nn.Linear(input_size, n_hid)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, out_size)

        print('Using Transformer interaction net decoder.')

        self.dropout_prob = do_prob

        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = torch.FloatTensor(encode_onehot(self.recv_edges))
        if self.gpu:
            self.edge2node_mat = self.edge2node_mat.cuda(non_blocking=True)

    def single_step_forward(self, inputs, rel_type, hidden):
        receivers = hidden[:, self.recv_edges, :]
        senders = hidden[:, self.send_edges, :]

        pre_msg = torch.cat([receivers, senders], dim=-1)

        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape, device=inputs.device)
        
        if self.skip_first_edge_type:
            start_idx = 1
            norm = float(len(self.msg_fc2)) - 1
        else:
            start_idx = 0
            norm = float(len(self.msg_fc2))

        for i in range(start_idx, len(self.msg_fc2)):
            msg = torch.tanh(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = torch.tanh(self.msg_fc2[i](msg))
            msg = msg * rel_type[:, :, i:i+1]
            all_msgs += msg / norm

        agg_msgs = all_msgs.transpose(-2, -1).matmul(self.edge2node_mat).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous() / (self.num_vars - 1)

        # Transformer block
        transformer_input = self.input_proj(inputs)
        hidden = self.transformer(transformer_input.permute(1, 0, 2), agg_msgs.permute(1, 0, 2)).permute(1, 0, 2)

        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        pred = inputs + pred

        return pred, hidden

    def forward(self, inputs, sampled_edges, teacher_forcing=False, teacher_forcing_steps=-1, return_state=False,
                prediction_steps=-1, state=None, burn_in_masks=None):

        time_steps = inputs.size(1)

        if prediction_steps > 0:
            pred_steps = prediction_steps
        else:
            pred_steps = time_steps

        if len(sampled_edges.shape) == 3:
            sampled_edges = sampled_edges.unsqueeze(1).expand(sampled_edges.size(0), pred_steps, sampled_edges.size(1), sampled_edges.size(2))

        if state is None:
            hidden = torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape, device=inputs.device)
        else:
            hidden = state

        if teacher_forcing_steps == -1:
            teacher_forcing_steps = inputs.size(1)

        pred_all = []
        for step in range(0, pred_steps):
            if burn_in_masks is not None and step != 0:
                current_masks = burn_in_masks[:, step, :]
                ins = inputs[:, step, :] * current_masks + pred_all[-1] * (1 - current_masks)
            elif step == 0 or (teacher_forcing and step < teacher_forcing_steps):
                ins = inputs[:, step, :]
            else:
                ins = pred_all[-1]
            edges = sampled_edges[:, step, :]
            pred, hidden = self.single_step_forward(ins, edges, hidden)

            pred_all.append(pred)
        preds = torch.stack(pred_all, dim=1)

        if return_state:
            return preds, hidden
        else:
            return preds
