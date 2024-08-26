import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
from .model_utils import encode_onehot

class GraphRNNDecoder(nn.Module):
    def __init__(self, params):
        super(GraphRNNDecoder, self).__init__()
        self.num_vars = params['num_vars']
        self.input_size = params['input_size']
        self.hidden_size = params['decoder_hidden']
        self.gpu = params['gpu']
        self.dropout_prob = params['decoder_dropout']

        # GRU layer
        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)

        # Fully connected layers for output
        self.out_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_fc3 = nn.Linear(self.hidden_size, self.input_size)

        print('Using GRU-based decoder.')

        # Initialize edge-to-node matrix
        edges = np.ones(self.num_vars) - np.eye(self.num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = torch.FloatTensor(encode_onehot(self.recv_edges))
        if self.gpu:
            self.edge2node_mat = self.edge2node_mat.cuda(non_blocking=True)

    def single_step_forward(self, inputs, hidden):
        # Inputs: [batch, num_atoms, num_dims]
        # Hidden: [batch, num_atoms, hidden_size]

        # Process each node with GRU
        pred, hidden = self.gru(inputs.unsqueeze(1), hidden)

        # Reshape prediction back to [batch, num_atoms, num_dims]
        pred = pred.squeeze(1)

        # Output MLP layers
        pred = F.dropout(F.relu(self.out_fc1(pred)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Skip connection
        pred = inputs + pred

        return pred, hidden

    def forward(self, inputs, sampled_edges, teacher_forcing=False, teacher_forcing_steps=-1, return_state=False,
                prediction_steps=-1, state=None, burn_in_masks=None):

        time_steps = inputs.size(1)

        if prediction_steps > 0:
            pred_steps = prediction_steps
        else:
            pred_steps = time_steps

        if state is None:
            if inputs.is_cuda:
                hidden = torch.cuda.FloatTensor(1, inputs.size(0) * self.num_vars, self.hidden_size).fill_(0.)
            else:
                hidden = torch.zeros(1, inputs.size(0) * self.num_vars, self.hidden_size)
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

            pred, hidden = self.single_step_forward(ins, hidden)

            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)

        if return_state:
            return preds, hidden
        else:
            return preds
