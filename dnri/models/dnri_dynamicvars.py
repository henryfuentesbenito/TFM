import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import dnri.models.model_utils as model_utils
from .model_utils import RefNRIMLP, encode_onehot, get_graph_info

class DNRI_DynamicVars(nn.Module):
    def __init__(self, params):
        super(DNRI_DynamicVars, self).__init__()
        # Model Params
        self.encoder = DNRI_DynamicVars_Encoder(params)
        self.decoder = DNRI_DynamicVars_Decoder(params)
        self.num_edge_types = params.get('num_edge_types')

        # Training params
        self.gumbel_temp = params.get('gumbel_temp')
        self.train_hard_sample = params.get('train_hard_sample')
        self.teacher_forcing_steps = params.get('teacher_forcing_steps', -1)

        self.normalize_kl = params.get('normalize_kl', False)
        self.normalize_kl_per_var = params.get('normalize_kl_per_var', False)
        self.normalize_nll = params.get('normalize_nll', False)
        self.normalize_nll_per_var = params.get('normalize_nll_per_var', False)
        self.kl_coef = params.get('kl_coef', 1.)
        self.nll_loss_type = params.get('nll_loss_type', 'gaussian')  # Cambiado a 'gaussian' por defecto
        self.prior_variance = params.get('prior_variance')
        self.timesteps = params.get('timesteps', 0)

        self.burn_in_steps = params.get('train_burn_in_steps')
        self.no_prior = params.get('no_prior', False)
        self.avg_prior = params.get('avg_prior', False)
        self.learned_prior = params.get('use_learned_prior', False)
        self.anneal_teacher_forcing = params.get('anneal_teacher_forcing', False)
        self.teacher_forcing_prior = params.get('teacher_forcing_prior', False)
        self.steps = 0
        self.gpu = params.get('gpu')

    def single_step_forward(self, inputs, node_masks, graph_info, decoder_hidden, edge_logits, hard_sample):
        old_shape = edge_logits.shape
        edges = model_utils.gumbel_softmax(
            edge_logits.reshape(-1, self.num_edge_types), 
            tau=self.gumbel_temp, 
            hard=hard_sample).view(old_shape)
        predictions, decoder_hidden = self.decoder(inputs, decoder_hidden, edges, node_masks, graph_info)
        return predictions, decoder_hidden, edges

    def calculate_loss(self, inputs, node_masks, node_inds, graph_info, is_train=False, teacher_forcing=True, return_edges=False, return_logits=False, use_prior_logits=False, normalized_inputs=None):
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        num_time_steps = inputs.size(1)
        all_edges = []
        all_predictions = []
        hard_sample = (not is_train) or self.train_hard_sample
        prior_logits, posterior_logits, _ = self.encoder(inputs[:, :-1], node_masks[:, :-1], node_inds, graph_info, normalized_inputs)
        if self.anneal_teacher_forcing:
            teacher_forcing_steps = math.ceil((1 - self.train_percent) * num_time_steps)
        else:
            teacher_forcing_steps = self.teacher_forcing_steps
        edge_ind = 0
        for step in range(num_time_steps-1):
            if (teacher_forcing and (teacher_forcing_steps == -1 or step < teacher_forcing_steps)) or step == 0:
                current_inputs = inputs[:, step]
            else:
                current_inputs = predictions
            current_node_masks = node_masks[:, step]
            node_inds = current_node_masks.nonzero()[:, -1]
            num_edges = len(node_inds) * (len(node_inds) - 1)
            current_graph_info = graph_info[0][step]
            if not use_prior_logits:
                current_p_logits = posterior_logits[:, edge_ind:edge_ind + num_edges]
            else:
                current_p_logits = prior_logits[:, edge_ind:edge_ind + num_edges]
            if self.gpu:
                current_p_logits = current_p_logits.cuda(non_blocking=True)
            edge_ind += num_edges
            predictions, decoder_hidden, edges = self.single_step_forward(current_inputs, current_node_masks, current_graph_info, decoder_hidden, current_p_logits, hard_sample)
            all_predictions.append(predictions)
            all_edges.append(edges)
        all_predictions = torch.stack(all_predictions, dim=1)
        target = inputs[:, 1:, :, :]
        target_masks = ((node_masks[:, :-1] == 1) * (node_masks[:, 1:] == 1)).float()
        loss_nll = self.nll(all_predictions, target, target_masks)
        prob = F.softmax(posterior_logits, dim=-1)
        if self.gpu:
            prob = prob.cuda(non_blocking=True)
            prior_logits = prior_logits.cuda(non_blocking=True)
        loss_kl = self.kl_categorical_learned(prob, prior_logits)
        loss = loss_nll + self.kl_coef * loss_kl
        loss = loss.mean()
        if return_edges:
            return loss, loss_nll, loss_kl, edges
        elif return_logits:
            return loss, loss_nll, loss_kl, posterior_logits, all_predictions
        else:
            return loss, loss_nll, loss_kl

    def nll(self, preds, target, masks):
        if self.nll_loss_type == 'gaussian':
            return self.nll_gaussian(preds, target, masks)
        elif self.nll_loss_type == 'crossent':
            return self.nll_crossent(preds, target, masks)
        elif self.nll_loss_type == 'poisson':
            return self.nll_poisson(preds, target, masks)

    def nll_gaussian(self, preds, target, masks, add_const=False):
        neg_log_p = ((preds - target) ** 2 / (2 * self.prior_variance)) * masks.unsqueeze(-1)
        const = 0.5 * np.log(2 * np.pi * self.prior_variance)
        if self.normalize_nll_per_var:
            raise NotImplementedError()
        elif self.normalize_nll:
            return (neg_log_p.sum(-1) + const * masks).view(preds.size(0), -1).sum(dim=-1) / (masks.view(masks.size(0), -1).sum(dim=1) + 1e-8)
        else:
            raise NotImplementedError()

    def nll_crossent(self, preds, target, masks):
        if self.normalize_nll:
            loss = nn.BCEWithLogitsLoss(reduction='none')(preds, target)
            return (loss * masks.unsqueeze(-1)).view(preds.size(0), -1).sum(dim=-1) / (masks.view(masks.size(0), -1).sum(dim=1))
        else:
            raise NotImplementedError()

    def kl_categorical_learned(self, preds, prior_logits):
        log_prior = nn.LogSoftmax(dim=-1)(prior_logits)
        kl_div = preds * (torch.log(preds + 1e-16) - log_prior)
        if self.normalize_kl:
            return kl_div.sum(-1).view(preds.size(0), -1).mean(dim=1)
        elif self.normalize_kl_per_var:
            raise NotImplementedError()
        else:
            raise NotImplementedError()

class DNRI_DynamicVars_Encoder(nn.Module):
    def __init__(self, params):
        super(DNRI_DynamicVars_Encoder, self).__init__()
        self.num_edges = params['num_edge_types']
        self.gpu = params.get('gpu')
        no_bn = params['no_encoder_bn']
        dropout = params['encoder_dropout']

        hidden_size = params['encoder_hidden']
        self.rnn_hidden_size = rnn_hidden_size = params['encoder_rnn_hidden']
        rnn_type = params['encoder_rnn_type']
        inp_size = params['input_size']
        self.mlp1 = RefNRIMLP(inp_size, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp2 = RefNRIMLP(hidden_size * 2, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp3 = RefNRIMLP(hidden_size, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp4 = RefNRIMLP(hidden_size * 3, hidden_size, hidden_size, dropout, no_bn=no_bn)

        if rnn_hidden_size is None:
            rnn_hidden_size = hidden_size
        if rnn_type == 'lstm':
            self.forward_rnn = nn.LSTM(hidden_size, rnn_hidden_size, batch_first=True)
            self.reverse_rnn = nn.LSTM(hidden_size, rnn_hidden_size, batch_first=True)
        elif rnn_type == 'gru':
            self.forward_rnn = nn.GRU(hidden_size, rnn_hidden_size, batch_first=True)
            self.reverse_rnn = nn.GRU(hidden_size, rnn_hidden_size, batch_first=True)
        out_hidden_size = 2 * rnn_hidden_size
        num_layers = params['encoder_mlp_num_layers']
        if num_layers == 1:
            self.encoder_fc_out = nn.Linear(out_hidden_size, self.num_edges)
        else:
            tmp_hidden_size = params['encoder_mlp_hidden']
            layers = [nn.Linear(out_hidden_size, tmp_hidden_size), nn.ELU(inplace=True)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(tmp_hidden_size, tmp_hidden_size))
                layers.append(nn.ELU(inplace=True))
            layers.append(nn.Linear(tmp_hidden_size, self.num_edges))
            self.encoder_fc_out = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs, node_masks, all_node_inds, all_graph_info, normalized_inputs=None):
        if self.normalize_mode == 'normalize_all':
            x = torch.cat(normalized_inputs, dim=0)
        else:
            raise NotImplementedError
        # Inputs is shape [batch, num_timesteps, num_vars, input_size]
        num_timesteps = node_masks.size(1)
        max_num_vars = inputs.size(2)
        max_num_edges = max_num_vars * (max_num_vars - 1)
        forward_state = (torch.zeros(1, max_num_edges, self.rnn_hidden_size, device=inputs.device),
                         torch.zeros(1, max_num_edges, self.rnn_hidden_size, device=inputs.device))
        reverse_state = (torch.zeros(1, max_num_edges, self.rnn_hidden_size, device=inputs.device),
                         torch.zeros(1, max_num_edges, self.rnn_hidden_size, device=inputs.device))
        all_x = []
        all_forward_states = []
        all_reverse_states = []
        x_ind = 0
        for timestep in range(num_timesteps):
            current_node_masks = node_masks[:, timestep]
            node_inds = all_node_inds[0][timestep]
            if len(node_inds) <= 1:
                all_forward_states.append(torch.empty(1, 0, self.rnn_hidden_size, device=inputs.device))
                all_x.append(None)
                continue
            send_edges, recv_edges, _ = all_graph_info[0][timestep]
            if self.gpu:
                send_edges, recv_edges = send_edges.cuda(non_blocking=True), recv_edges.cuda(non_blocking=True)
            global_send_edges = node_inds[send_edges]
            global_recv_edges = node_inds[recv_edges]
            global_edge_inds = global_send_edges * (max_num_vars - 1) + global_recv_edges - (global_recv_edges >= global_send_edges).long()
            current_x = x[x_ind:x_ind + len(global_send_edges)]
            if self.gpu:
                current_x = current_x.cuda(non_blocking=True)
            x_ind += len(global_send_edges)

            old_shape = current_x.shape
            current_x = current_x.view(old_shape[-2], 1, old_shape[-1])
            current_state = (forward_state[0][:, global_edge_inds], forward_state[1][:, global_edge_inds])
            current_x, current_state = self.forward_rnn(current_x, current_state)
            forward_state[0][:, global_edge_inds] = current_state[0]
            forward_state[1][:, global_edge_inds] = current_state[1]
            all_forward_states.append(current_state[0])

        # Reverse pass
        x_ind = x.size(0)
        for timestep in range(num_timesteps - 1, -1, -1):
            current_node_masks = node_masks[:, timestep]
            node_inds = all_node_inds[0][timestep]
            if len(node_inds) <= 1:
                continue
            send_edges, recv_edges, _ = all_graph_info[0][timestep]
            if self.gpu:
                send_edges, recv_edges = send_edges.cuda(non_blocking=True), recv_edges.cuda(non_blocking=True)
            global_send_edges = node_inds[send_edges]
            global_recv_edges = node_inds[recv_edges]
            global_edge_inds = global_send_edges * (max_num_vars - 1) + global_recv_edges - (global_recv_edges >= global_send_edges).long()
            current_x = x[x_ind - len(global_send_edges):x_ind]
            if self.gpu:
                current_x = current_x.cuda(non_blocking=True)
            x_ind -= len(global_send_edges)
            old_shape = current_x.shape
            current_x = current_x.view(old_shape[-2], 1, old_shape[-1])

            current_state = (reverse_state[0][:, global_edge_inds], reverse_state[1][:, global_edge_inds])
            current_x, current_state = self.reverse_rnn(current_x, current_state)

            reverse_state[0][:, global_edge_inds] = current_state[0]
            reverse_state[1][:[:, global_edge_inds]] = current_state[1]
            all_reverse_states.append(current_state[0])
        all_forward_states = torch.cat(all_forward_states, dim=1)
        all_reverse_states = torch.cat(all_reverse_states, dim=1).flip(1)
        all_states = torch.cat([all_forward_states, all_reverse_states], dim=-1)
        prior_result = self.prior_fc_out(all_forward_states)
        encoder_result = self.encoder_fc_out(all_states)
        return prior_result, encoder_result, forward_state


class DNRI_DynamicVars_Decoder(nn.Module):
    def __init__(self, params):
        super(DNRI_DynamicVars_Decoder, self).__init__()
        input_size = params['input_size']
        self.gpu = params['gpu']
        n_hid = params['decoder_hidden']
        edge_types = params['num_edge_types']
        skip_first = params['skip_first']
        out_size = params['input_size']
        do_prob = params['decoder_dropout']

        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_hid, n_hid) for _ in range(edge_types)]
        )
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)]
        )
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(input_size, n_hid, bias=True)
        self.input_i = nn.Linear(input_size, n_hid, bias=True)
        self.input_n = nn.Linear(input_size, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, out_size)

        print('Using learned recurrent interaction net decoder.')

        self.dropout_prob = do_prob
        
    def forward(self, inputs, hidden, edges, node_masks, graph_info):
        max_num_vars = inputs.size(1)
        node_inds = node_masks.nonzero()[:, -1]

        current_hidden = hidden[:, node_inds]
        current_inputs = inputs[:, node_inds]
        num_vars = current_hidden.size(1)
        
        if num_vars > 1:
            send_edges, recv_edges, edge2node_inds = graph_info
            if self.gpu:
                send_edges, recv_edges, edge2node_inds = send_edges.cuda(non_blocking=True), recv_edges.cuda(non_blocking=True), edge2node_inds.cuda(non_blocking=True)
            global_send_edges = node_inds[send_edges]
            global_recv_edges = node_inds[recv_edges]
            receivers = current_hidden[:, recv_edges]
            senders = current_hidden[:, send_edges]
            pre_msg = torch.cat([receivers, senders], dim=-1)

            all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                            self.msg_out_shape, device=inputs.device)
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
                msg = msg * edges[:, :, i:i+1]
                all_msgs += msg / norm

            incoming = all_msgs[:, edge2node_inds[:, 0], :].clone()
            for i in range(1, edge2node_inds.size(1)):
                incoming += all_msgs[:, edge2node_inds[:, i], :]
            agg_msgs = incoming / (num_vars - 1)
        elif num_vars == 0:
            pred_all = torch.zeros(inputs.size(0), max_num_vars, inputs.size(-1), device=inputs.device)
            return pred_all, hidden
        else:
            agg_msgs = torch.zeros(current_inputs.size(0), num_vars, self.msg_out_shape, device=inputs.device)

        inp_r = self.input_r(current_inputs).view(current_inputs.size(0), num_vars, -1)
        inp_i = self.input_i(current_inputs).view(current_inputs.size(0), num_vars, -1)
        inp_n = self.input_n(current_inputs).view(current_inputs.size(0), num_vars, -1)
        r = torch.sigmoid(inp_r + self.hidden_r(agg_msgs))
        i = torch.sigmoid(inp_i + self.hidden_i(agg_msgs))
        n = torch.tanh(inp_n + r * self.hidden_h(agg_msgs))
        current_hidden = (1 - i) * n + i * current_hidden

        pred = F.dropout(F.relu(self.out_fc1(current_hidden)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        pred = current_inputs + pred
        hidden[:, node_inds] = current_hidden
        pred_all = torch.zeros(inputs.size(0), max_num_vars, inputs.size(-1), device=inputs.device)
        pred_all[0, node_inds] = pred

        return pred_all, hidden
