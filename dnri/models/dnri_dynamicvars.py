import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import dnri.models.model_utils as model_utils
from .model_utils import RefNRIMLP, encode_onehot, get_graph_info
import math


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
        self.nll_loss_type = params.get('nll_loss_type', 'crossent')
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

    def get_graph_info(self, masks):
        num_vars = masks.size(-1)
        edges = torch.ones(num_vars, device=masks.device) - torch.eye(num_vars, device=masks.device)
        tmp = torch.where(edges)
        send_edges = tmp[0]
        recv_edges = tmp[1]
        tmp_inds = torch.tensor(list(range(num_vars)), device=masks.device, dtype=torch.long).unsqueeze_(1) #TODO: should initialize as long
        edge2node_inds = (tmp_inds == recv_edges.unsqueeze(0)).nonzero()[:, 1].contiguous().view(-1, num_vars-1)
        edge_masks = masks[:, :, send_edges]*masks[:, :, recv_edges] #TODO: gotta figure this one out still
        return send_edges, recv_edges, edge2node_inds, edge_masks

    def single_step_forward(self, inputs, node_masks, graph_info, decoder_hidden, edge_logits, hard_sample):
        old_shape = edge_logits.shape
        edges = model_utils.gumbel_softmax(
            edge_logits.reshape(-1, self.num_edge_types), 
            tau=self.gumbel_temp, 
            hard=hard_sample).view(old_shape)
        predictions, decoder_hidden = self.decoder(inputs, decoder_hidden, edges, node_masks, graph_info)
        return predictions, decoder_hidden, edges


    def normalize_inputs(self, inputs, node_masks):
        return self.encoder.normalize_inputs(inputs, node_masks)

    #@profile
    def calculate_loss(self, inputs, node_masks, node_inds, graph_info, is_train=False, teacher_forcing=True, return_edges=False, return_logits=False, use_prior_logits=False, normalized_inputs=None):
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        num_time_steps = inputs.size(1)
        all_edges = []
        all_predictions = []
        all_priors = []
        hard_sample = (not is_train) or self.train_hard_sample
        prior_logits, posterior_logits, _ = self.encoder(inputs[:, :-1], node_masks[:, :-1], node_inds, graph_info, normalized_inputs)
        if self.anneal_teacher_forcing:
            teacher_forcing_steps = math.ceil((1 - self.train_percent)*num_time_steps)
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
            num_edges = len(node_inds)*(len(node_inds)-1)
            current_graph_info = graph_info[0][step]
            if not use_prior_logits:
                current_p_logits = posterior_logits[:, edge_ind:edge_ind+num_edges]
            else:
                current_p_logits = prior_logits[:, edge_ind:edge_ind+num_edges]
            if self.gpu:
                current_p_logits = current_p_logits.cuda(non_blocking=True)
            edge_ind += num_edges
            predictions, decoder_hidden, edges = self.single_step_forward(current_inputs, current_node_masks, current_graph_info, decoder_hidden, current_p_logits, hard_sample)
            all_predictions.append(predictions)
            all_edges.append(edges)
        all_predictions = torch.stack(all_predictions, dim=1)
        target = inputs[:, 1:, :, :]
        target_masks = ((node_masks[:, :-1] == 1)*(node_masks[:, 1:] == 1)).float()
        loss_nll = self.nll(all_predictions, target, target_masks)
        prob = F.softmax(posterior_logits, dim=-1)
        if self.gpu:
            prob = prob.cuda(non_blocking=True)
            prior_logits = prior_logits.cuda(non_blocking=True)
        loss_kl = self.kl_categorical_learned(prob, prior_logits)
        loss = loss_nll + self.kl_coef*loss_kl
        loss = loss.mean()
        if return_edges:
            return loss, loss_nll, loss_kl, edges
        elif return_logits:
            return loss, loss_nll, loss_kl, posterior_logits, all_predictions
        else:
            return loss, loss_nll, loss_kl

    def get_prior_posterior(self, inputs, student_force=False, burn_in_steps=None):
        self.eval()
        posterior_logits = self.encoder(inputs)
        posterior_probs = torch.softmax(posterior_logits, dim=-1)
        prior_hidden = self.prior_model.get_initial_hidden(inputs)
        all_logits = []
        if student_force:
            decoder_hidden = self.decoder.get_initial_hidden(inputs)
            for step in range(burn_in_steps):
                current_inputs= inputs[:, step]
                predictions, prior_hidden, decoder_hidden, _, prior_logits = self.single_step_forward(current_inputs, prior_hidden, decoder_hidden, None, True)
                all_logits.append(prior_logits)
            for step in range(inputs.size(1) - burn_in_steps):
                predictions, prior_hidden, decoder_hidden, _, prior_logits = self.single_step_forward(predictions, prior_hidden, decoder_hidden, None, True)
                all_logits.append(prior_logits)
        else:
            for step in range(inputs.size(1)):
                current_inputs = inputs[:, step]
                prior_logits, prior_hidden = self.prior_model(prior_hidden, current_inputs)
                all_logits.append(prior_logits)
        logits = torch.stack(all_logits, dim=1)
        prior_probs = torch.softmax(logits, dim=-1)
        return prior_probs, posterior_probs

    def get_edge_probs(self, inputs):
        self.eval()
        prior_hidden = self.prior_model.get_initial_hidden(inputs)
        all_logits = []
        for step in range(inputs.size(1)):
            current_inputs = inputs[:, step]
            prior_logits, prior_hidden = self.prior_model(prior_hidden, current_inputs)
            all_logits.append(prior_logits)
        logits = torch.stack(all_logits, dim=1)
        edge_probs = torch.softmax(logits, dim=-1)
        return edge_probs

    def predict_future(self, inputs, masks, node_inds, graph_info, burn_in_masks):
        '''
        Here, we assume the following:
        * inputs contains all of the gt inputs, including for the time steps we're predicting
        * masks keeps track of the variables that are being tracked
        * burn_in_masks is set to 1 whenever we're supposed to feed in that variable's state
          for a given time step
        '''
        total_timesteps = inputs.size(1)
        prior_hidden = self.encoder.get_initial_hidden(inputs)
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        predictions = inputs[:, 0]
        preds = []
        for step in range(total_timesteps-1):
            current_masks = masks[:, step]
            current_burn_in_masks = burn_in_masks[:, step].unsqueeze(-1).type(inputs.dtype)
            current_inps = inputs[:, step]
            current_node_inds = node_inds[0][step] #TODO: check what's passed in here
            current_graph_info = graph_info[0][step]
            encoder_inp = current_burn_in_masks*current_inps + (1-current_burn_in_masks)*predictions
            current_edge_logits, prior_hidden = self.encoder.single_step_forward(encoder_inp, current_masks, current_node_inds, current_graph_info, prior_hidden)
            predictions, decoder_hidden, _ = self.single_step_forward(encoder_inp, current_masks, current_graph_info, decoder_hidden, current_edge_logits, True)
            preds.append(predictions)
        return torch.stack(preds, dim=1)

    def copy_states(self, prior_state, decoder_state):
        if isinstance(prior_state, tuple) or isinstance(prior_state, list):
            current_prior_state = (prior_state[0].clone(), prior_state[1].clone())
        else:
            current_prior_state = prior_state.clone()
        if isinstance(decoder_state, tuple) or isinstance(decoder_state, list):
            current_decoder_state = (decoder_state[0].clone(), decoder_state[1].clone())
        else:
            current_decoder_state = decoder_state.clone()
        return current_prior_state, current_decoder_state

    def merge_hidden(self, hidden):
        if isinstance(hidden[0], tuple) or isinstance(hidden[0], list):
            result0 = torch.cat([x[0] for x in hidden], dim=0)
            result1 = torch.cat([x[1] for x in hidden], dim=0)
            return (result0, result1)
        else:
            return torch.cat(hidden, dim=0)

    def predict_future_fixedwindow(self, inputs, burn_in_steps, prediction_steps, batch_size):
        if self.fix_encoder_alignment:
            prior_logits, _, prior_hidden = self.encoder(inputs)
        else:
            prior_logits, _, prior_hidden = self.encoder(inputs[:, :-1])
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        for step in range(burn_in_steps-1):
            current_inputs = inputs[:, step]
            current_edge_logits = prior_logits[:, step]
            predictions, decoder_hidden, _ = self.single_step_forward(current_inputs, decoder_hidden, current_edge_logits, True)
        all_timestep_preds = []
        for window_ind in range(burn_in_steps - 1, inputs.size(1)-1, batch_size):
            current_batch_preds = []
            prior_states = []
            decoder_states = []
            for step in range(batch_size):
                if window_ind + step >= inputs.size(1):
                    break
                predictions = inputs[:, window_ind + step] 
                current_edge_logits, prior_hidden = self.encoder.single_step_forward(predictions, prior_hidden)
                predictions, decoder_hidden, _ = self.single_step_forward(predictions, decoder_hidden, current_edge_logits, True)
                current_batch_preds.append(predictions)
                tmp_prior, tmp_decoder = self.copy_states(prior_hidden, decoder_hidden)
                prior_states.append(tmp_prior)
                decoder_states.append(tmp_decoder)
            batch_prior_hidden = self.merge_hidden(prior_states)
            batch_decoder_hidden = self.merge_hidden(decoder_states)
            current_batch_preds = torch.cat(current_batch_preds, 0)
            current_timestep_preds = [current_batch_preds]
            for step in range(prediction_steps - 1):
                current_batch_edge_logits, batch_prior_hidden = self.encoder.single_step_forward(current_batch_preds, batch_prior_hidden)
                current_batch_preds, batch_decoder_hidden, _ = self.single_step_forward(current_batch_preds, batch_decoder_hidden, current_batch_edge_logits, True)
                current_timestep_preds.append(current_batch_preds)
            all_timestep_preds.append(torch.stack(current_timestep_preds, dim=1))
        result =  torch.cat(all_timestep_preds, dim=0)
        return result.unsqueeze(0)

    def nll(self, preds, target, masks):
        if self.nll_loss_type == 'crossent':
            return self.nll_crossent(preds, target, masks)
        elif self.nll_loss_type == 'gaussian':
            return self.nll_gaussian(preds, target, masks)
        elif self.nll_loss_type == 'poisson':
            return self.nll_poisson(preds, target, masks)

    def nll_gaussian(self, preds, target, masks, add_const=False):
        neg_log_p = ((preds - target) ** 2 / (2 * self.prior_variance))*masks.unsqueeze(-1)
        const = 0.5 * np.log(2 * np.pi * self.prior_variance)
        #neg_log_p += const
        if self.normalize_nll_per_var:
            raise NotImplementedError()
        elif self.normalize_nll:
            return (neg_log_p.sum(-1) + const*masks).view(preds.size(0), -1).sum(dim=-1)/(masks.view(masks.size(0), -1).sum(dim=1)+1e-8)
        else:
            raise NotImplementedError()


    def nll_crossent(self, preds, target, masks):
        if self.normalize_nll:
            loss = nn.BCEWithLogitsLoss(reduction='none')(preds, target)
            return (loss*masks.unsqueeze(-1)).view(preds.size(0), -1).sum(dim=-1)/(masks.view(masks.size(0), -1).sum(dim=1))
        else:
            raise NotImplementedError()

    def nll_poisson(self, preds, target, masks):
        if self.normalize_nll:
            loss = nn.PoissonNLLLoss(reduction='none')(preds, target)
            return (loss*masks.unsqueeze(-1)).view(preds.size(0), -1).sum(dim=-1)/(masks.view(masks.size(0), -1).sum(dim=1))
        else:
            raise NotImplementedError()

    def kl_categorical_learned(self, preds, prior_logits):
        log_prior = nn.LogSoftmax(dim=-1)(prior_logits)
        kl_div = preds*(torch.log(preds + 1e-16) - log_prior)
        if self.normalize_kl:     
            return kl_div.sum(-1).view(preds.size(0), -1).mean(dim=1)
        elif self.normalize_kl_per_var:
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class DNRI_DynamicVars_Encoder(nn.Module):
    def __init__(self, params):
        super(DNRI_DynamicVars_Encoder, self).__init__()
        self.num_edges = params['num_edge_types']
        self.hidden_size = hidden_size = params['encoder_hidden']
        self.num_heads = params.get('num_heads', 8)
        self.num_layers = params.get('num_layers', 6)
        self.dropout = params.get('encoder_dropout', 0.1)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=self.num_heads,
            dim_feedforward=params.get('ffn_dim', 2048),
            dropout=self.dropout,
            activation='relu'
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        self.encoder_fc_out = nn.Linear(hidden_size, self.num_edges)
        self.prior_fc_out = nn.Linear(hidden_size, self.num_edges)

    def forward(self, inputs, node_masks, all_node_inds, all_graph_info, normalized_inputs=None):
        if self.normalize_mode == 'normalize_all':
            if normalized_inputs is not None:
                x = torch.cat(normalized_inputs, dim=0)
            else:
                x = torch.cat(self.normalize_inputs(inputs, node_masks), dim=0)
        else:
            raise NotImplementedError

        x = x.transpose(0, 1)  # [seq_len, batch_size, feature_size] para Transformer
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # Volver a [batch_size, seq_len, feature_size]

        encoder_result = self.encoder_fc_out(x)
        prior_result = self.prior_fc_out(x)
        return prior_result, encoder_result, None

class DNRI_DynamicVars_Decoder(nn.Module):
    def __init__(self, params):
        super(DNRI_DynamicVars_Decoder, self).__init__()
        input_size = params['input_size']
        self.hidden_size = hidden_size = params['decoder_hidden']
        self.num_heads = params.get('num_heads', 8)
        self.num_layers = params.get('num_layers', 6)
        self.dropout = params.get('decoder_dropout', 0.1)

        # Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=self.num_heads,
            dim_feedforward=params.get('ffn_dim', 2048),
            dropout=self.dropout,
            activation='relu'
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.num_layers
        )

        self.out_fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, inputs, hidden, edges, node_masks, graph_info):
        # Preparar el input para el Transformer
        inputs = inputs.transpose(0, 1)  # [seq_len, batch_size, feature_size]
        
        if hidden is not None:
            memory = hidden.transpose(0, 1)  # [seq_len, batch_size, feature_size]
        else:
            memory = torch.zeros_like(inputs)

        # Transformer forward
        outputs = self.transformer_decoder(inputs, memory)
        outputs = outputs.transpose(0, 1)  # Volver a [batch_size, seq_len, feature_size]
        
        preds = self.out_fc(outputs)
        return preds, outputs
