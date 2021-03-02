import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np


class criticAttention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size):
        super(criticAttention, self).__init__()

        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size), requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size), requires_grad=True))

    def forward(self, encoder_hidden, decoder_hidden):

        batch_size, hidden_size, _ = encoder_hidden.size()

        hidden = decoder_hidden.unsqueeze(2).expand_as(encoder_hidden)
        hidden = torch.cat((encoder_hidden, hidden), 1)

        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        logit = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        logit = torch.softmax(logit, dim=2)
        return logit 


class Critic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, hidden_size, with_label, share_RNN, encoder_type='rn2', num_layers=1, n_process_blocks=3, dropout=0.2):
        super(Critic, self).__init__()
        if encoder_type == 'rnn':
            Encoder = RnnEncoder
            self.encoder = Encoder(1, hidden_size, with_label, share_RNN, num_layers, dropout)
            self.decoder = Encoder(1, hidden_size, with_label, share_RNN, num_layers, dropout)
        elif encoder_type == 'rn2':
            Encoder = RnnEncoder_2
            self.encoder = Encoder(2, hidden_size, num_layers, dropout)
            self.decoder = Encoder(2, hidden_size, num_layers, dropout)
        elif encoder_type == 'rn1':
            Encoder = RnnEncoder_2
            self.encoder = Encoder(2, hidden_size, num_layers, dropout)
            self.decoder = self.encoder
        else:
            Encoder = ConvEncoder
            self.encoder = Encoder(2, hidden_size, with_label, num_layers, dropout)
            self.decoder = Encoder(2, hidden_size, with_label, num_layers, dropout)

        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)

        self.attention = criticAttention(hidden_size)

        self.n_process_blocks = n_process_blocks

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.num_layers = num_layers
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, encoder_input, encoder_label, decoder_input, decoder_label, last_hh=None):

        # Use the probabilities of visiting each
        encoder_hidden = self.encoder(encoder_input, encoder_label)
        decoder_hidden = self.decoder(decoder_input, decoder_label)

        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)
        rnn_out = rnn_out.squeeze(1)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh) 

        for i in range(self.n_process_blocks):
            prob = self.attention(encoder_hidden, rnn_out)
            # Given a summary of the output, find an  input context
            context = prob.bmm(encoder_hidden.permute(0, 2, 1))
            # Calculate the next output using Batch-matrix-multiply ops
            rnn_out = context.squeeze(1)

        output = self.fc(rnn_out)
        return output


class RnnEncoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size, with_label, share_RNN, num_layers, dropout):
        super(RnnEncoder, self).__init__()

        if with_label:
            hidden_size = int(hidden_size / 2)
    
        self.gru_data = nn.GRU( input_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.gru_label = nn.GRU( input_size, hidden_size, num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0)
        self.drop_hh_data = nn.Dropout(p=dropout)
        self.drop_hh_label = nn.Dropout(p=dropout)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.with_label = with_label
        self.share_RNN = share_RNN

    def embedding(self, data, is_label=False):
        # encoder_input  batch_size x data_num x dim_num
        batch_size = data.shape[0]
        data_num = data.shape[1]
        dim_num = data.shape[2]

        if is_label:
            gru = self.gru_label
            drop_hh = self.drop_hh_label
        else:
            gru = self.gru_data
            drop_hh = self.drop_hh_data

        if data.is_cuda:
            output = torch.zeros( batch_size, self.hidden_size, dim_num ).cuda()
        else:
            output = torch.zeros( batch_size, self.hidden_size, dim_num )

        
        for dim_index in range(dim_num):            
            # dim_input  batch_size x data_num x input_size(1)
            dim_input = data[:,:,dim_index:dim_index+1]
            last_hh = None
            rnn_out, last_hh = gru(dim_input, last_hh)
            
            if self.num_layers == 1:
                # If > 1 layer dropout is already applied
                last_hh = drop_hh(last_hh)
            output[:,:,dim_index] = last_hh
            
        # output  batch_size x hidden_size x dim_num
        return output

    def forward(self, encoder_data, encoder_label):
        # batch_size x data_num x dim_num
        # output  batch_size x (hidden_size*2) x dim_num
        if self.with_label:
            if self.share_RNN:
                output_data = self.embedding(encoder_data, False)
                output_label = self.embedding(encoder_label, False)
            else:
                output_data = self.embedding(encoder_data, False)
                output_label = self.embedding(encoder_label, True)
            return torch.cat( (output_label, output_data), dim=1 )
        else:
            output_data = self.embedding(encoder_data, False)
            return output_data
        

class RnnEncoder_2(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(RnnEncoder_2, self).__init__()

        self.gru = nn.GRU( input_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        # self.gru_label = nn.GRU( input_size, hidden_size, num_layers,
        #             batch_first=True,
        #             dropout=dropout if num_layers > 1 else 0)
        self.drop_hh = nn.Dropout(p=dropout)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def embedding(self, data, label):
        # encoder_input  batch_size x data_num x dim_num
        batch_size = data.shape[0]
        data_num = data.shape[1]
        dim_num = data.shape[2]

        gru = self.gru
        drop_hh = self.drop_hh

        if data.is_cuda:
            output = torch.zeros( batch_size, self.hidden_size, dim_num ).cuda()
        else:
            output = torch.zeros( batch_size, self.hidden_size, dim_num )

        
        for dim_index in range(dim_num):            
            # dim_input  batch_size x data_num x input_size(1)
            data_input = data[:,:,dim_index:dim_index+1]
            label_input = label[:,:,dim_index:dim_index+1]
            last_hh = None
            rnn_out, last_hh = gru(  torch.cat( (label_input, data_input), dim=-1 ) , last_hh)
            
            if self.num_layers == 1:
                # If > 1 layer dropout is already applied
                last_hh = drop_hh(last_hh)
            output[:,:,dim_index] = last_hh
            
        # output  batch_size x hidden_size x dim_num
        return output

    def forward(self, encoder_data, encoder_label_code):
        # batch_size x data_num x dim_num
        # output  batch_size x (hidden_size) x dim_num
        output_data = self.embedding(encoder_data, encoder_label_code)
        return output_data
        

class ConvEncoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size, with_label, num_layers, dropout):
        super(ConvEncoder, self).__init__()
        
        if with_label:
            hidden_size = int(hidden_size / 2)
            
        self.conv_data = nn.Conv1d(input_size, int(hidden_size), kernel_size=1)
        self.conv_label = nn.Conv1d(input_size, int(hidden_size), kernel_size=1)
        self.with_label = with_label

    def forward(self, encoder_data, encoder_label):
        if self.with_label:
            output_data = self.conv_data(encoder_data)
            output_label = self.conv_data(encoder_label)
            return torch.cat( (output_label, output_data ), dim=1 )
        else:
            output_data = self.conv_data(encoder_data)
            return output_data


class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(Attention, self).__init__()

        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, decoder_hidden_size), requires_grad=True))
        self.W = nn.Parameter(torch.zeros((1, decoder_hidden_size, encoder_hidden_size + decoder_hidden_size), requires_grad=True))


    def forward(self, encoder_hidden, decoder_hidden):

        batch_size, hidden_size = decoder_hidden.size()

        decoder_hidden = decoder_hidden.unsqueeze(2).repeat(1, 1, encoder_hidden.shape[-1])

        hidden = torch.cat((encoder_hidden, decoder_hidden), 1)

        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        attns = F.softmax(attns, dim=2)  # (batch, seq_len)
        return attns


class Pointer(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, encoder_hidden_size, decoder_hidden_size, with_label, num_layers=1, dropout=0.2):
        super(Pointer, self).__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.num_layers = num_layers
        self.with_label = with_label

        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(torch.zeros((1, 1, decoder_hidden_size), requires_grad=True))
        self.W = nn.Parameter(torch.zeros((1, decoder_hidden_size, decoder_hidden_size + encoder_hidden_size), requires_grad=True))

        # Used to compute a representation of the current decoder output
        self.gru = nn.GRU( decoder_hidden_size, decoder_hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)

        self.encoder_attn = Attention( encoder_hidden_size, decoder_hidden_size)

        self.num_head = 1
        if self.num_head > 1:
            self.multi_head_encoder_attns = nn.ModuleList(
                [
                    Attention( encoder_hidden_size, decoder_hidden_size) 
                    for i in range(self.num_head)
                ]
            )
            self.multi_head_linear = nn.Linear(self.num_head, 1)
        else:
            self.encoder_attn = Attention( encoder_hidden_size, decoder_hidden_size )

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, encoder_hidden, decoder_hidden, last_hh):

        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)
        rnn_out = rnn_out.squeeze(1)

        # Always apply dropout on the RNN output
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh)

        # Given a summary of the output, find an  input context

        # enc_attn = self.encoder_attn( encoder_hidden, rnn_out)
        if self.num_head > 1:
            multi_head_enc = []
            for head_attn in self.multi_head_encoder_attns:
                # batch_size * 1 * seq_len
                multi_head_enc.append( head_attn( encoder_hidden, rnn_out ) )
            final_enc_attn = torch.cat( multi_head_enc, dim=1 )
            enc_attn = self.multi_head_linear( final_enc_attn.transpose(2,1) ).transpose(2,1)
        else:
            enc_attn = self.encoder_attn( encoder_hidden, rnn_out)
                
        context = enc_attn.bmm( encoder_hidden.permute(0, 2, 1))  # (B, 1, num_feats)

        # Calculate the next output using Batch-matrix-multiply ops
        context = context.transpose(1, 2).expand_as( encoder_hidden )
        
        energy = torch.cat(( encoder_hidden, context), dim=1)  # (B, num_feats, seq_len)

        v = self.v.expand(encoder_hidden.size(0), -1, -1)
        W = self.W.expand(encoder_hidden.size(0), -1, -1)

        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)

        return probs, last_hh


class DRL(nn.Module):
    """Defines the main Encoder, Decoder, and Pointer combinatorial models.

    Parameters
    ----------
    hidden_size: int
        Defines the number of units in the hidden layer for all static, dynamic,
        and decoder output units.
    mask_fn: function or None
        Allows us to specify which elements of the input sequence are allowed to
        be selected. This is useful for speeding up training of the networks,
        by providing a sort of 'rules' guidlines to the algorithm. If no mask
        is provided, we terminate the search after a fixed number of iterations
        to avoid tours that stretch forever
    num_layers: int
        Specifies the number of hidden layers to use in the decoder RNN
    dropout: float
        Defines the dropout rate for the decoder
    """

    def __init__(self, encoder_hidden_size, decoder_hidden_size, 
                mask_fn, 
                vis_type, reward_type, encoder_type, 
                with_label, share_RNN,
                use_cuda,
                num_layers, dropout):
        super(DRL, self).__init__()

        self.mask_fn = mask_fn
        # Define the encoder & decoder models
        if encoder_type == 'rnn':
            Encoder = RnnEncoder
            self.encoder = Encoder(1, encoder_hidden_size, with_label, share_RNN, num_layers, dropout)
            self.decoder = Encoder(1, decoder_hidden_size, with_label, share_RNN, num_layers, dropout)
        elif encoder_type == 'rn2':
            Encoder = RnnEncoder_2
            self.encoder = Encoder(2, encoder_hidden_size, num_layers, dropout)
            self.decoder = Encoder(2, decoder_hidden_size, num_layers, dropout)
        elif encoder_type == 'rn1':
            Encoder = RnnEncoder_2
            self.encoder = Encoder(2, encoder_hidden_size, num_layers, dropout)
            self.decoder = self.encoder
        else:
            Encoder = ConvEncoder
            self.encoder = Encoder(2, encoder_hidden_size, with_label, num_layers, dropout)
            self.decoder = Encoder(2, decoder_hidden_size, with_label, num_layers, dropout)

        self.pointer = Pointer(encoder_hidden_size, decoder_hidden_size, with_label, num_layers, dropout)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.use_cuda = use_cuda
        self.vis_type = vis_type
        self.reward_type = reward_type
        self.with_label = with_label

    def forward(self, encoder_input, encoder_label_code, decoder_input, decoder_label_code, last_hh=None):
        """
        Parameters
        ----------
        encoder_input: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates, which won't change
        decoder_input: Array of size (batch_size, num_feats)
            Defines the outputs for the decoder. Currently, we just use the
            static elements (e.g. (x, y) coordinates), but this can technically
            be other things as well
        last_hh: Array of size (batch_size, num_hidden)
            Defines the last hidden state for the RNN
        """
        batch_size, data_num, dim_num = encoder_input.size()

        # Always use a mask - if no function is provided, we don't update it
        mask = torch.ones(batch_size, dim_num)
        if self.use_cuda:
            mask = mask.cuda()

        # Structures for holding the output sequences
        dim_idx, dim_logp = [], []
        max_steps = dim_num if self.mask_fn is None else 1000

        # t1 = time.time()

        encoder_hidden = self.encoder(encoder_input, encoder_label_code)

        # my_probs = []
        # my_p = []

        for _ in range(max_steps):

            if not mask.byte().any():
                break

            decoder_hidden = self.decoder(decoder_input, decoder_label_code)

            probs, last_hh = self.pointer(encoder_hidden, decoder_hidden, last_hh)
            # my_p.append(probs[0].cpu().numpy())
            
            probs = F.softmax(probs + mask.log(), dim=1)

            # my_probs.append(probs[0].cpu().numpy())
            
            # When training, sample the next step according to its probability.
            # During testing, we can take the greedy approach and choose highest
            if self.training:
                m = torch.distributions.Categorical(probs)
                ptr = m.sample()
                while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
                    ptr = m.sample()
                logp = m.log_prob(ptr)
            else:
                prob, ptr = torch.max(probs, 1)  # Greedy
                logp = prob.log()

            # And update the mask so we don't re-visit if we don't need to
            if self.mask_fn is not None:
                mask = self.mask_fn(mask, ptr.data)
                mask = mask.detach()


            decoder_input = torch.gather(encoder_input, 2,
                            ptr.view(-1, 1, 1)
                            .expand(-1, data_num, 1)).detach() # TODO
            batch_size, label_code_n, dim_num = encoder_label_code.shape
            decoder_label_code = torch.gather(encoder_label_code, 2,
                            ptr.view(-1, 1, 1)
                            .expand(batch_size, label_code_n, 1)).detach() # TODO
            dim_logp.append(logp.unsqueeze(1))
            dim_idx.append(ptr.data.unsqueeze(1))


        dim_idx = torch.cat(dim_idx, dim=1)  # (batch_size, seq_len)
        dim_logp = torch.cat(dim_logp, dim=1)  # (batch_size, seq_len)

        # np.savetxt('./bk/probs.txt', my_probs)
        # np.savetxt('./bk/p.txt', my_p)

        return dim_idx, dim_logp


if __name__ == '__main__':
    raise Exception('Cannot be called from main')
