import torch
import torch.nn as nn
import numpy as np
import neutic.Constants as Constants
from neutic.Layers import SelfAttention
import torch.nn.functional as F


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


class BN_ReLU(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.bn = nn.BatchNorm1d(channel, track_running_stats=False)
        self.relu = nn.ReLU()


    def forward(self, inputs):
        outputs = self.bn(inputs)
        outputs = self.relu(outputs)
        return outputs

class NeuTic(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
             n_src_vocab, h, grained,
            d_word_vec=512, d=512, d_inner=2048,
            L=2, n_head=8, d_k=64, d_v=64, dropout=0.1):

        super().__init__()

        assert d == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        n_position = h + 1

        self.src_length_emb = nn.Embedding(
            n_src_vocab[0], d_word_vec, padding_idx=Constants.PAD)
        self.src_window_emb = nn.Embedding(
            n_src_vocab[1], d_word_vec, padding_idx=Constants.PAD)
        self.src_flag_emb = nn.Embedding(
            n_src_vocab[2], d_word_vec, padding_idx=Constants.PAD)

        self.combine_fully = nn.Linear(d_word_vec*3, d_word_vec)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            SelfAttention(d, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(L)])

        self.conv_1 = nn.Conv1d(d, d, 1, stride=1, padding=(1 // 2))
        self.conv_2 = nn.Conv1d(d, d, 3, stride=1, padding=(3 // 2))
        self.conv_3 = nn.Conv1d(d, d, 5, stride=1, padding=(5 // 2))

        self.conv_gate = nn.Conv1d(d*3, d, 1, stride=1, padding=0)

        self.layer_norm = nn.LayerNorm(d)
        self.bn_norm =  BN_ReLU(d)
        

        self.dropout = nn.Dropout(p=0.3)
        self.fully_1=nn.Linear(h * d, 512)
        self.fully_2 = nn.Linear(512, grained)



    def forward(self, src_seq, src_pos):

        len_seq, wid_seq, flag_seq = src_seq

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=len_seq, seq_q=len_seq)
        non_pad_mask = get_non_pad_mask(len_seq)


        # -- Input Embedding
        len_seq = self.src_length_emb(len_seq)
        wid_seq = self.src_window_emb(wid_seq)
        flag_seq = self.src_flag_emb(flag_seq)
        cmb_seq = torch.cat((len_seq, wid_seq, flag_seq), axis=-1)
        cmb_seq = self.combine_fully(cmb_seq)
        pos_output = cmb_seq + self.position_enc(src_pos)

        inputs = torch.transpose(pos_output, 1, 2)

        # -- Multi-kernel Convolution
        branch_1 = self.conv_1(inputs)
        branch_2 = self.conv_2(inputs)
        branch_3 = self.conv_3(inputs)

        branch = torch.cat((branch_1, branch_2, branch_3), axis=1)
        branch = self.conv_gate(branch)
        branch = self.bn_norm(branch)

        att_input = self.bn_norm(branch + inputs)
        att_output = torch.transpose(att_input, 1, 2)

        # -- Sequence Self-attention
        for att_layer in self.layer_stack:

            att_output, _ = att_layer(
                att_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        # -- Classification Output
        att_output = att_output.reshape([att_output.shape[0], -1])
        out_state = self.fully_1(att_output)
        out = self.dropout(out_state)
        final_output = self.fully_2(out)
        return final_output



