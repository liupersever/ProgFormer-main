import math
import torch
from torch import nn
from functools import partial



class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


def transpose_qkv(X, num_heads):

    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):

    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class DiffAttention(nn.Module):

    def __init__(self, dropout):
        super(DiffAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        # print('*'*50)
        # print('keys',keys.shape)
        d = torch.einsum("nhm,lhm->nlh", queries, keys)
        # print('d',d.shape)
        attention_num = torch.sigmoid(torch.einsum("nhm,lhm->nlh", queries, keys))
        all_ones = torch.ones([keys.shape[0]]).to(keys.device)
        attention_normalizer = torch.einsum("nlh,l->nh", attention_num, all_ones)
        # print('atten_norm1',attention_normalizer.shape)
        attention_normalizer = attention_normalizer.unsqueeze(1).repeat(1, keys.shape[0], 1)
        # print('atten_norm2', attention_normalizer.shape)
        attention = attention_num / attention_normalizer
        # print('atten',attention.shape)

        attention = self.dropout(attention)
        attn_output = torch.einsum("nlh,lhd->nhd", attention, values)
        # print('attn_output.shape',attn_output.shape)
        # print('*' * 50)
        return attn_output


class MultiHeadAttention(nn.Module):

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = DiffAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        output = self.attention(queries, keys, values)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)



class AddNorm(nn.Module):

    def __init__(self, normalized_shape, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, num_heads,
                 dropout, use_bias=False):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.num_heads = num_heads

    def forward(self, X):
        Y = self.addnorm1(0.5 * X, 0.5 * self.attention(X, X, X))

        return Y

class ProsMMA(nn.Module):
    """model"""

    def __init__(self, in_channel, cv_kernel, num_heads, num_layers, length,token_e,channl_e, dropout, use_bias=False):
        super(ProsMMA, self).__init__()

        self.cv1 = nn.Conv1d(in_channel, cv_kernel[0], (1,), (1,), 0)
        self.cv3 = nn.Conv1d(in_channel, cv_kernel[1], (3,), (2,), 1)
        self.cv5 = nn.Conv1d(in_channel, cv_kernel[2], (5,), (3,), 2)

        self.encoder1 = nn.Sequential()
        for i in range(num_layers):
            self.encoder1.add_module("block" + str(i),
                                     EncoderBlock(cv_kernel[0], cv_kernel[0], cv_kernel[0], cv_kernel[0],
                                                  cv_kernel[0],
                                                  num_heads, dropout, use_bias))
        self.encoder3 = nn.Sequential()
        for i in range(num_layers):
            self.encoder3.add_module("block" + str(i),
                                     EncoderBlock(cv_kernel[1], cv_kernel[1], cv_kernel[1], cv_kernel[1],
                                                  cv_kernel[1],
                                                  num_heads, dropout, use_bias))
        self.encoder5 = nn.Sequential()
        for i in range(num_layers):
            self.encoder5.add_module("block" + str(i),
                                     EncoderBlock(cv_kernel[2], cv_kernel[2], cv_kernel[2], cv_kernel[2],
                                                  cv_kernel[2],
                                                  num_heads, dropout, use_bias))


        self.chan_first, self.chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        self.mlp_mixer1 = nn.Sequential(
            PreNormResidual(cv_kernel[0], FeedForward(length, 2, dropout, self.chan_first)),
            PreNormResidual(cv_kernel[0], FeedForward(cv_kernel[0], 2, dropout, self.chan_last)))
        self.mlp_mixer3 = nn.Sequential(
            PreNormResidual(cv_kernel[1], FeedForward(math.ceil(length / 2), 2, dropout, self.chan_first)),
            PreNormResidual(cv_kernel[1], FeedForward(cv_kernel[1], 2, dropout, self.chan_last)))
        self.mlp_mixer5 = nn.Sequential(
            PreNormResidual(cv_kernel[2], FeedForward(math.ceil(length / 3), token_e, dropout, self.chan_first)),
            PreNormResidual(cv_kernel[2], FeedForward(cv_kernel[2], channl_e, dropout, self.chan_last)))

        self.linear1 = nn.Linear(cv_kernel[0] * length, 128)
        self.linear3 = nn.Linear(cv_kernel[1] * math.ceil(length / 2), 128)
        self.linear5 = nn.Linear(cv_kernel[2] * math.ceil(length / 3), 128)

        self.fc = nn.Linear(128, 2)

    def forward(self, X):
        X1 = self.cv1(X).permute(0, 2, 1)
        X3 = self.cv3(X).permute(0, 2, 1)
        X5 = self.cv5(X).permute(0, 2, 1)

        X1 = self.mlp_mixer1(X1)
        X3 = self.mlp_mixer3(X3)
        X5 = self.mlp_mixer5(X5)

        X1 = self.encoder1(X1).view(X.shape[0], -1)
        X3 = self.encoder3(X3).view(X.shape[0], -1)
        X5 = self.encoder5(X5).view(X.shape[0], -1)

        X1 = self.linear1(X1)
        X3 = self.linear3(X3)
        X5 = self.linear5(X5)

        output = X1 + X3 + X5
        output = self.fc(output)

        return output


if __name__ == '__main__':
    netG = ProsMMA(1, [128, 64, 32], 4, 1, 30,4,2, 0.1).cuda()

    data = torch.normal(0, 1, (128, 1, 30)).cuda()
    logits = netG(data)

    print(logits.shape)
