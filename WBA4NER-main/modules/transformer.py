

import torch
import torch.nn.functional as F

from torch import nn
import math
from copy import deepcopy

from .relative_transformer import RelativeMultiHeadAttn
import numpy as np
import fitlog
use_fitlog = True
if not use_fitlog:
    fitlog.debug()
fitlog.set_log_dir('logs')


class MultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, scale=False,vocab=None):
        """

        :param d_model:
        :param n_head:
        :param scale: 是否scale输出
        """
        super().__init__()
        assert d_model%n_head==0

        self.n_head = n_head
        self.qkv_linear = nn.Linear(d_model, 3*d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)

        if scale:
            self.scale = math.sqrt(d_model//n_head)
        else:
            self.scale = 1
        self.vocab = vocab
        self.count=0
        self.times = 0

    def forward(self, x,char_ids,seq_len, mask):
        """

        :param x: bsz x max_len x d_model
        :param mask: bsz x max_len
        :return:
        """
        batch_size, max_len, d_model = x.size()
        x = self.qkv_linear(x)
        q, k, v = torch.chunk(x, 3, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        attn = torch.matmul(q, k)  # batch_size x n_head x max_len x max_len
        attn = attn/self.scale
        attn.masked_fill_(mask=mask[:, None, None].eq(0), value=float('-inf'))

        attn = F.softmax(attn, dim=-1)  # batch_size x n_head x max_len x max_len
        attn = self.dropout_layer(attn)

        if self.training:
            self.times += 1
        # if self.times == 1:
        #     print(att.size())
        if self.times % 1350 == 0:
            self.count+=1
            # print(att[1,1,:])
            # print(att.size())
            # print(att[1])
            # print(x_chars[1])
            # print(x_words[1])
            # print(mask[1][0])
            # print(att_score[1])

            # hot map
            def attention_visualization(head_i,count):
                import matplotlib.pyplot as plt
                import seaborn as sns
                sns.set(font="SIMHEI")
                # from sklearn.preprocessing import normalize
                index_ = 2
                seq_length = seq_len[index_].item()
                #lex_length = lex_num[index_].item()
                print("第%d个头注意力可视化" % head_i)
                data = attn.cpu().detach().numpy()[index_][head_i][:seq_length, :seq_length].T
                #fitlog.add_hyper_in_file(data)
                np.savetxt("./plt/tfile.log", data)
                plt.figure(figsize=(seq_length // 4 + 2, seq_length // 4 + 2))
                plt.rcParams['font.sans-serif'] = ['SIMHEI']  # 设置字体为黑体
                plt.rcParams['axes.unicode_minus']=False 
                #plt.xticks(fontsize=5)
                
                if not char_ids is None :
                    row_chars = char_ids[index_][:seq_length].cpu().numpy().tolist()
                    row_words = char_ids[index_][:seq_length].cpu().numpy().tolist()
                    row_chars = [self.vocab.to_word(x) for x in row_chars]
                    row_words = [self.vocab.to_word(x) for x in row_words]
                    sns.heatmap(data,
                                xticklabels=row_chars,
                                yticklabels=row_words,
                                cbar_kws={"orientation": "horizontal"})
                    plt.yticks(rotation=0)
                    plt.savefig('./plt/bt{}a{}.png'.format(count,head_i))
                    plt.show()
                
            # output all head
            for i in range(0, self.n_head):
                
                attention_visualization(i,self.count)
            # output single head
            # attention_visualization(random.randint(0, self.n_head - 1))

        
        v = torch.matmul(attn, v)  # batch_size x n_head x max_len x d_model//n_head
        v = v.transpose(1, 2).reshape(batch_size, max_len, -1)
        v = self.fc(v)

        return v


class TransformerLayer(nn.Module):
    def __init__(self, d_model, self_attn, feedforward_dim, after_norm, dropout, vocab):
        """

        :param int d_model: 一般512之类的
        :param self_attn: self attention模块，输入为x:batch_size x max_len x d_model, mask:batch_size x max_len, 输出为
            batch_size x max_len x d_model
        :param int feedforward_dim: FFN中间层的dimension的大小
        :param bool after_norm: norm的位置不一样，如果为False，则embedding可以直接连到输出
        :param float dropout: 一共三个位置的dropout的大小
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = self_attn

        self.after_norm = after_norm

        self.ffn = nn.Sequential(nn.Linear(d_model, feedforward_dim),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(feedforward_dim, d_model),
                                 nn.Dropout(dropout))

    def forward(self, x,char_ids, seq_len, mask):
        """

        :param x: batch_size x max_len x hidden_size
        :param mask: batch_size x max_len, 为0的地方为pad
        :return: batch_size x max_len x hidden_size
        """
        residual = x
        if not self.after_norm:
            x = self.norm1(x)

        x = self.self_attn(x,char_ids, seq_len, mask)
        x = x + residual
        if self.after_norm:
            x = self.norm1(x)
        residual = x
        if not self.after_norm:
            x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        if self.after_norm:
            x = self.norm2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_head, feedforward_dim, dropout, after_norm=True, attn_type='naive',
                 scale=False, attn_dropout=0, pos_embed=None, vocab=None):
        super().__init__()
        self.d_model = d_model

        if pos_embed is None:
            self.pos_embed = None
        elif pos_embed == 'sin':
            self.pos_embed = SinusoidalPositionalEmbedding(d_model, 0, init_size=1024)
        elif pos_embed == 'fix':
            self.pos_embed = LearnedPositionalEmbedding(1024, d_model, 0)

        if attn_type == 'transformer':
            self_attn = MultiHeadAttn(d_model, n_head, attn_dropout, scale=scale,vocab=vocab)
        elif attn_type == 'adatrans':
            self_attn = RelativeMultiHeadAttn(d_model, n_head, attn_dropout, scale=scale, vocab=vocab)

        self.layers = nn.ModuleList([TransformerLayer(d_model, deepcopy(self_attn), feedforward_dim, after_norm, dropout, vocab=vocab)
                       for _ in range(num_layers)])

    def forward(self, x, char_ids, seq_len,mask):
        """

        :param x: batch_size x max_len
        :param mask: batch_size x max_len. 有value的地方为1
        :return:
        """
        if self.pos_embed is not None:
            x = x + self.pos_embed(mask)

        for layer in self.layers:
            x = layer(x,char_ids, seq_len, mask)
        return x


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
        torch.cumsum(mask, dim=1).type_as(mask) * mask
    ).long() + padding_idx


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        positions = make_positions(input, self.padding_idx)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)

    def forward(self, input):
        # positions: batch_size x max_len, 把words的index输入就好了
        positions = make_positions(input, self.padding_idx)
        return super().forward(positions)
