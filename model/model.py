import torch
from torch import nn
from .components import *


class RecurrentModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, lstm_hidden_size):
        super(RecurrentModel, self).__init__()
        self.embdu = Embed(num_embeddings, embedding_dim, lstm_hidden_size)
        self.embdv = Embed(num_embeddings, embedding_dim, lstm_hidden_size)
        self.co_attention = CoAttention(lstm_hidden_size * 2)
        self.attend = Attend()
        self.poolu = Pool(lstm_hidden_size * 2, lstm_hidden_size)
        self.poolv = Pool(lstm_hidden_size * 2, lstm_hidden_size)
        self.classify = Classify(lstm_hidden_size * 2, 2)

    def forward(self, u, v, u_length, v_length, u_mask, v_mask):
        u = self.embdu(u, u_length)
        v = self.embdv(v, v_length)
        A = self.co_attention(u, v)
        ut, vt = self.attend(A, u, v)
        u_star = self.poolu(u, ut, u_mask, u_length)
        v_star = self.poolv(v, vt, v_mask, v_length)
        out = self.classify(u_star, v_star)
        return out
