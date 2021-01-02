import torch
from torch import nn
from torch.nn.parameter import Parameter


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2):
        super(BiLSTM, self).__init__()
        self.BiLSTM = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                              batch_first=True, dropout=dropout, bidirectional=True)

    def forward(self, x, x_length):
        # x is a padded sequence with shape (batch_size, seq_len, input_size)
        x = nn.utils.rnn.pack_padded_sequence(x, x_length, batch_first=True, enforce_sorted=False)
        x, _ = self.BiLSTM(x)
        x = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)[0]
        # x is now a tensor with shape (batch_size, seq_len, 2 * hidden_size)
        return x


class Embed(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size):
        super(Embed, self).__init__()
        self.embd = nn.Embedding(num_embeddings, embedding_dim)
        self.BiLSTM = BiLSTM(embedding_dim, hidden_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, u, u_length):
        out = self.embd(u)
        self.dropout(out)
        out = self.BiLSTM(out, u_length)
        self.dropout(out)
        return out


class CoAttention(nn.Module):
    def __init__(self, vec_len):
        super(CoAttention, self).__init__()
        self.w1 = Parameter(torch.randn((vec_len), requires_grad=True))
        self.w2 = Parameter(torch.randn((vec_len), requires_grad=True))
        self.w3 = Parameter(torch.randn((vec_len), requires_grad=True))

    def forward(self, u, v):
        (batch_size, seq_lenu, vec_len) = u.shape
        (batch_size, seq_lenv, vec_len) = v.shape
        u = u.reshape((batch_size, seq_lenu, 1, vec_len))
        v = v.reshape((batch_size, 1, seq_lenv, vec_len))
        # using pytorch's broadcast machanisim
        w1u = torch.matmul(u, self.w1)
        w2v = torch.matmul(v, self.w2)
        w3uv = torch.matmul(u * v, self.w3)
        A = w1u + w2v + w3uv
        # A is shape (batch_size, seq_lenu, seq_lenv)
        return A


class Attend(nn.Module):
    def __init__(self):
        super(Attend, self).__init__()
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)

    def forward(self, A, u, v):
        up = self.softmax1(A)
        up = up.transpose(1, 2)
        vp = self.softmax2(A)
        vt = torch.matmul(up, u)
        ut = torch.matmul(vp, v)
        return ut, vt


class Pool(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Pool, self).__init__()
        self.BiLSTMu = BiLSTM(input_size * 3, hidden_size)
        self.wu = Parameter(torch.randn((2 * hidden_size), requires_grad=True))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, u, ut, u_mask, u_length):
        # Prepare
        uut = u * ut
        u_in = torch.cat((u, ut, uut), dim=2)
        u_mask = u_mask.unsqueeze(2)
        u_in = u_in * u_mask

        # BiLSTM
        u_out = self.BiLSTMu(u_in, u_length)

        # Attention
        au = torch.matmul(u_out, self.wu)
        pu = self.softmax(au)
        u_out = u_out.transpose(1, 2)
        pu = pu.unsqueeze(2)
        u_star = torch.matmul(u_out, pu)
        u_star = u_star.squeeze()
        return u_star


class Classify(nn.Module):
    def __init__(self, in_features, class_num):
        super(Classify, self).__init__()
        self.fc1 = nn.Linear(in_features * 2, in_features)
        self.fc2 = nn.Linear(in_features, class_num)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, u_star, v_star):
        fc_in = torch.cat((u_star, v_star), dim=1)
        fc_in = self.dropout(fc_in)
        out = self.fc1(fc_in)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out
