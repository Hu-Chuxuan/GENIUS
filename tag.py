import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from math import ceil
from torch_geometric.nn import dense_diff_pool
from torch_geometric.nn.dense import DenseGraphConv

def sim_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])

        e = Wh1 + Wh2.T

        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class TAG(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, teams, max_len):
        super().__init__()
        
        self.conv1 = DenseGraphConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv3 = DenseGraphConv(hidden_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        self.lin1 = torch.nn.Linear(2 * hidden_channels + out_channels,
                                    out_channels)

        self.attentions = [nn.Parameter(torch.empty(size=(max_len,))) for _ in range(len(teams))]
        for i, attention in enumerate(self.attentions):
            self.attentions[i] = nn.init.xavier_uniform_(attention.data, gain=1.414)

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, teams, sample_teams, mask=None):
        x0 = x
        x1 = self.bn(1, self.conv1(x0, adj, mask).relu())
        x2 = self.bn(2, self.conv2(x1, adj, mask).relu())
        x3 = self.bn(3, self.conv3(x2, adj, mask).relu())

        x = torch.cat([x1, x2, x3], dim=-1)
        appended_x = torch.cat((x[0], torch.zeros(x.shape[2]).unsqueeze(0).cuda()),dim=0)

        s = self.lin1(x).relu()

        team_embedding = []
        sample_embedding = []
        remain_embedding = []
        res = []
        for i in tqdm(range(len(teams))):
            team = appended_x[teams[i],:].T@F.softmax(self.attentions[i]).cuda()
            team_embedding.append(team)
            sample = appended_x[sample_teams[i],:].T@F.softmax(self.attentions[i][[teams[i].tolist().index(j) for j in sample_teams[i]]]).cuda()
            sample_embedding.append(sample)
            remain = appended_x[list(set(teams[i])-set(sample_teams[i])-set([15004])),:].T@F.softmax(self.attentions[i][[teams[i].tolist().index(j) for j in list(set(teams[i])-set(sample_teams[i])-set([15004]))]]).cuda()#team[list(set(teams[i])-set(sample_teams[i]))]
            remain_embedding.append(remain)
            res.append(self.attentions[i].detach().numpy())
        return s, x, team_embedding, sample_embedding, remain_embedding, res

class TeamEncoder(torch.nn.Module):
    def __init__(self, num_features, teams, max_len, max_nodes):
        super().__init__()
        num_nodes = ceil(max_nodes/20)
        self.gnn_pool = TAG(num_features, 32, num_nodes, teams, max_len)
        self.softmax1 = nn.Softmax(dim = 2)

    def forward(self, x, adj, sample_teams, sample_replaces, mask=None):
        s1, embedding, team_embedding, sample_embedding, remain_embeddings, attentions = self.gnn_pool(x, adj, sample_teams, sample_replaces, mask=None)

        s1 = self.softmax1(s1)
        tmp = s1[0] @ s1[0].T
        tmp = tmp-adj
        norm = torch.norm(tmp, p=2)
        
        _, __, ___, e1 = dense_diff_pool(x, adj, s1, mask)

        sim_x = x
        sim_s = s1[0]
        sim_x_mat = sim_matrix(sim_x, sim_x)
        sim_s_mat = sim_matrix(sim_s, sim_s)
        sim = torch.sum(torch.diagonal(sim_matrix(sim_x_mat, sim_s_mat),0))
        sim = -sim/sim_x_mat.shape[0]

        embedding_loss = 0
        
        embedding = embedding[0]

        length = len(team_embedding)
        for i in range(length):
            replace_embedding = sample_embedding[i]
            remain_embedding = remain_embeddings[i]

            diff = replace_embedding - remain_embedding
            embedding_loss += torch.norm(diff)/length

        total_loss = 100*norm + 10*e1 + 100*sim + embedding_loss

        return s1, norm, e1, sim, total_loss, embedding_loss, embedding, attentions