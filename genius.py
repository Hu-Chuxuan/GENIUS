import torch
import itertools
import torch.nn as nn

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

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        
        self.conv1 = DenseGraphConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseGraphConv(hidden_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)
        self.lin1 = torch.nn.Linear(2 * hidden_channels + out_channels, out_channels)
        self.lin2 = torch.nn.Linear(2 * hidden_channels + out_channels, 1)

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)

        return x

    def forward(self, x, adj, mask=None):
        x0 = x
        x1 = self.bn(1, self.conv1(x0, adj, mask).relu())
        x2 = self.bn(2, self.conv2(x1, adj, mask).relu())
        x3 = self.bn(3, self.conv3(x2, adj, mask).relu())

        x = torch.cat([x1, x2, x3], dim=-1)

        s = self.lin1(x).relu()
        val = self.lin2(x).relu()

        return s, x, val

class TeamEncoder(torch.nn.Module):
    def __init__(self, num_features, max_nodes):
        super().__init__()
        num_nodes = ceil(max_nodes/20)
        self.gnn1_pool = GNN(num_features, 32, num_nodes)
        self.softmax1 = nn.Softmax(dim = 2)
        

    def forward(self, x, adj, sample_teams, sample_replaces, mask=None):
        s1, embedding = self.gnn1_pool(x, adj, mask)

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

        length = len(sample_teams)
        for i in range(length):
            team = sample_teams[i]
            replace = sample_replaces[i]

            replace_embedding = torch.sum(embedding[replace,:], dim = 0)/len(replace)
            remain = list(set(team).difference(set(replace)))
            remain_embedding = torch.sum(embedding[remain,:], dim = 0)/len(remain)

            diff = replace_embedding - remain_embedding
            embedding_loss += torch.norm(diff,p=1)
        
        embedding_loss = embedding_loss/length
        total_loss = 100*norm + 10*e1 + 100*sim + embedding_loss

        return s1, norm, e1, sim, total_loss, embedding_loss, embedding 

def SubteamRecommender(original_node_list, team_to_be_replaced, cluster_whole, hard_assignment, embeddings):
    
    # find the cluster candidates
    cluster_cand = []
    for i in team_to_be_replaced:
        cluster_cand.append(hard_assignment[i])
    cluster_cand = list(set(cluster_cand))
    
    clusters = []
    
    for i in cluster_cand:
        clusters.append(list(set(cluster_whole[i]).difference(set(original_node_list))))
    
    # replacing candidates
    ls = list(itertools.product(*clusters))
    
    # finding remains
    remains = list(set(original_node_list).difference(set(team_to_be_replaced)))
    
    sim_s = []
    sim_ret = 0
    ret = []
    
    for i in tqdm(range(len(ls))):

        replace = ls[i]
        replace_embedding = torch.sum(embeddings[replace,:], dim = 0)/len(replace)
        remain_embedding = torch.sum(embeddings[original_node_list,:], dim = 0)/len(original_node_list)

        sim = torch.nn.functional.cosine_similarity(remain_embedding, replace_embedding, dim=0, eps=1e-08)
        if(sim > sim_ret):
            sim_ret = sim
            team_new = remains + list(ls[i])
            ret = team_new[:]

        sim_s.append(sim)
        
    return ret, sim_ret


