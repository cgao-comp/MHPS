import os
import pickle

import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import RGCNConv, GCNConv, HypergraphConv

from utils.Constants import *
from utils.GraphBuilder import LoadHeteStaticGraph, LoadHyperGraph
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv
class Fusion(nn.Module):
    def __init__(self, input_size, out=1, dropout=0.2):
        super(Fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, out)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden, dy_emb):
        '''
        hidden: 这个子超图HGAT的输入，dy_emb: 这个子超图HGAT的输出
        hidden和dy_emb都是用户embedding矩阵，大小为(用户数, 64)
        '''
        # tensor.unsqueeze(dim) 扩展维度，返回一个新的向量，对输入的既定位置插入维度1
        # tensor.cat(inputs, dim=?) --> Tensor    inputs：待连接的张量序列     dim：选择的扩维，沿着此维连接张量序列
        emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)
        emb_score = nn.functional.softmax(self.linear2(torch.nn.functional.gelu(self.linear1(emb))), dim=0)
        #emb_score = self.dropout(emb_score)  # 随机丢弃每个用户embedding的权重
        out = torch.sum(emb_score * emb, dim=0)  # 将输入的embedding和输出的embedding按照对应的用户加权求和
        return out

# basic class for graph embedding generater
class GraphNN(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.1):
        super(GraphNN, self).__init__()
        self.embedding = nn.Embedding(ntoken, ninp, padding_idx=0)
        self.ntoken = ntoken
        #GCNConv()
        self.ninp = ninp

        self.hypergcn = HypergraphConv(in_channels=self.ninp, out_channels=self.ninp)
        self.global_hypergcn = HypergraphConv(in_channels=self.ninp, out_channels=self.ninp)
        ### attention parameters
        self.att = nn.Parameter(torch.zeros(1, self.ninp))
        self.att_m = nn.Parameter(torch.zeros(self.ninp, self.ninp))
        self.dropout = nn.Dropout(dropout)

        ### channel self-gating parameters
        self.n_channel = 3
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.ninp, self.ninp)) for _ in range(self.n_channel)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.ninp)) for _ in range(self.n_channel)])
        self.fus = Fusion(self.ninp)

        self.Graph = None
        self.gamma = 0.1
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.embedding.weight)
        init.xavier_normal_(self.weights[0])
        init.xavier_normal_(self.weights[1])
        init.xavier_normal_(self.att)
        init.xavier_normal_(self.att_m)

    def self_gating(self, em, channel):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.weights[channel]) + self.bias[channel]))

    def get_normalize_graph(self, index):
        data = torch.ones(index.size(-1)).int()
        Graph = torch.sparse.IntTensor(index, data,
                                            torch.Size([self.ntoken, self.ntoken]))
        dense = Graph.to_dense()
        D = torch.sum(dense, dim=1).float()
        D[D == 0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense / D_sqrt
        dense = dense / D_sqrt.t()
        index = dense.nonzero()
        data = dense[dense >= 1e-9]
        assert len(index) == len(data)
        Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.ntoken, self.ntoken]))
        Graph = Graph.coalesce().cuda(3)
        return Graph


    def lightgcn(self, all_emb):

        """
        propagate methods for lightGCN
        """
        embs = [all_emb]
        for layer in range(6):
            norm = torch.norm(all_emb, dim=1) + 1e-12
            all_emb = all_emb / norm[:, None]
            all_emb = torch.mm(self.Graph.to_dense(), all_emb)
            embs.append(all_emb)

        embs_zero = embs[0]
        embs_prop = torch.mean(torch.stack(embs[1:], dim=1), dim=1)
        light_out = (self.gamma * embs_zero) + ((1 - self.gamma) * embs_prop)
        return light_out


    def diff_loss(self, shared_embedding, task_embedding):
        shared_embedding = shared_embedding -  torch.mean(shared_embedding, 0)
        task_embedding = task_embedding - torch.mean(task_embedding, 0)

        # p=2时是l2正则
        shared_embedding = nn.functional.normalize(shared_embedding, dim=1, p=2)
        task_embedding = nn.functional.normalize(task_embedding, dim=1, p=2)

        correlation_matrix = task_embedding.t() @ shared_embedding
        loss_diff = torch.mean(torch.square_(correlation_matrix)) * 0.01
        loss_diff = torch.where(loss_diff > 0, loss_diff, 0)
        return loss_diff

    def forward(self, heter_graph, hyper_graph, hyper_graph_micro):
        heter_graph_edge_index = heter_graph.edge_index

        #u_emb = self.user_map(self.embedding.weight)
        u_emb = self.self_gating(self.embedding.weight, 0)
        c_emb = self.self_gating(self.embedding.weight, 1)

        if self.Graph is None:
            self.Graph = self.get_normalize_graph(heter_graph_edge_index)
        u_emb = self.lightgcn(u_emb)

        hg_embeddings = []
        for i in range(len(hyper_graph)):
            subhg_embedding = self.hypergcn(c_emb, hyper_graph[i].edge_index.cuda(3))
            if i == 0:
                hg_embeddings.append(subhg_embedding)
            else:
                subhg_embedding = self.fus(hg_embeddings[-1], subhg_embedding)
                hg_embeddings.append(subhg_embedding)

        h_emb = hg_embeddings[-1]
        # 微观超图
        hg_embeddings_micro = []
        for i in range(len(hyper_graph_micro)):
            subhg_embedding = self.hypergcn(c_emb, hyper_graph_micro[i].edge_index.cuda(3))
            if i == 0:
                hg_embeddings_micro.append(subhg_embedding)
            else:
                subhg_embedding = self.fus(hg_embeddings_micro[-1], subhg_embedding)
                hg_embeddings_micro.append(subhg_embedding)

        h_emb_micro = hg_embeddings_micro[-1]
 
        high_embs, attention_score = self.channel_attention(u_emb, h_emb, h_emb_micro)

        return high_embs, 0.0

    def channel_attention(self, *channel_embeddings):
        weights = []

        for embedding in channel_embeddings:
            weights.append(
                torch.sum(torch.multiply(self.att, torch.matmul(embedding, self.att_m)), 1))
        embs = torch.stack(weights, dim=0)
        score = F.softmax(embs.t(), dim=-1)
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += torch.multiply(score.t()[i], channel_embeddings[i].t()).t()
        return mixed_embeddings, score


class GraphEncoder(nn.Module):
    def __init__(self, opt, Type, hypergraph_list, hypergraph_list_micro, dropout=0.15):
        super(GraphEncoder, self).__init__()
        self.dropedge=opt.dropout

        self.ntoken = opt.ntoken
        self.user_size = opt.user_size
        self.ninp = opt.d_word_vec
        self.output_dim = opt.d_word_vec

        self.heter_graph = LoadHeteStaticGraph(opt.data_path,Type)
        self.hyper_graph = hypergraph_list#LoadHyperGraph(opt.data_path,Type)
        self.hyper_graph_micro = hypergraph_list_micro
 
        self.global_hyper_graph = LoadHyperGraph(opt.data_path,Type)
        self.gnn_layer = GraphNN(self.ntoken, self.ninp)
        #self.hyper_layer = HyperGraphNN(self.ntoken, self.ninp)

    def forward(self,input, input_timestamp,train=True):

        batch_size, max_len = input.size()
        user_embedding_lookup, diff_loss = self.gnn_layer(self.heter_graph, self.hyper_graph, self.hyper_graph_micro)
        #user_hyper_embedding_lookup = self.hyper_layer(self.hyper_graph).cuda(3)  # [user_size, user_embedding]

        user_input=input.contiguous().view(batch_size*max_len,1).cuda(3)
        user_social_embedding_one_hot=torch.zeros(batch_size*max_len, self.ntoken).cuda(3)
        user_social_embedding_one_hot=user_social_embedding_one_hot.scatter_(1, user_input, 1)

        user_embedding=torch.einsum("bt,td->bd",user_social_embedding_one_hot,user_embedding_lookup).view(batch_size,max_len,
                                                                                                          self.ninp).cuda(3)
        return user_embedding.cuda(3)


