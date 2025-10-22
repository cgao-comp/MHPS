import math

import torch

import utils.Constants as Constants

from model.Merger import *
from model.TransformerBlock import TransformerBlock, Long_term_atention
from model.GraphEncoder import GraphEncoder
from model.Decoder import Decoder

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
        emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)
        emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score * emb, dim=0)
        return out

class   MHPS(nn.Module):
    def __init__(self,opt, hypergraph_list, hypergraph_list_micro):
        super(MHPS, self).__init__()
        # hypers 
        self.opt=opt
        dropout=opt.dropout
        self.ntoken = opt.ntoken
        self.user_size = opt.user_size
        self.ninp = opt.d_word_vec
        self.transformer_dim = opt.transformer_dim  # 最终的维度
        self.pos_dim = opt.pos_dim
        self.__name__ = "GRU"
        #self.time_encoder = TimeEncoder(opt)
        # module control 
        self.graph_type = opt.graph_type
        self.time_encoder_type = opt.time_encoder_type

        # dropout module
        self.dropout = nn.Dropout(dropout)
        #self.drop_timestamp = nn.Dropout(dropout)

        # modules 
        self.user_encoder = GraphEncoder(opt, self.graph_type, hypergraph_list, hypergraph_list_micro)
        self.pos_dim = self.ninp
        self.linear_pos_embedding = nn.Embedding(1000, self.pos_dim)
        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / self.pos_dim) for i in range(self.pos_dim)]).cuda(3)
        #self.decoder = TransformerBlock(input_size=opt.transformer_dim, n_heads=8)

        ### attention parameters
        self.att = nn.Parameter(torch.zeros(1, self.ninp))
        self.att_m = nn.Parameter(torch.zeros(self.ninp, self.ninp))

        self.stu_ca = TransformerBlock(d_q=self.ninp, d_k = self.ninp , d_v =self.ninp, n_heads=1, is_layer_norm=True)
        self.linear_pos_ca = TransformerBlock(d_q=self.ninp, d_k=self.ninp, d_v =self.ninp, n_heads=1, is_layer_norm=True)
        self.sin_cos_pos_ca = Long_term_atention(input_size = self.ninp)#,(d_q=self.ninp, d_k=self.ninp, d_v=self.ninp, n_heads=1, is_layer_norm=True)
        #self.cas_encoder = GRAU(self.ninp, self.ninp, opt=opt)

        self.cas_encoder = nn.GRU(self.ninp, self.ninp, batch_first=True)
        self.cas_encoder2 = nn.GRU(self.ninp, self.ninp, batch_first=True)
        #self.cas_encoder = CustomGRAU(self.ninp, self.ninp)
        #self.cas_encoder = CustomGRU(self.ninp, self.ninp)
        #self.next_str = nn.GRU(self.ninp, self.ninp, batch_first=True)
        self.layernorm = nn.LayerNorm(self.ninp)
        self.layernorm2 = nn.LayerNorm(self.ninp)
        self.fus1 = Fusion(self.ninp, self.ninp, dropout=dropout)
        self.fus2 = Fusion(self.ninp, self.ninp, dropout=dropout)
        #self.out_layernorm = nn.LayerNorm(self.ninp)
        #self.cas_encoder = nn.GRU(input_size=self.ninp * 2, hidden_size=self.ninp * 2, batch_first=True)
        self.decoder = Decoder(input_size=self.ninp, user_size=self.user_size, opt=opt)
        self.init_weights()
        #print(self)

    def init_weights(self):
        init.xavier_normal_(self.linear_pos_embedding.weight)
        init.xavier_normal_(self.att_m)
        init.xavier_normal_(self.att)

    def sin_cos_temporal_enc(self, time):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])

        return result


    def channel_attention(self, *channel_embeddings):
        weights = []
        for embedding in channel_embeddings:
            weights.append(
                torch.sum(
                    torch.multiply(self.att, torch.matmul(embedding, self.att_m)),
                    1))
        embs = torch.stack(weights, dim=0)
        score = F.softmax(embs.t(), dim = -1)
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += torch.multiply(score.t()[i], channel_embeddings[i].t()).t()
        return mixed_embeddings


    def forward(self, input, input_timestamp, input_id, train=True):
        input = input[:, :-1]  # [bsz,max_len]   input_id:[batch_size]
        batch_size, max_len = input.shape
        # graph encoder
        seed_emb = self.user_encoder(input, input_timestamp, train)
        seed_emb = self.dropout(seed_emb)

        # positional embedding
        mask = (input == Constants.PAD).cuda(3)
        seed_batch_t = torch.arange(input.size(1)).expand(input.size()).cuda(3)
        tgt_batch_t = seed_batch_t + 1

        # timeencoder
        linear_seed_batch_t_emb = self.linear_pos_embedding(seed_batch_t)
        linear_tgt_batch_t_emb = self.linear_pos_embedding(tgt_batch_t)

        linear_zero = self.linear_pos_embedding(torch.tensor([[0]]).cuda(3)).view(1,self.ninp).repeat(batch_size,1)
        pos0_emb = seed_emb[:,0]
        # 目标位置注意力
        linear_t_hidden = self.linear_pos_ca(linear_tgt_batch_t_emb, linear_seed_batch_t_emb + seed_emb, seed_emb, mask = mask)

        sincos_t_hidden = self.sin_cos_pos_ca(pos0_emb + linear_zero, linear_seed_batch_t_emb + seed_emb, seed_emb, mask = mask)
        stu_hidden = self.stu_ca(linear_seed_batch_t_emb + seed_emb, linear_seed_batch_t_emb + seed_emb, seed_emb, mask = mask)

        user_hidden = self.fus1(stu_hidden, sincos_t_hidden)

        user_hidden = self.layernorm2(user_hidden)
        outputs, _ = self.cas_encoder(user_hidden)

        outputs = self.fus2(outputs, linear_t_hidden)
        outputs = self.layernorm(outputs)
        outputs, _ = self.cas_encoder2(outputs)

        # Output prediction
        pred = self.decoder(outputs)  # (bsz, max_len, |U|)
        mask = self.get_previous_user_mask(input.cuda(3), self.user_size)
        output = pred.cuda(3) + mask.cuda(3)
        user_predction = output.view(-1, output.size(-1))  # (bsz*max_len, |U|)
        return user_predction, 0.

    def get_previous_user_mask(self, seq, user_size):
        ''' Mask previous activated users.'''
        assert seq.dim() == 2
        prev_shape = (seq.size(0), seq.size(1), seq.size(1))
        seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
        previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
        previous_mask = torch.from_numpy(previous_mask)
        if seq.is_cuda:
            previous_mask = previous_mask.cuda(3)

        masked_seq = previous_mask * seqs.data.float()
        # print(masked_seq.size())

        # force the 0th dimension (PAD) to be masked
        PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
        if seq.is_cuda:
            PAD_tmp = PAD_tmp.cuda(3)
        masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
        ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
        if seq.is_cuda:
            ans_tmp = ans_tmp.cuda(3)
        masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))
        masked_seq = Variable(masked_seq, requires_grad=False)
        return masked_seq




