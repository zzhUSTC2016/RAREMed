import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils.layers import GraphConvolution
import numpy as np
import dill

class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class GAMENet(nn.Module):
    def __init__(self, args, voc_size, ehr_adj, ddi_adj):
        super(GAMENet, self).__init__()
        K = len(voc_size)
        self.K = K
        self.args = args
        self.device = torch.device('cuda:{}'.format(args.cuda))
        self.voc_size = voc_size
        self.emb_dim = args.embed_dim
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(self.device)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(voc_size[i], self.emb_dim) for i in range(K-1)])
        self.dropout = nn.Dropout(p=0.5)

        self.encoders = nn.ModuleList([nn.GRU(self.emb_dim, self.emb_dim * 2, batch_first=True) for _ in range(K-1)])

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.emb_dim * 4, self.emb_dim),
        )

        self.ehr_gcn = GCN(voc_size=voc_size[2], emb_dim=self.emb_dim, adj=ehr_adj, device=self.device)
        self.ddi_gcn = GCN(voc_size=voc_size[2], emb_dim=self.emb_dim, adj=ddi_adj, device=self.device)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.emb_dim * 3, self.emb_dim * 2),
            nn.ReLU(),
            nn.Linear(self.emb_dim * 2, voc_size[2])
        )
        self.init_weights()

    def patient_encoder(self, input):
        # generate medical embeddings and queries
        i1_seq = []
        i2_seq = []
        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        for adm in input: 
            # 每一次admission患的所有病的embedding取mean
            i1 = mean_embedding(self.dropout(self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)
            i2 = mean_embedding(self.dropout(self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))
            i1_seq.append(i1)
            i2_seq.append(i2)

        i1_seq = torch.cat(i1_seq, dim=1) #(1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1) #(1,seq,dim)

        o1, h1 = self.encoders[0](
            i1_seq
        ) 
        o2, h2 = self.encoders[1](
            i2_seq
        )
        return o1, o2

    def forward(self, input):
        # input (adm, 3, codes)

        o1, o2 = self.patient_encoder(input)
        patient_representations = torch.cat([o1, o2], dim=-1).squeeze(dim=0) # (seq, dim*4)
        queries = self.query(patient_representations) # (seq, dim)

        # graph memory module
        '''I:generate current input'''
        query = queries[-1:] # (1,dim)

        '''G:generate graph memory bank and insert history information'''
        drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter  # (size, dim)

        if len(input) > 1:
            history_keys = queries[:(queries.size(0)-1)] # (seq-1, dim)

            history_values = np.zeros((len(input)-1, self.voc_size[2]))
            for idx, adm in enumerate(input):
                if idx == len(input)-1:
                    break
                history_values[idx, adm[2]] = 1
            history_values = torch.FloatTensor(history_values).to(self.device) # (seq-1, size)
            
        '''O:read from global memory bank and dynamic memory bank'''
        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)

        if len(input) > 1:
            visit_weight = F.softmax(torch.mm(query, history_keys.t())) # (1, seq-1)
            weighted_values = visit_weight.mm(history_values) # (1, size)
            fact2 = torch.mm(weighted_values, drug_memory) # (1, dim)
        else:
            fact2 = fact1
        '''R:convert O and predict'''
        output = self.output(torch.cat([query, fact1, fact2], dim=-1)) # (1, dim)

        # prompt the probability of indication medications to be higher
        # indication_medication = input[-1][4]
        # indication_medication = torch.LongTensor(indication_medication).to(self.device)
        # contraindication_medication = input[-1][5]
        # contraindication_medication = torch.LongTensor(contraindication_medication).to(self.device)

        # output_mean_0 = output.mean(dim=1)
        # output[0, indication_medication] +=  0.8
        # output = output - output.mean(dim=1) + output_mean_0
        
        # print the max probability of output
        # print('max probability of output:', output.max(dim=1).values)
        # print('min probability of output:', output.min(dim=1))
        # print('mean probability of output:', output.mean(dim=1))
        # print('std probability of output:', output.std(dim=1))
        # print('\n\n')

        # output_mean_0 = output.mean(dim=1)
        # output[0, contraindication_medication] += 0.5
        # output = output - output.mean(dim=1) + output_mean_0


        if self.training:
            neg_pred_prob = F.sigmoid(output)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()

            return output, batch_neg
        else:
            return output

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.embeddings[0].weight.data.uniform_(-initrange, initrange)
        self.embeddings[1].weight.data.uniform_(-initrange, initrange)

        self.inter.data.uniform_(-initrange, initrange)
