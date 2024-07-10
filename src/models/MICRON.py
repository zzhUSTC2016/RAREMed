import torch
import torch.nn as nn
import torch.nn.functional as F

class MICRON(nn.Module):
    def __init__(self, vocab_size, ddi_adj, emb_dim=256, device=torch.device('cpu:0')):
        super(MICRON, self).__init__()

        self.device = device

        # pre-embedding
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(2)])
        self.dropout = nn.Dropout(p=0.5)
        
        self.health_net = nn.Sequential(
                nn.Linear(2 * emb_dim, emb_dim)
        )

        # 
        self.prescription_net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.ReLU(),
            nn.Linear(emb_dim * 4, vocab_size[2])
        )

        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.init_weights()

    def forward(self, input):

	    # patient health representation
        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        
        diag_emb = sum_embedding(self.dropout(self.embeddings[0](torch.LongTensor(input[-1][0]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)
        prod_emb = sum_embedding(self.dropout(self.embeddings[1](torch.LongTensor(input[-1][1]).unsqueeze(dim=0).to(self.device))))
        # diag_emb = torch.cat(diag_emb, dim=1) #(1,seq,dim)
        # prod_emb = torch.cat(prod_emb, dim=1) #(1,seq,dim)

        if len(input) < 2:
            diag_emb_last = diag_emb * torch.tensor(0.0)
            prod_emb_last = diag_emb * torch.tensor(0.0)
        else:
            diag_emb_last = sum_embedding(self.dropout(self.embeddings[0](torch.LongTensor(input[-2][0]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)
            prod_emb_last = sum_embedding(self.dropout(self.embeddings[1](torch.LongTensor(input[-2][1]).unsqueeze(dim=0).to(self.device))))
            # diag_emb_last = torch.cat(diag_emb_last, dim=1) #(1,seq,dim)
            # prod_emb_last = torch.cat(prod_emb_last, dim=1) #(1,seq,dim)

        health_representation = torch.cat([diag_emb, prod_emb], dim=-1).squeeze(dim=0) # (seq, dim*2)
        health_representation_last = torch.cat([diag_emb_last, prod_emb_last], dim=-1).squeeze(dim=0) # (seq, dim*2)

        health_rep = self.health_net(health_representation)[-1:, :] # (seq, dim)
        health_rep_last = self.health_net(health_representation_last)[-1:, :] # (seq, dim)
        health_residual_rep = health_rep - health_rep_last

	    # drug representation
        drug_rep = self.prescription_net(health_rep)
        drug_rep_last = self.prescription_net(health_rep_last)
        drug_residual_rep = self.prescription_net(health_residual_rep)  

        # reconstructon loss
        rec_loss = 1 / self.tensor_ddi_adj.shape[0] * torch.sum(torch.pow((F.sigmoid(drug_rep) - F.sigmoid(drug_rep_last + drug_residual_rep)), 2))
        
        # ddi_loss
        neg_pred_prob = F.sigmoid(drug_rep)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)

        batch_neg = 1 / self.tensor_ddi_adj.shape[0] *  neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return drug_rep, drug_rep_last, drug_residual_rep, batch_neg, rec_loss

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
