import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import *

class Casseq(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(Casseq, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.num_layers = args.num_layers
        self.H = args.hidden_units
        self.L = args.maxlen
        self.C = args.num_context
        self.p = args.dropout_rate
        self.N = args.N # number of pseudo sequences
        self.backbone = args.backbone

        if self.backbone != 'ssept': 
            self.item_emb = nn.Embedding(self.item_num + 1, self.H, padding_idx=0)
        else: 
            self.embed = SSEPTEmbedding(user_num, item_num, args)
        self.pos_emb = nn.Embedding(self.L, self.H) # TO IMPROVE
        self.emb_dropout = nn.Dropout(p=self.p)
        
        if args.backbone == 'att' or args.backbone == 'ssept': 
            self.expert_layers = nn.ModuleList([SAInferenceLayer(args) for _ in range(self.num_layers)])
        elif args.backbone == 'gru': 
            self.expert_layers = nn.ModuleList([GRUInferenceLayer(args) for _ in range(self.num_layers)])
        else: 
            raise Exception
        
        self.branch_layers = nn.ModuleList([BranchingLayer(args) for _ in range(self.num_layers)])            
        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
        

    def log2feats(self, log_seqs, user_ids): 
        if self.backbone == 'ssept': 
            seqs = self.embed(log_seqs, user_ids, mode = 'train')
        else:
            seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev)) # [B, L, H]
        seqs *= self.H ** 0.5
        
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs) 
        
        timeline_mask = torch.BoolTensor(log_seqs == 0).unsqueeze(dim = -1).to(self.dev) # [B, L]
        seqs = seqs.masked_fill(timeline_mask, 0.) # broadcast in last dim
        attention_mask = ~torch.tril(torch.ones((self.L, self.L), dtype=torch.bool, device=self.dev)) 
        
        input_seqs = seqs.unsqueeze(dim = -2).repeat(1, 1, self.C, 1) # [B, L, C, H]
        weights_list = list()
        for i in range(self.num_layers):
            weights = self.branch_layers[i](input_seqs, timeline_mask) # [B, L, C(out_num), C(in_num)]
            if i == self.num_layers-1: 
                weights = weights.mean(dim = 2, keepdim = True)
            weights_list.append(weights.transpose(-1, -2))
            
            output_seqs = self.expert_layers[i](input_seqs, timeline_mask, attention_mask) # [B, L, H] * c
            stacked_output = torch.stack(output_seqs, dim = -1) # [B, L, H, C]
            # [B, L, 1, H, C] * [B, L, C(each_input), 1, C] -> [B, L, C, H]
            input_seqs = (stacked_output.unsqueeze(dim = 2) * weights.unsqueeze(dim = -2)).sum(dim = -1) 
        
        log_feats = self.last_layernorm(input_seqs.squeeze(dim = 2)) # [B, L, C, H] -> [B, L, H]
        return log_feats, weights_list
    
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats, weights_list = self.log2feats(log_seqs, user_ids) # [batch_size, maxlen, d]

        if self.backbone == 'ssept': 
            pos_embs = self.embed(pos_seqs, user_ids, mode = 'train')
            neg_embs = self.embed(neg_seqs, user_ids, mode = 'train')
        else:
            pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev)) # [batch_size, maxlen, d]
            neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        
        context_probs = self.weights_product(weights_list)
        context_probs = context_probs.flatten(start_dim = 2) # [B, L, C**layer]
        
        return pos_logits, neg_logits, context_probs # pos_pred, neg_pred
    
    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats, _ = self.log2feats(log_seqs, user_ids) 
        final_feat = log_feats[:, -1, :] 

        if self.backbone == 'ssept': 
            item_embs = self.embed(item_indices, user_ids, mode = 'test')
        else:
            item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)
        
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits # preds # (U, I)
    
    
    def weights_product(self, weights_list):
        '''
        input : [torch.randn(3,4), torch.randn(4,5), torch.randn(5,6)]
        output : torch.Size([3, 4, 5, 6])
        '''
        result = weights_list[0]
        for i in range(len(weights_list) - 1):
            new_layer_w = weights_list[i+1].clone()
            for _ in range(i): 
                new_layer_w.unsqueeze_(dim = 2)
                size = [-1 for _ in range(len(new_layer_w.shape))]
                size[2] = self.C
                new_layer_w = new_layer_w.expand(size)
            result = torch.einsum('{0}ij,{0}jk->{0}ijk'.format('abcdefg'[:i+2]), result, new_layer_w)
        return result
    
    def kl_loss(self, weights, indices):
        kl_criterion = torch.nn.KLDivLoss(reduction='batchmean')
        prior = self.kl_forward().unsqueeze(dim = 0).unsqueeze(dim = 0).repeat(weights.shape[0], weights.shape[1], 1)
        return kl_criterion(prior[indices].log(), weights[indices])
    
    def kl_forward(self):
        log_seqs = torch.empty([self.N, self.L]).random_(1, self.item_num+1).long()
        if self.backbone != 'ssept': seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev)) # [B, L, H]
        else: raise Exception
        seqs *= self.H ** 0.5
        
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        
        pre_log_seqs = torch.ones(self.N, self.L)
        timeline_mask = torch.BoolTensor(pre_log_seqs == 0).unsqueeze(dim = -1).to(self.dev) # [B, L]
        seqs = seqs.masked_fill(timeline_mask, 0.) # broadcast in last dim
        attention_mask = ~torch.tril(torch.ones((self.L, self.L), dtype=torch.bool, device=self.dev)) 
        
        input_seqs = seqs.unsqueeze(dim = -2).repeat(1, 1, self.C, 1) # [B, L, C, H]
        weights_list = list()
        for i in range(self.num_layers):
            weights = self.branch_layers[i](input_seqs, timeline_mask) # [B, L, C(out_num), C(in_num)]
            if i == self.num_layers-1: 
                weights = weights.mean(dim = 2, keepdim = True)
            weights_list.append(weights.transpose(-1, -2))
            output_seqs = self.expert_layers[i](input_seqs, timeline_mask, attention_mask) # [B, L, H] * c
            stacked_output = torch.stack(output_seqs, dim = -1) # [B, L, H, C]
            # [B, L, 1, H, C] * [B, L, C(each_input), 1, C] -> [B, L, C, H]
            input_seqs = (stacked_output.unsqueeze(dim = 2) * weights.unsqueeze(dim = -2)).sum(dim = -1) 
        
        context_probs = self.weights_product(weights_list)
        context_probs = context_probs.flatten(start_dim = 2) # [B, L, C**layer]
        return context_probs.mean(dim = 0, keepdim = False)[-1]