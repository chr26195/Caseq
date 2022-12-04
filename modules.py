import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class SSEPTEmbedding(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SSEPTEmbedding, self).__init__()
        self.item_num = item_num
        self.user_num = user_num
        self.iH = args.hidden_units - args.user_D
        self.uH = args.user_D
        self.dev = args.device
        self.sse_prob = args.sse_prob
        
        self.item_emb = torch.nn.Embedding(self.item_num+1, self.iH, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num+1, self.uH, padding_idx=0)
        
    def sse_mask(self, user_ids):
        user_ids = torch.LongTensor(user_ids).to(self.dev)
        indices = torch.empty_like(user_ids).float().uniform_(0, 1).lt(self.sse_prob)
        user_ids[indices] = user_ids[indices].random_(1, self.user_num+1)
        return user_ids
    
    def forward(self, log_seqs, user_ids, mode = 'train'):
        item_rep = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        if mode == 'train': user_rep = self.user_emb(self.sse_mask(user_ids)).unsqueeze(dim = 1).repeat(1, item_rep.shape[1], 1)
        elif mode == 'test': user_rep = self.user_emb(self.sse_mask(user_ids)).repeat(item_rep.shape[0], 1)
        else: raise Exception
        seqs = torch.cat([item_rep, user_rep], dim = -1)
        return seqs

class PointWiseFeedForward(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(input_size, input_size, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(input_size, output_size, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        if outputs.shape[-1] == inputs.shape[-1]: outputs += inputs # residual link
        return outputs
    
class SAInferenceLayer(nn.Module):
    def __init__(self, args):
        super(SAInferenceLayer, self).__init__()
        self.C = args.num_context
        self.H = args.hidden_units
        self.p = args.dropout_rate
        
        self.attention_layernorm = nn.ModuleList([nn.LayerNorm(self.H, eps=1e-8) for _ in range(self.C)])
        self.attention_layer = nn.ModuleList([nn.MultiheadAttention(self.H, 1, self.p) for _ in range(self.C)])
        self.forward_layernorm = nn.ModuleList([nn.LayerNorm(self.H, eps=1e-8) for _ in range(self.C)])
        self.forward_layer = nn.ModuleList([PointWiseFeedForward(self.H, self.H, self.p) for _ in range(self.C)])
        
    def expert_forward(self, i, seqs, timeline_mask, attention_mask):
        seqs = torch.transpose(seqs, 0, 1)
        Q = self.attention_layernorm[i](seqs)
        mha_outputs, _ = self.attention_layer[i](Q, seqs, seqs, attn_mask = attention_mask)
        seqs = Q + mha_outputs
        seqs = torch.transpose(seqs, 0, 1)  # [B, L, H]
        seqs = self.forward_layernorm[i](seqs)
        seqs = self.forward_layer[i](seqs)
        seqs = seqs.masked_fill(timeline_mask, 0.)   # [B, L, H]
        return seqs
                
    def forward(self, seqs, timeline_mask, attention_mask): # seqs : [B, L, C, H]
        experts = [self.expert_forward(i, seqs[:,:,i,:], timeline_mask, attention_mask) for i in range(self.C)] 
        return experts 
    
class GRUInferenceLayer(nn.Module):
    def __init__(self, args):
        super(GRUInferenceLayer, self).__init__()
        self.C = args.num_context
        self.H = args.hidden_units
        self.L = args.maxlen
        self.p = args.dropout_rate
        
        self.gru_layer = nn.ModuleList([nn.GRU(self.H, self.H, batch_first=True) for _ in range(self.C)])
        
    def expert_forward(self, i, seqs, timeline_mask, attention_mask):
        length = torch.count_nonzero(~timeline_mask.squeeze(dim = -1), dim = 1) # count sequence length, [B]
        zero_length = torch.count_nonzero(timeline_mask.squeeze(dim = -1), dim = 1)
        batch_size = len(length)
        
        seqs = torch.unbind(seqs, dim = 0)
        seqs = list(map(torch.roll, seqs, length.tolist(), [0] * batch_size))
        seqs = torch.stack(seqs, dim = 0)
        seqs, _ = self.gru_layer[i](seqs)
        seqs = torch.unbind(seqs, dim = 0)
        seqs = list(map(torch.roll, seqs, zero_length.tolist(), [0] * batch_size))
        seqs = torch.stack(seqs, dim = 0)
        
        seqs = seqs.masked_fill(timeline_mask, 0.)
        return seqs
                
    def forward(self, seqs, timeline_mask, attention_mask): # seqs : [B, L, C, H]
        experts = [self.expert_forward(i, seqs[:,:,i,:], timeline_mask, attention_mask) for i in range(self.C)] 
        return experts # [B, L, H] * c

class BranchingLayer(nn.Module):
    def __init__(self, args):
        super(BranchingLayer, self).__init__()
        self.args = args
        self.C = args.num_context
        self.H = args.hidden_units
        self.use_gumbel = False
        self.activation = F.tanh
        
        self.global_gate1 = nn.Parameter(torch.randn(self.C, self.H, self.H), requires_grad=True) 
        self.global_gate2 = nn.Parameter(torch.randn(self.C, self.H, 1), requires_grad=True) 
    
    def forward(self, seqs, timeline_mask):
        inputs = seqs # [B, L, C, H]
        weight_logits = []
        for i in range(self.C):
            sim = self.activation(inputs @ self.global_gate1[i])
            sim = sim @ self.global_gate2[i]
            weight_logits.append(sim.squeeze(dim = -1))

        weight_logits = torch.stack(weight_logits, dim = -1) # [B, L, C(in_num), C]
        
        if self.use_gumbel: weights = F.gumbel_softmax(weight_logits, tau=1, dim=-1)
        else: weights = weight_logits.softmax(dim = -1)
        weights = weights.masked_fill(timeline_mask.unsqueeze(dim = -1), 0.)
        return weights # [B, L, C(in_num), C]