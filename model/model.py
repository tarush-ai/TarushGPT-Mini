import numpy as np
import torch
import torch.nn as nn
from config import vocab_size, layers, d_model, heads, d_head, d_ff


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock() for i in range(layers)])
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, X):
        embeddings = self.embeddings(X)
        positionals = torch.tensor(self.positionals[:X.shape[0]]).float()
        embeddings = embeddings + positionals
        for block in self.blocks:
            embeddings = block(embeddings)
        
        return self.linear(embeddings)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.prenorm = LayerNorm()
        self.attentionblock = AttentionBlock()
        self.ffn = FFN()
        self.norm2 = LayerNorm()
    
    def forward(self, X):
        X = self.prenorm(X + self.attentionblock(X))
        X = self.norm2(X + self.ffn(X))
        return X
        
class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attentionheads = nn.ModuleList([AttentionHead() for i in range(heads)])
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, X):
        headoutputs = [head(X) for head in self.attentionheads]
        MHA = torch.cat(headoutputs, dim=-1)
        return self.Wo(MHA)


class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, X):
        return self.norm(X)

class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.f2 = nn.Linear(d_ff, d_model)
    
    def forward(self, X):
        return self.f2(self.relu(self.f1(X)))



class AttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.queries = nn.Linear(d_model, d_head, bias=False)
        self.keys = nn.Linear(d_model, d_head, bias=False)
        self.values = nn.Linear(d_model, d_head, bias=False)

    def forward(self, X):
        Q = self.queries(X)
        K = self.keys(X)
        V = self.values(X)

        scores = Q @ K.T
        scores /= (d_head ** 0.5)
        mask = torch.tril(torch.ones(X.shape[0], X.shape[0], device=X.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(scores, dim=-1)
        return attention @ V
