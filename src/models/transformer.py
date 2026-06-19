import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout_rate):
        super().__init__()

        assert embed_dim % n_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.dropout_rate = dropout_rate

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)
        self.out_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Dim here is head dim * n heads
        batches, length, dim = x.shape

        q, k, v = self.qkv(x).chunk(3, dim = -1) 
        # when we operate on QKV, we want to do so on a view where the heads are seperate so F.scaled_dot_product_attention can work on mutliple heads instead of one huge one
        q = q.view(batches, length, self.n_heads, self.head_dim).transpose(1, 2) # Transpose because F.sdpa requires B H L D/H
        k = k.view(batches, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batches, length, self.n_heads, self.head_dim).transpose(1, 2)

        x = F.scaled_dot_product_attention(
            query = q, key = k, value = v, attn_mask = None, is_causal = False,
            dropout_p = self.dropout_rate if self.training else 0
        )

        x = x.transpose(1, 2).reshape(batches, length, dim) # reorder to B L H D/H, then combine H and D back together for multiplication by output
        x = self.output(x)
        x = self.out_dropout(x)

        return x

# Basic MLP implementation with dropout
# why LN instead of BN: https://stats.stackexchange.com/questions/474440/why-do-transformers-use-layer-norm-instead-of-batch-norm
# Transformers have small batches, sequence lengths are variable --> feature-wise normalization does not make much sense
class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

# This is a transformer layer consisting of a attention layer and an MLP
# Using postnorm for consistency with maskgit paper -- but prenorm should be better: https://arxiv.org/pdf/2002.04745
class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout_rate, n_heads):
        super().__init__()
        self.attention = Attention(embed_dim, n_heads, dropout_rate)
        self.mlp = MLP(embed_dim, hidden_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = x + self.attention(x)
        x = self.norm1(x)
        x = x + self.mlp(x)
        x = self.norm2(x)
        return x

# Using the basic 1D sequence embedding, which fixes a trained transformer to a certain sequence length (otherwise the positions and wrapping no longer make sense).
# Look into an extendable embedding strategy, like rope, sinusoidal, or alibi
# For now, though, work around this by shifting the window depending on what needs to be generated
class Embed(nn.Module):
    def __init__(self, codebook_size, embed_dim, sequence_length, dropout_rate):
        super().__init__()
        self.mask_token = codebook_size
        # Indices 0 .. codebook_size - 1 are codes; index codebook_size is the mask token.
        self.word_embedder = nn.Embedding(codebook_size + 1, embed_dim)
        self.pos_embedder = nn.Embedding(sequence_length, embed_dim)

        self.register_buffer("pos_indices", torch.arange(sequence_length))

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        word_embedding = self.word_embedder(x) # B L D

        pos_embedding = self.pos_embedder(self.pos_indices) # B L D

        # TODO: I really hate this conceptually, even if it turns out to work in practice
        # https://link.springer.com/article/10.1007/s11063-024-11539-7
        x = self.norm(word_embedding + pos_embedding)
        x = self.dropout(x)
        return x

# This module is responsible for mapping from transformer space back to codebook space
"""
the word embedder matrix originally mapped from codebook indices 
to transformer space. each row is the corresponding vector for one codebook entry. 
transposed, now each column is a codebook entry's transformer space equivalent. 
multiply the output vectors each by the transposed matrix --> vector of length codebook size where each entry is
the dot product between the predicted vector and one of the codebook vectors.
the final output of this layer is length sequence length, depth codebook size, and each entry is a logit.
""" 
class MLM(nn.Module):
    def __init__(self, embed_dim, sequence_length, codebook_size, word_embedder):
        super().__init__()
        self.word_embedder = word_embedder
        self.codebook_size = codebook_size
        self.bias = nn.Parameter(torch.zeros(codebook_size))

        self.lin = nn.Linear(embed_dim, embed_dim)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.lin(x)
        x = self.gelu(x)
        x = self.norm(x)

        weight = self.word_embedder.weight[:self.codebook_size]
        # This is the same as weight = x = torch.matmul(x, weight) + self.bias
        # the word embedder  is (codebooks size, embedding dim), and F.linear expects weights to be of shape
        # (output dim, input dim)
        x = F.linear(x, weight, self.bias)  # Double check dims on this

        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.codebook_size = config['codebook_size']
        self.hidden_dim = config['hidden_dim']
        self.num_hidden_layers = config['num_hidden_layers']
        self.num_attention_heads = config['num_attention_heads']
        self.mlp_hidden_dim = config['mlp_hidden_dim']
        self.hidden_dropout_rate = config['hidden_dropout_rate']
        self.attention_dropout_rate = config['attention_dropout_rate']
        self.sequence_length = config['sequence_length']
        
        self.embed = Embed(codebook_size = self.codebook_size, embed_dim = self.hidden_dim, 
                           sequence_length = self.sequence_length, dropout_rate = self.hidden_dropout_rate
                           )
        self.transformer = nn.ModuleList([])
        for _ in range(self.num_hidden_layers):
            self.transformer.append(
                TransformerLayer(self.hidden_dim, self.mlp_hidden_dim, self.attention_dropout_rate, self.num_attention_heads)
                )

        self.mlm = MLM(embed_dim = self.hidden_dim, sequence_length = self.sequence_length, 
                       codebook_size = self.codebook_size, word_embedder = self.embed.word_embedder
                       )

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path)['model_state_dict'])

    def forward(self, x):

        # embed codebook indices
        x = self.embed(x)

        for layer in self.transformer:
            x = layer(x)

        logits = self.mlm(x)

        return logits