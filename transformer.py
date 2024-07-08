import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InputEmbeddings(nn.Module):
    
    def __init__(self, vocab_size: int, d_model:int) -> None:
        super(InputEmbeddings, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
        
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model :int, seq_len :int, dropout :float) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        #create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        #create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        dev_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        #apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * dev_term)
        pe[:, 1::2] = torch.cos(position * dev_term)
        
        pe = pe.unsqueeze(0) #add a batch dimension
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    
    def __init__(self, eps: float = 10e-6) -> None:
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #multiplicative parameter
        self.bias = nn.Parameter(torch.zeros(1)) #additive parameter
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super(FeedForwardBlock, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) #W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) #W2 and B2
        
    def forward(self, x):
        # (Batch, Seq, d_model) -> (Batch, Seq, d_ff) -> (Batch, Seq, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0 #d_model should be divisible by h
        
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) #Wk
        self.w_v = nn.Linear(d_model, d_model) #Wv
        
        self.w_o = nn.Linear(d_model, d_model) #Wo
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod
    def attention(query_prime, key_prime, value_prime, mask, dropout):
        d_k = query_prime.shape[-1]
        
        # (Batch, h, Seq, d_k) -> (Batch, h, Seq, Seq)
        attention_scores = torch.matmul(query_prime, key_prime.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return torch.matmul(attention_scores, value_prime), attention_scores
        
    def forward(self, q, k, v, mask):
        query_prime = self.w_q(q) # (Batch, Seq, d_model) --> (Batch, Seq, d_model)
        key_prime = self.w_k(k) # (Batch, Seq, d_model) --> (Batch, Seq, d_model)
        value_prime = self.w_v(v) # (Batch, Seq, d_model) --> (Batch, Seq, d_model)
        
        #(Batch, Seq, d_model) -> (Batch, Seq, h, d_k) -> (Batch, h, Seq, d_k)
        query_prime = query_prime.view(-1, query_prime.shape[1], self.h, self.d_k).transpose(1, 2)
        key_prime = key_prime.view(-1, key_prime.shape[1], self.h, self.d_k).transpose(1, 2)
        value_prime = value_prime.view(-1, value_prime.shape[1], self.h, self.d_k).transpose(1, 2)
        
        x, attention_scores = MultiHeadAttention.attention(query_prime, key_prime, value_prime, mask, self.dropout)
        
        # (Batch, h, Seq, d_k) -> (Batch, Seq, h, d_k) -> (Batch, Seq, d_model)
        x = x.transpose(1, 2).contiguous().view(-1, x.shape[2], self.d_model)
        
        # (Batch, Seq, d_model) -> (Batch, Seq, d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    
    def __init__(self, dropout: float) -> None:
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super(EncoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
            
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super(Encoder, self).__init__()
        self.layers = layers  
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x) 
    
class DecoderBlock(nn.Module):
    
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super(DecoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        
    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList) -> None:
        super(Decoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
        
    
#test the transformer
if __name__ ==  "__main__":

    vocab_size = 10
    d_model = 512
    dff = 2048
    num_heads = 8
    
    x = torch.randint(0, 10, (5, 10))
    y = torch.randint(0, 10, (5, 10))
    print(f"Input shape {x.shape}")
    
    
    emb = InputEmbeddings(vocab_size, d_model)
    x = emb(x)
    print(f"Shape of the embedding {x.shape}")
    # [print(f"Input {x[i]}") for i in range(x.shape[0])]
    
    pe = PositionalEncoding(d_model=d_model, seq_len=10, dropout=0.1)
    x = pe(x)
    print(f"Shape of the positional encoding {x.shape}")
    
    norm = LayerNormalization()
    x = norm(x)
    print(f"Shape of the normalization {x.shape}")
    
    ff = FeedForwardBlock(d_model=d_model, d_ff=2048, dropout=0.1)
    x = ff(x)
    print(f"Shape of the feed forward block {x.shape}")
    
    mha = MultiHeadAttention(d_model=d_model, h=num_heads, dropout=0.1)
    # x = mha.attention(x, x, x, None, None)
    x = mha(x, x, x, None)
    print(f"Shape of the multihead attention {x.shape}")
    
    

    