import math
import torch
import torch.nn as nn
import numpy as np

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model # 512
        self.vocab_size = vocab_size # size of the vocabulary
        self.embeddings = nn.Embedding(vocab_size, d_model) # embedding layer with each token represented by a vector of size d_model
    
    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model)

class PositionalEmbeddings(nn.Module):
    def __init__(self, d_model: int, seq_length: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)

        # Creating a matrix of shape (seq_length, d_model)
        pe = torch.zeros(seq_length, d_model)

        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1) # shape (seq_length, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # shape (d_model/2,)
        pe[:, 0::2] = torch.sin(pos * div_term) # sin(position * (10000 ** (2i / d_model))
        pe[:, 1::2] = torch.cos(pos * div_term) # cos(position * (10000 ** (2i / d_model))

        # Now we will add a batch
        pe = pe.unsqueeze(0) # (1, seq_length, d_model)

        # We will now store it in a buffer so that it is not updated during backpropagation
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return self.dropout(x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)) # adding positional embeddings to the word embeddings
    

class LayerNormalization(nn.Module):
    def __init__(self,features: int, eps: float = 1e-06):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features)) # It is multiplied
        self.beta = nn.Parameter(torch.zeros(features)) # It is added
    
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # d_model = 512, d_ff = 2048
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        # 512 -> 2048 -> 512
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h # Number of heads, 8

        # Making sure that d_model is divisible by h
        assert d_model % h == 0, "d_model should be divisible by h"

        # Initializing the weights
        self.d_k = d_model // h
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch_size, h, seq_len, d_k) @ (batch_size, h, seq_len, d_k) --> (batch_size, h, seq_len, seq_len)
        attention_score = (query @ key.transpose(-2, -1))/math.sqrt(d_k)
        if mask is not None:
            # we will replace the 0 with (-inf) before pasing it throught the softmax
            attention_score.masked_fill(mask == 0, -torch.inf)
        
        attention_score = attention_score.softmax(dim=-1)

        if dropout is not None:
            attention_score = dropout(attention_score)
        
        # (batch_size, h, seq_len, seq_len) @ (batch_size, h, seq_len, d_k) --> (batch_size, h, seq_len, d_k)
        return (attention_score @ value), attention_score
    
    def forward(self, q, k, v, mask):
        query = self.wq(q) # (batch_size, seq_length, d_model) -> (batch_size, seq_length, d_model)
        key = self.wk(k) # (batch_size, seq_length, d_model) -> (batch_size, seq_length, d_model)
        value = self.wv(v) # (batch_size, seq_length, d_model) -> (batch_size, seq_length, d_model)
        
        # Splitting the query, key and value into h heads and reordering
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_score = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # Joining back all the heads together
        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Finally multiply with W_o
        # (batch_size, seq_len, d_model) @ (batch_size, d_model, d_model) -> (batch_size, seq-len, d_model) 
        return self.wo(x)

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    # This is a single encoder block out of the N encoder blocks
    def __init__(self, features: int, self_attention: MultiHeadAttention, feed_forward: FFN, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x

class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, feed_forward: FFN, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
    
    def forward(self, x, encoder_out, src_mask, target_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, target_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, encoder_out, encoder_out, src_mask))
        x = self.residual_connections[2](x, self.feed_forward)
        return x
    
class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    
    def forward(self, x, encoder_out, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, target_mask)
        return self.norm(x)

class LinearLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return self.linear(x)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embd: InputEmbeddings, target_embd: InputEmbeddings, src_pos: PositionalEmbeddings, target_pos: PositionalEmbeddings, linear: LinearLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embd = src_embd
        self.target_embd = target_embd
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.linear = linear
    
    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embd(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_out, src_mask, target, target_mask):
        # (batch, seq_len, d_model)
        target = self.target_embd(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_out, src_mask, target_mask)
    
    def linear_layer(self, x):
        return self.linear(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048):
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEmbeddings(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEmbeddings(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FFN(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FFN(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = LinearLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer