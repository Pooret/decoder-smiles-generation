import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# Positional Encoding

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, model_dim)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2) * -(math.log(10000.0) / model_dim)
        )
        encoding[:, 0::2] = torch.sin(pos * div_term)
        encoding[:, 1::2] = torch.cos(pos * div_term)
        encoding = encoding.unsqueeze(0)
        self.register_buffer('encoding', encoding) # ensures tensor is moved along with model to device

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :]

class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, feed_forward_dim=128, dropout_rate=0.2):
        super(DecoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout_rate)
        self.linear1 = nn.Linear(model_dim, feed_forward_dim)
        self.linear2 = nn.Linear(feed_forward_dim, model_dim)
        
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        outputs = self.self_attention(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        x = x + self.dropout1(outputs)
        x = self.layer_norm1(x)
        
        outputs = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout2(outputs)
        x = self.layer_norm2(x)
        
        return x
    
class SMILESDecoder(nn.Module):
    def __init__(self, vocab_size, model_dim, num_heads, num_layers, dropout_rate):
        super(SMILESDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        self.fc_out = nn.Lnear(model_dim, vocab_size)
        self.layers = nn.ModuleList([DecoderLayer(model_dim, num_heads, dropout_rate) for _ in range(num_layers)])
        self.fc_out = nn.Linear(model_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_rate) 
        
    def forward(self, x):
        x = self.embedding(x) * torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float32))
        x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer(x)
            
        output = self.fc_out(self.dropout(x))