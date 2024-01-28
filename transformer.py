import torch
import einops
from utils import Module


def scaled_dot_product_attention(queries, keys, values, masks):
    ## all are batched
    d_k = keys.shape[-1]
    scores = torch.bmm(queries, keys.transpose(1, 2)) / d_k
    scores = scores + masks
    weights = scores.softmax(dim=-1)

    attn = torch.bmm(weights, values)

    return attn


class SimpleMultiHeadAttention(Module):
    def __init__(self, h, d_model, d_k, d_v):
        self.h = h
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.W_q = [torch.rand(d_model, d_k, requires_grad=True) for _ in range(h)]
        self.W_k = [torch.rand(d_model, d_k, requires_grad=True) for _ in range(h)]
        self.W_v = [torch.rand(d_model, d_v, requires_grad=True) for _ in range(h)]

        self.w_o = torch.rand(h * d_v, d_model, requires_grad=True)

    def forward(self, queries, keys, values, masks):
        outputs = []
        for i in range(self.h):
            new_queries = torch.matmul(queries, self.W_q[i])
            new_keys = torch.matmul(keys, self.W_k[i])
            new_values = torch.matmul(values, self.W_v[i])
            head_i = scaled_dot_product_attention(
                new_queries, new_keys, new_values, masks
            )
            outputs.append(head_i)

        output = torch.concat(outputs, dim=-1)

        output = torch.matmul(output, self.w_o)

        return output


class FasterMultiHeadAttention(Module):
    def __init__(self, h, d_model, d_k, d_v):
        self.h = h
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        ## concat all head weights into one
        self.W_Q = torch.rand(d_model, h * d_k, requires_grad=True)
        self.W_K = torch.rand(d_model, h * d_k, requires_grad=True)
        self.W_V = torch.rand(d_model, h * d_v, requires_grad=True)

        self.w_o = torch.rand(h * d_v, d_model, requires_grad=True)

    def forward(self, queries, keys, values, masks):
        queries = torch.matmul(queries, self.W_Q)
        keys = torch.matmul(keys, self.W_K)
        values = torch.matmul(values, self.W_V)

        ## b, t, h*d_whatever => b*h, ts, h
        queries = einops.rearrange(queries, "b t (h d) -> (b h) t d", h=self.h)
        keys = einops.rearrange(keys, "b t (h d) -> (b h) t d", h=self.h)
        values = einops.rearrange(values, "b t (h d) -> (b h) t d", h=self.h)
        ## b t t => b*h t t
        masks = einops.repeat(masks, "b t1 t2 -> (b h) t1 t2", h=self.h)

        output = scaled_dot_product_attention(queries, keys, values, masks)
        output = einops.rearrange(output, "(b h) t d -> b t  (h d)", h=self.h)
        output = torch.matmul(output, self.w_o)

        return output


class EncoderSelfAttention(FasterMultiHeadAttention):
    def __init__(self, h, d_model):
        super(EncoderSelfAttention, self).__init__(h, d_model, d_model, d_model)

    def forward(self, x):
        b, t, d = x.shape
        masks = torch.zeros(b, t, t)
        return super().forward(x, x, x, masks)


class DecoderSelfAttention(FasterMultiHeadAttention):
    def __init__(self, h, d_model):
        super().__init__(h, d_model, d_model, d_model)

    def forward(self, x):
        b, t, d = x.shape
        masks = torch.triu(torch.empty(b, t, t).fill_(float("-inf")), diagonal=1)
        return super().forward(x, x, x, masks)


class EncoderDecoderAttention(FasterMultiHeadAttention):
    def __init__(self, h, d_model):
        super().__init__(h, d_model, d_model, d_model)

    def forward(self, x, encoder_output):
        b, t, d = x.shape
        masks = torch.zeros(b, t, t)
        return super().forward(x, x, encoder_output, masks)


# ~Layernorm~,
class LayerNorm(Module):
    def __init__(self, d_model, eps):
        self.d_model = d_model
        self.eps = eps
        self.gamma = torch.ones((d_model,), requires_grad=True)
        self.beta = torch.zeros((d_model,), requires_grad=True)

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        norm = norm * self.gamma + self.beta
        return norm


# ~FFN~
class FeedForwardNetwork(Module):
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff

        self.W1 = torch.rand(d_model, d_ff, requires_grad=True)
        self.b1 = torch.rand(d_ff, requires_grad=True)
        self.W2 = torch.rand(d_ff, d_model, requires_grad=True)
        self.b2 = torch.rand(d_model, requires_grad=True)

    def forward(self, x):
        x = torch.matmul(x, self.W1) + self.b1
        x = torch.relu(x)
        x = torch.matmul(x, self.W2) + self.b2
        return x


# ~PE~
def positional_encoding(T, d_model):
    pe = torch.zeros(T, d_model)
    pos = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
    div = torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float) / d_model)
    angle = pos / div
    pe[:, ::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)

    return pe


class PositionEmbedding(Module):
    def __init__(self, vocab_size, embedding_size):
        self.emb_weight = torch.rand(vocab_size, embedding_size, requires_grad=True)

    def forward(self, x):
        emb = self.emb_weight[x]
        b, t, d = emb.shape
        pos_enc = positional_encoding(t, d)
        return emb + pos_enc


class EncoderLayer(Module):
    def __init__(self, d_model, h, d_ff, layernorm_eps=1e-8):
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.layernorm_eps = layernorm_eps

        self.self_attn = EncoderSelfAttention(h=h, d_model=d_model)
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff)
        self.layer_norm1 = LayerNorm(d_model=d_model, eps=layernorm_eps)
        self.layer_norm2 = LayerNorm(d_model=d_model, eps=layernorm_eps)

    def forward(self, x):
        self_attn = self.self_attn(x)
        x = self.layer_norm1(x + self_attn)
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)

        return x


class Encoder(Module):
    def __init__(self, vocab_size, n_layers, d_model, h, d_ff):
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff

        self.pos_emb = PositionEmbedding(vocab_size=vocab_size, embedding_size=d_model)
        self.encoder_layers = [
            EncoderLayer(d_model, h, d_ff) for _ in range(self.n_layers)
        ]

    def forward(self, x):
        x = self.pos_emb(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return x


class DecoderLayer(Module):
    def __init__(self, d_model, h, d_ff, layernorm_eps=1e-8):
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.layernorm_eps = layernorm_eps

        self.self_attn = DecoderSelfAttention(self.h, self.d_model)
        self.cross_attn = EncoderDecoderAttention(self.h, self.d_model)

        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff)
        self.layer_norm1 = LayerNorm(d_model=d_model, eps=layernorm_eps)
        self.layer_norm2 = LayerNorm(d_model=d_model, eps=layernorm_eps)
        self.layer_norm3 = LayerNorm(d_model=d_model, eps=layernorm_eps)

    def forward(self, x, encoder_output):
        self_attn = self.self_attn(x)
        x = self.layer_norm1(x + self_attn)

        cross_attn = self.cross_attn(x, encoder_output)
        x = self.layer_norm2(x + cross_attn)

        ffn_out = self.ffn(x)
        x = self.layer_norm3(x + ffn_out)

        return x


class Decoder(Module):
    def __init__(self, vocab_size, n_layers, d_model, h, d_ff):
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff

        self.pos_emb = PositionEmbedding(vocab_size=vocab_size, embedding_size=d_model)

        self.decoder_layers = [
            DecoderLayer(d_model, h, d_ff) for _ in range(self.n_layers)
        ]

        ## paper shares these weights but fuck it for now
        self.out_weight = torch.rand(d_model, vocab_size, requires_grad=True)
        self.bias = torch.rand(vocab_size, requires_grad=True)

    def forward(self, x, encoder_out):
        x = self.pos_emb(x)
        for layer in self.decoder_layers:
            x = layer(x, encoder_out)

        x = torch.matmul(x, self.out_weight)
        x = x + self.bias

        return x


class Transformer(Module):
    def __init__(self, in_vocab_size, out_vocab_size, n_layers, d_model, h, d_ff):
        self.in_vocab_size = in_vocab_size
        self.out_vocab_size = out_vocab_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff

        self.encoder = Encoder(in_vocab_size, n_layers, d_model, h, d_ff)
        self.decoder = Decoder(out_vocab_size, n_layers, d_model, h, d_ff)

    def forward(self, input, shifted_target):
        encoder_out = self.encoder(input)
        decoder_out = self.decoder(shifted_target, encoder_out)
        return decoder_out


if __name__ == "__main__":
    d_model = 512
    n_layers = 6
    d_ff = 2048
    h = 8

    in_vocab_size = 100
    out_vocab_size = 52

    batch = 5
    T = 128

    input_seq = torch.randint(0, in_vocab_size, (batch, T))
    shifted_target_seq = torch.randint(0, out_vocab_size, (batch, T))

    transformer = Transformer(in_vocab_size, out_vocab_size, n_layers, d_model, h, d_ff)

    print(transformer(input_seq, shifted_target_seq).shape)
