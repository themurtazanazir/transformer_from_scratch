import torch
import einops


def scaled_dot_product_attention(queries, keys, values, masks):
    ## all are batched
    d_k = keys.shape[-1]

    scores = torch.bmm(queries, keys.transpose(1, 2)) / d_k
    scores = scores + masks
    weights = scores.softmax(dim=-1)

    attn = torch.bmm(weights, values)

    return attn


class SimpleMultiHeadAttention:
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


class FasterMultiHeadAttention:
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
        queries = x
        keys = x
        values = x
        masks = torch.zeros_like(queries)
        return super().forward(queries, keys, values, masks)


class DecoderSelfAttention(FasterMultiHeadAttention):
    def __init__(self, h, d_model):
        super().__init__(h, d_model, d_model, d_model)

    def forward(self, x):
        queries = x
        keys = x
        values = x
        b, t, d = queries.shape
        masks = torch.triu(torch.empty(b, t, t).fill_(float("-inf")), diagonal=1)
        return super().forward(queries, keys, values, masks)


# ~Layernorm~,
class LayerNorm:
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
class FeedForwardNetwork:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff

        self.W1 = torch.rand(d_model, d_ff, requires_grad=True)
        self.b1 = torch.rand(d_ff, requires_grad=True)
        self.W2 = torch.rand(d_ff, d_model, requires_grad=True)
        self.b2 = torch.rand(d_ff, requires_grad=True)

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


if __name__ == "__main__":
    bs = 2
    ts = 10
    d_model = 64
    n_heads = 8

    Q = torch.rand(bs, ts, d_model)
    K = torch.rand(bs, ts, d_model)
    V = torch.rand(bs, ts, d_model)
    mask = torch.rand(bs, ts, ts)
    m1 = SimpleMultiHeadAttention(n_heads, d_model, d_model, d_model)
    o1 = m1.forward(Q, K, V, mask)
    # print(o1.shape)

    m2 = FasterMultiHeadAttention(n_heads, d_model, d_model, d_model)
    m2.W_Q = torch.concat(m1.W_q, dim=-1)
    m2.W_K = torch.concat(m1.W_k, dim=-1)
    m2.W_V = torch.concat(m1.W_v, dim=-1)
    m2.w_o = m1.w_o
    o2 = m2.forward(Q, K, V, mask)
    # print(o2.shape)

    print((o1 == o2).all())

    ln = LayerNorm(d_model, 1e-6)
    from torch.nn import LayerNorm as TorchLayerNorm

    tln = TorchLayerNorm(d_model, eps=1e-6)
    out = ln.forward(o1)
    tout = tln(o1)
    print((out - tout).max())
