import torch
import einops


def scaled_dot_product_attention(queries, keys, values, masks):
    ## all are batched
    d_k = keys.shape[-1]
    
    scores = torch.bmm(queries, keys.transpose(1, 2))/d_k
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
        
        
        self.w_o =  torch.rand(h*d_v, d_model, requires_grad=True)
    
    def forward(self, queries, keys, values, masks):
        outputs = []
        for i in range(self.h):
            new_queries = torch.matmul(queries, self.W_q[i])
            new_keys = torch.matmul(keys, self.W_k[i])
            new_values = torch.matmul(values, self.W_v[i])
            head_i = scaled_dot_product_attention(new_queries, new_keys, new_values, masks)
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
        self.W_Q = torch.rand(d_model, h*d_k, requires_grad=True)       
        self.W_K = torch.rand(d_model, h*d_k, requires_grad=True)       
        self.W_V = torch.rand(d_model, h*d_v, requires_grad=True)   
        
        self.w_o =  torch.rand(h*d_v, d_model, requires_grad=True)
   
    def forward(self, queries, keys, values, masks):
        queries = torch.matmul(queries, self.W_Q)
        keys = torch.matmul(keys, self.W_K)
        values = torch.matmul(values, self.W_V)

        ## b, t, h*d_whatever => b*h, ts, h
        queries = einops.rearrange(queries, 'b t (h d) -> (b h) t d', h=self.h)
        keys = einops.rearrange(keys, 'b t (h d) -> (b h) t d', h=self.h)
        values = einops.rearrange(values, 'b t (h d) -> (b h) t d', h=self.h)
        ## b t t => b*h t t
        masks = einops.repeat(masks, 'b t1 t2 -> (b h) t1 t2', h=self.h)

        output = scaled_dot_product_attention(queries, keys, values, masks)
        output = einops.rearrange(output, '(b h) t d -> b t  (h d)', h=self.h)
        output = torch.matmul(output, self.w_o)

        return output
       
## TODO: add layernorm   
       
    
if __name__ == '__main__':
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
    m2.W_Q  = torch.concat(m1.W_q, dim=-1)
    m2.W_K  = torch.concat(m1.W_k, dim=-1)
    m2.W_V  = torch.concat(m1.W_v, dim=-1)
    m2.w_o = m1.w_o
    o2 = m2.forward(Q, K, V, mask)
    # print(o2.shape)
    
    print(o1-o2)
    
    