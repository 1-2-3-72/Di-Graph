import torch.nn as nn
from models.Attention import WindowAttention

class PFTransformerLayer(nn.Module):
    """Cross-attention layer with progressive top-k focusing."""

    def __init__(self,
                 dim,
                 block_id,
                 layer_id,
                 num_heads,
                 num_topk,
                 mlp_ratio,
                 qkv_bias=True,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()

        self.dim = dim
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.norm1_q = norm_layer(dim)
        self.norm1_kv = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn_win = WindowAttention(
            self.dim,
            layer_id=layer_id,
            num_heads=num_heads,
            num_topk=num_topk,
            qkv_bias=qkv_bias,
        )

    def forward(self, x_q, x_kv, pfa_list, params):
        pfa_values, pfa_indices = pfa_list[0], pfa_list[1]
        B, Nq, C = x_q.shape
        _, Nk, _ = x_kv.shape

        q = self.q_proj(self.norm1_q(x_q))
        k = self.k_proj(self.norm1_kv(x_kv))
        v = self.v_proj(self.norm1_kv(x_kv))

        x, pfa_values, pfa_indices = self.attn_win(q, k, v, pfa_values=pfa_values, pfa_indices=pfa_indices)
        x = self.norm2(x)

        pfa_list = [pfa_values, pfa_indices]
        return x, x_kv,pfa_list

class BasicBlock(nn.Module):
    """Stack progressive focusing layers and optionally return attention maps."""

    def __init__(self,
                 dim,
                 idx,
                 layer_id,
                 depth,
                 num_heads,
                 num_topk,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False, ):

        super().__init__()
        self.dim = dim

        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                PFTransformerLayer(
                    dim=dim,
                    block_id=idx,
                    layer_id= layer_id + i,
                    num_heads=num_heads,
                    num_topk=num_topk,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                )
            )

    def forward(self, x_q, x_kv, pfa_list, params, return_intermediates=False):
        intermediates = []
        for layer in self.layers:

            x_q, x_kv, pfa_list = layer(x_q, x_kv, pfa_list, params)
            if return_intermediates:
                intermediates.append({
                    "layer_id": layer.layer_id,
                    "pfa_values": pfa_list[0],
                    "pfa_indices": pfa_list[1],
                })

        if return_intermediates:
            return x_q, pfa_list, intermediates
        return x_q, pfa_list

