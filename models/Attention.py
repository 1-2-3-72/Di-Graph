import torch
import torch.nn as nn

class WindowAttention(nn.Module):
    """Multi-head attention with progressively retained key indices."""

    def __init__(self, dim, layer_id, num_heads, num_topk, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.num_topk = num_topk
        self.qkv_bias = qkv_bias
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.eps = 1e-20
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.topk_ratio = self.num_topk[self.layer_id]

    def forward(self, q, k, v, pfa_values, pfa_indices):
        """Attend from q (B, Nq, C) to k/v (B, Nk, C), reusing prior attention values and indices when provided."""

        B, Nq, C = q.shape
        Nk = k.shape[1]
        self.topk = max(1, int(self.topk_ratio * Nk))

        q = q.view(B, Nq, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B * self.num_heads, Nq, -1)
        k = k.view(B, Nk, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B * self.num_heads, Nk, -1)
        v = v.view(B, Nk, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B * self.num_heads, Nk, -1)

        q = q * self.scale #缩放 query，防止后续 softmax 变动过大

        # 标准Attention
        if pfa_indices is None:
            attn = (q @ k.transpose(-2, -1))  # [B*H, Nq, Nk]

        # Sparse attention over previous progressive indices.
        else:
            topk = pfa_indices.shape[-1]
            smm_index = pfa_indices.view(B * self.num_heads, Nq, topk).long()
            k_selected = torch.gather(
                k.unsqueeze(1).expand(-1, Nq, -1, -1),
                dim=2,
                index=smm_index.unsqueeze(-1).expand(-1, -1, -1, k.size(-1))
            )
            attn = (q.unsqueeze(2) * k_selected).sum(dim=-1).view(B, self.num_heads, Nq, topk)
        attn = self.softmax(attn)

        # Apply Hadamard product for PFA and normalize.
        if pfa_values is not None:
                
                attn = (attn * pfa_values)
                attn = (attn + self.eps) / (attn.sum(dim=-1, keepdim=True) + self.eps)

        if self.topk < Nk:
            topk_values, topk_indices = torch.topk(attn, self.topk, dim=-1, largest=True, sorted=False)
            attn = topk_values
            if pfa_indices is not None:
                pfa_indices = pfa_indices.view(B, self.num_heads, Nq, topk)

                pfa_indices = torch.gather(pfa_indices, dim=-1, index=topk_indices)
            else:
                pfa_indices = topk_indices

        # Save the current attention results as PFA maps.
        pfa_values = attn
        
        # Aggregate values using the retained sparse indices or dense attention.
        if pfa_indices is None:
            x = (attn @ v) .transpose(1, 2).reshape(B, Nq, C)
        else:
            topk = pfa_indices.shape[-1]
            attn = attn.view(B * self.num_heads, Nq, topk)
            smm_index = pfa_indices.view(B * self.num_heads, Nq, topk).long()
            v_selected = torch.gather(
                v.unsqueeze(1).expand(-1, Nq, -1, -1),
                dim=2,
                index=smm_index.unsqueeze(-1).expand(-1, -1, -1, v.size(-1))
            )
            x = (attn.unsqueeze(-1) * v_selected).sum(dim=2).view(B, self.num_heads, Nq, self.head_dim).transpose(1, 2).reshape(B, Nq, C)
         
        x = self.proj(x)
        return x, pfa_values, pfa_indices

