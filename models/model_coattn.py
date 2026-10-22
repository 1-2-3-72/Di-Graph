import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv

from .util import initialize_weights
from models.model_utils import BilinearFusion, SNN_Block
from nystrom_attention import NystromAttention
from models.Block import BasicBlock

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            residual=True,
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class Transformer_P(nn.Module):
    def __init__(self, feature_dim=512):
        super(Transformer_P, self).__init__()
        # Encoder
        self.pos_layer = PPEG(dim=feature_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features):
        H = features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([features, features[:, :add_length, :]], dim=1)  # [B, N, C]
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        h1 = self.layer1(h)  # [B, N, C]
        h_ = self.norm(h1)
        h = self.pos_layer(h1, _H, _W)  # [B, N, C]
        h2= self.layer2(h)  # [B, N, C]
        h2 = self.norm(h2)
        return h_[:, 0], h_[:, 1:], h2[:, 0], h2[:, 1:]

class Transformer_G(nn.Module):
    def __init__(self, feature_dim=512):
        super(Transformer_G, self).__init__()
        # Encoder
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features):
        cls_tokens = self.cls_token.expand(features.shape[0], -1, -1).cuda()
        h = torch.cat((cls_tokens, features), dim=1)
        h1 = self.layer1(h)  # [B, N, C]
        h_ = self.norm(h1)
        h2 = self.layer2(h1)  # [B, N, C]
        h2 = self.norm(h2)
        return h_[:, 0], h_[:, 1:], h2[:, 0], h2[:, 1:]

class AttentionEdgeWeight(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

    def forward(self, query, key):
        weights = F.normalize(query, p=2, dim=-1) @ F.normalize(key, p=2, dim=-1).transpose(-2, -1)
        return weights.squeeze(1)

class HeteroGraphFusion(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.cross_attn = AttentionEdgeWeight(embed_dim=hidden_dim)
        self.self_attn = AttentionEdgeWeight(embed_dim=hidden_dim)

        self.final_cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1, batch_first=True)

        self.layers = nn.ModuleList([
            HGTConv(in_channels=hidden_dim, out_channels=hidden_dim, metadata=(
                ['gene', 'patch'],
                [('gene', 'guides', 'patch'), ('gene', 'to', 'gene'), ('patch', 'to', 'patch')]
            ), heads=num_heads) for _ in range(num_layers)
        ])

    def forward(self, gene_tokens, patch_tokens):

        data = HeteroData()
        data['gene'].x = gene_tokens.squeeze(0)
        data['patch'].x = patch_tokens.squeeze(0)

        # Gene-patch cosine affinity.
        gp_weights = self.cross_attn(
            query=gene_tokens,
            key=patch_tokens
        )
        gp_weights = gp_weights.squeeze(0)

        # Patch-patch cosine affinity.
        pp_weights = self.self_attn(
            query=patch_tokens,
            key=patch_tokens
        )
        pp_weights = pp_weights.squeeze(0)

        # Gene-gene cosine affinity.
        gg_weights = self.self_attn(
            query=gene_tokens,
            key=gene_tokens
        )
        gg_weights = gg_weights.squeeze(0)

        # 基因->病理边 (guides)
        gene_idx, patch_idx = torch.where(gp_weights > 0)
        data['gene', 'guides', 'patch'].edge_index = torch.stack([gene_idx, patch_idx], dim=0)
        data['gene', 'guides', 'patch'].edge_weight = gp_weights[gene_idx, patch_idx]
        # 基因->基因边 (to)
        g1, g2 = torch.where(gg_weights > 0)
        data['gene', 'to', 'gene'].edge_index = torch.stack([g1, g2], dim=0)
        data['gene', 'to', 'gene'].edge_weight = gg_weights[g1, g2]

        # 病理->病理边 (to)
        p1, p2 = torch.where(pp_weights > 0)
        data['patch', 'to', 'patch'].edge_index = torch.stack([p1, p2], dim=0)
        data['patch', 'to', 'patch'].edge_weight = pp_weights[p1, p2]
        x_dict = {'gene': data['gene'].x, 'patch': data['patch'].x}
        edge_index_dict = data.edge_index_dict

        # 异构图Transformer
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)

        # 融合表示（全局平均）

        gene_repr = x_dict['gene'].unsqueeze(0)
        patch_repr = x_dict['patch'].unsqueeze(0)
        
        fused, weights = self.final_cross_attn(gene_repr, patch_repr, patch_repr)
        fusion_repr = fused.mean(dim=1)  
        return fusion_repr, gene_repr, patch_repr, gp_weights, pp_weights, gg_weights  # [1, 256]

class Teacher_G_HANet_Surv(nn.Module):
    def __init__(self, omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4, fusion = "concat", model_size_wsi="small",
                 model_size_omic="small", path_input_dim=1024, alpha_layers=3, teacher_topk=None):
        super().__init__()
        self.fusion = fusion
        self.n_classes = n_classes
        self.size_dict = {
            "pathomics": {"small": [path_input_dim, 256, 256], "large": [path_input_dim, 512, 256]},
            "genomics": {"small": [1024, 256], "large": [1024, 1024, 1024, 256]},
        }
        self.omic_sizes = omic_sizes

        hidden = self.size_dict["pathomics"][model_size_wsi]
        fc = []
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.25))
        self.pathomics_fc = nn.Sequential(*fc)

        hidden = self.size_dict["genomics"][model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.genomics_fc = nn.ModuleList(sig_networks)

        self.patch_assign = BasicBlock(
    dim=256,
    idx=0,
    layer_id=0,
    depth=alpha_layers,
    num_heads=4,
    num_topk=teacher_topk if teacher_topk is not None else ((0.7, 0.3, 0.2) if alpha_layers == 3 else (0.7, 0.5, 0.3, 0.2, 0.1)),
    mlp_ratio=4.,
    qkv_bias=True,
    norm_layer=nn.LayerNorm,
    downsample=None
)

        self.norm = nn.LayerNorm(256)

        self.path_global_encoder = Transformer_P(feature_dim=hidden[-1])
        self.omic_global_encoder = Transformer_G(feature_dim=hidden[-1])
        
        if self.fusion == "concat":
            self.mm = nn.Sequential(
                nn.Linear(3 * 256, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
            )
        elif self.fusion == "bilinear":
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        else:
            raise NotImplementedError
       
        self.hetero_fusion = HeteroGraphFusion(hidden_dim=256, num_heads=4, num_layers=2)
        self.classifier = nn.Linear(256, n_classes)
        self.apply(initialize_weights)
        
    def forward(self, **kwargs):
        return_intermediates = kwargs.get("return_intermediates", False)
        x_path = kwargs["x_path"]
        x_omic = [kwargs[f"x_omic{i}"] for i in range(1, 7)]

        genomics_features = [self.genomics_fc[idx](sig_feat) for idx, sig_feat in enumerate(x_omic)]
        genomics_features = torch.stack(genomics_features).unsqueeze(0)
        pathomics_features = self.pathomics_fc(x_path).unsqueeze(0)

        # pathomics encoder
        cls_token_pathomics_encoder1, patch_token_pathomics_encoder1, cls_token_pathomics_encoder2, patch_token_pathomics_encoder2 = self.path_global_encoder(
            pathomics_features)  # cls token + patch tokens
        # genomics encoder
        cls_token_genomics_encoder1, patch_token_genomics_encoder1, cls_token_genomics_encoder2, patch_token_genomics_encoder2 = self.omic_global_encoder(
            genomics_features)  # cls token + patch tokens

        pfa_list = [None, None]
        params = {}

        if return_intermediates:
            out_x, pfa_list, pfa_intermediates = self.patch_assign(
                patch_token_genomics_encoder2, patch_token_pathomics_encoder2, pfa_list, params, return_intermediates=True)
        else:
            out_x, pfa_list = self.patch_assign(patch_token_genomics_encoder2,patch_token_pathomics_encoder2, pfa_list, params)
            pfa_intermediates = None

        pfa_values, pfa_indices = pfa_list[0], pfa_list[1]

        pfa_values = F.softmax(pfa_values, dim=-1)
        pfa_values_merged = pfa_values.mean(dim=1)[0] # [6, topk]
        pfa_indices = pfa_indices[0, 0]  # 选第一个 head

        patch_token_pathomics_encoder2_expanded = patch_token_pathomics_encoder2.expand(6, -1, -1)# [6, n, 256]
        indices_expanded = pfa_indices.unsqueeze(-1).expand(-1, -1, 256)  # [6, topk, 256]

        assign_path = torch.gather(patch_token_pathomics_encoder2_expanded, dim=1, index=indices_expanded)
        grouped_path_feats= (assign_path * pfa_values_merged.unsqueeze(-1)).sum(dim=1).unsqueeze(0)  # [1, 6, 256]

        gene_tokens_with_cls = torch.cat([cls_token_genomics_encoder2.unsqueeze(1), out_x], dim=1)
        patch_tokens_with_cls = torch.cat([cls_token_pathomics_encoder2.unsqueeze(1), grouped_path_feats], dim=1)
        
        fusion, gene_repr, patch_repr, gp_weights, pp_weights, gg_weights = self.hetero_fusion(
            gene_tokens=gene_tokens_with_cls,      # shape: [1, 7, 256]
            patch_tokens=patch_tokens_with_cls     # shape: [1, 7, 256]
        )

        if self.fusion == "concat":
            fusion = self.mm(torch.cat([fusion, cls_token_pathomics_encoder2, cls_token_genomics_encoder2], dim=1))
        elif self.fusion == "bilinear":
            fusion = self.mm(fusion, torch.cat([cls_token_pathomics_encoder2, cls_token_genomics_encoder2], dim=1))

        fusion = self.norm(fusion)
        logits = self.classifier(fusion)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        if return_intermediates:
            focus_dict = {
                "pfa_intermediates": pfa_intermediates,
                "pfa_values": pfa_values,
                "pfa_indices": pfa_indices,
                "gene_tokens_with_cls": gene_tokens_with_cls,
                "gene_repr": gene_repr,
                "patch_repr": patch_repr,
                "gp_weights": gp_weights,
                "pp_weights": pp_weights,
                "gg_weights": gg_weights,
            }
            return hazards, S, logits, pfa_values, gene_tokens_with_cls, fusion, gene_repr, patch_repr, gp_weights, pp_weights, gg_weights, focus_dict

        return hazards, S, logits, pfa_values, gene_tokens_with_cls, fusion, gene_repr, patch_repr, gp_weights, pp_weights, gg_weights

class GenePredictor(nn.Module):
    """将 bag‑level 全局向量投射为 6 个伪基因 token"""

    def __init__(self, in_dim: int = 256, hid: int = 128, out_dim: int = 256, k: int = 6):
        super().__init__()
        self.k = k
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, k * out_dim),
            nn.LayerNorm(k * out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim) → return (B, k, out_dim)"""
        b = x.size(0)
        out = self.fc(x).view(b, self.k, -1)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=512, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Student_G_HANet_Surv(nn.Module):
    def __init__(self, n_classes: int = 4, model_size_wsi: str = "small", dropout: float = 0.25, path_input_dim: int = 1024, alpha_layers: int = 3, student_topk=None):
        super().__init__()
        self.n_classes = n_classes
        self.size_dict = {
            "pathomics": {"small": [path_input_dim, 256, 256], "large": [path_input_dim, 512, 256]},
        }
        hidden = self.size_dict["pathomics"][model_size_wsi]

        # WSI feature encoder.
        fc = []
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.25))
        self.pathomics_fc = nn.Sequential(*fc)
        from models.model_coattn import Transformer_P
        self.pathomics_encoder = Transformer_P(feature_dim=hidden[-1])

        # Gene predictor and gene CLS token.
        self.gene_pred = GenePredictor(in_dim=hidden[-1], hid=128, out_dim=hidden[-1], k=6)

        self.path_assign_s = BasicBlock(
    dim=256,
    idx=0,
    layer_id=0,
    depth=alpha_layers,
    num_heads=4,
    num_topk=student_topk if student_topk is not None else ((0.6, 0.4, 0.2) if alpha_layers == 3 else (0.6, 0.5, 0.4, 0.2, 0.1)),
    mlp_ratio=4.,
    qkv_bias=True,
    norm_layer=nn.LayerNorm,
    downsample=None
)

        self.hetero_fusion = HeteroGraphFusion(hidden_dim=256, num_heads=4, num_layers=2)

        self.ffn = FeedForward(dim=hidden[-1], dropout=dropout)
        self.ln1 = nn.LayerNorm(hidden[-1])
        self.classifier = nn.Linear(hidden[-1], n_classes)

    def forward(self, **kwargs):

        x_path = kwargs["x_path"]

        pathomics_features = self.pathomics_fc(x_path).unsqueeze(0)
        cls_token_pathomics_encoder1, patch_token_pathomics_encoder1, cls_token_pathomics_encoder2, patch_token_pathomics_encoder2 = self.pathomics_encoder(pathomics_features)

        # Predict six gene tokens and form the gene CLS token.
        global_p = patch_token_pathomics_encoder2.mean(dim=1)          # (B,256)
        pseudo_gene_tok = self.gene_pred(global_p)      # (B,6,256)
        gene_cls_tok = global_p.unsqueeze(1)            # (B,1,256)
        
        # Assign patch features to predicted gene tokens.
        pfa_list = [None, None]
        params = {}

        out_x_s, pfa_list_s = self.path_assign_s(pseudo_gene_tok, patch_token_pathomics_encoder2,pfa_list, params)  
        pfa_values_s, pfa_indices_s = pfa_list_s[0], pfa_list_s[1]

        pfa_values_s = F.softmax(pfa_values_s, dim=-1)
        pfa_values_merged_s = pfa_values_s.mean(dim=1)[0] # [6, topk]
        pfa_indices_s = pfa_indices_s[0, 0]  # 选第一个 head

        patch_token_pathomics_encoder2_expanded_s = patch_token_pathomics_encoder2.expand(6, -1, -1)# [6, n, 256]
        indices_expanded_s = pfa_indices_s.unsqueeze(-1).expand(-1, -1, 256)  # [6, topk, 256]
    
        assign_path_s = torch.gather(patch_token_pathomics_encoder2_expanded_s, dim=1, index=indices_expanded_s)
        grouped_path_feats_s= (assign_path_s * pfa_values_merged_s.unsqueeze(-1)).sum(dim=1).unsqueeze(0)  # [1, 6, 256]

        anchor_tokens = torch.cat([gene_cls_tok, out_x_s], dim=1)  # (B,7,256)

        grouped_patch_feats = torch.cat([cls_token_pathomics_encoder2.unsqueeze(1), grouped_path_feats_s], dim=1)  # (B,7,256)

        fusion_s, gene_repr_s, patch_repr_s, ap_weights, pp_weights, aa_weights = self.hetero_fusion(
            gene_tokens=anchor_tokens,      # shape: [1, 7, 256]
            patch_tokens=grouped_patch_feats     # shape: [1, 7, 256]
        )

        final_feature = self.ffn(self.ln1(fusion_s)) + fusion_s
        logits = self.classifier(final_feature)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        return (hazards, S, logits, pfa_values_s, anchor_tokens, fusion_s, gene_repr_s, patch_repr_s, ap_weights, pp_weights, aa_weights)

