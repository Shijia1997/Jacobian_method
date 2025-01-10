import torch
import torch.nn as nn
import torch.nn.functional as F
from literature_models import J_CNN3DModel, J_CNN3DEncoder

class CrossAttention3D(nn.Module):
    """
    Cross-attention with a toggle for which input (MRI or JSM) forms Q
    and which forms K, V.

    By default:
      - Q from MRI features
      - K, V from JSM features

    Alternatively:
      - Q from JSM features
      - K, V from MRI features

    We assume embed_dim == C for simplicity, or do extra linear layers.
    """
    def __init__(self, embed_dim, num_heads=4, qkv_mode='mri2jsm'):
        """
        Args:
          embed_dim: dimension for Q/K/V
          num_heads: number of attention heads
          qkv_mode: 'mri2jsm' means Q = MRI, K = V = JSM
                    'jsm2mri' means Q = JSM, K = V = MRI
        """
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.qkv_mode = qkv_mode

        # We can rely on nn.MultiheadAttention
        # batch_first=True => expects [B, T, E]
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Optional linear layers to project Q, K, V from dimension 64 to embed_dim
        self.query_proj = nn.Linear(64, embed_dim)
        self.key_proj   = nn.Linear(64, embed_dim)
        self.value_proj = nn.Linear(64, embed_dim)

    def forward(self, mri_feats, jsm_feats):
        """
        Inputs:
          mri_feats: [B, C, D, H, W]
          jsm_feats: [B, C, D, H, W]
        Output:
          attended_feats: [B, embed_dim, D, H, W]
          attn_weights:   [B, num_heads, N, N] (if needed)
        """
        B, C, D, H, W = mri_feats.shape

        # Flatten spatial => N = D*H*W, so shapes become [B, N, C]
        mri_2d = mri_feats.view(B, C, -1).transpose(1, 2)  # [B, N, C]
        jsm_2d = jsm_feats.view(B, C, -1).transpose(1, 2)  # [B, N, C]

        # Decide who is Q vs K, V based on qkv_mode
        if self.qkv_mode == 'mri2jsm':
            # Q from MRI, K/V from JSM
            Q = self.query_proj(mri_2d)  # [B, N, embed_dim]
            K = self.key_proj(jsm_2d)    # [B, N, embed_dim]
            V = self.value_proj(jsm_2d)  # [B, N, embed_dim]
        else:  # 'jsm2mri'
            # Q from JSM, K/V from MRI
            Q = self.query_proj(jsm_2d)  # [B, N, embed_dim]
            K = self.key_proj(mri_2d)    # [B, N, embed_dim]
            V = self.value_proj(mri_2d)  # [B, N, embed_dim]

        # Cross-attention => [B, N, embed_dim]
        attended, attn_weights = self.attn(Q, K, V)

        # Reshape back to [B, embed_dim, D, H, W]
        attended = attended.transpose(1, 2).view(B, self.embed_dim, D, H, W)
        return attended, attn_weights


class CrossAttention3DClassifier(nn.Module):
    """
    3D classification network with:
      - 3D encoders for MRI & JSM (each outputs [B, 64, D', H', W'])
      - CrossAttention3D => [B, embed_dim, D', H', W']
      - Two fusion options:
         1) 'attend_only': classify the attention output alone
         2) 'concat': concatenate attention output with original MRI feats
      - A final linear classifier
    """
    def __init__(self,
                 fusion_type="attend_only",  # or "concat"
                 num_heads=4,
                 num_classes=2,
                 qkv_mode='mri2jsm'):
        """
        Args:
          fusion_type: 'attend_only' or 'concat'
          num_heads: number of attention heads
          num_classes: classifier output dimension
          qkv_mode: see CrossAttention3D docstring for usage
        """
        super().__init__()

        # We assume each encoder outputs 64 channels
        if fusion_type == "concat":
            in_features = 64 * 2
            embed_dim = 64
        else:  # "attend_only"
            # You might want to keep the dimension consistent
            # with the cross-attention embed_dim if you plan to do something else
            # For clarity, we set in_features = 128 if we want the final classifier
            # to see 128 channels (like a bigger projection).
            in_features = 128
            embed_dim = 128

        self.mri_encoder = J_CNN3DEncoder(input_channels=1, base_channels=64)
        self.jsm_encoder = J_CNN3DEncoder(input_channels=1, base_channels=64)

        # Cross-attention with a new `qkv_mode` argument
        # By default embed_dim=64 if you want exact matching with encoders
        self.cross_attn = CrossAttention3D(embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           qkv_mode=qkv_mode)

        self.fusion_type = fusion_type

        # Classifier
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, mri, jsm):
        """
        Args:
          mri, jsm: [B, 1, D, H, W]
        Returns:
          logits: [B, num_classes]
        """
        # 1) Encode => [B, 64, D', H', W']
        mri_feats = self.mri_encoder(mri)
        jsm_feats = self.jsm_encoder(jsm)

        # 2) Cross-Attention => [B, embed_dim, D', H', W']
        attended, _ = self.cross_attn(mri_feats, jsm_feats)

        # 3) Fusion
        if self.fusion_type == "concat":
            # cat along channel dimension => [B, 128, D', H', W']
            fused_feats = torch.cat((mri_feats, attended), dim=1)
        else:
            # "attend_only": just use attention output
            fused_feats = attended

        # 4) Global average pool => shape [B, in_features]
        pooled = F.adaptive_avg_pool3d(fused_feats, (1, 1, 1)).view(fused_feats.size(0), -1)

        # 5) Classify
        logits = self.classifier(pooled)
        return logits
