import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))


class DBMG(nn.Module):
    def __init__(self, embed_dim_in: int = 512, embed_dim_out: int = 96):
        super().__init__()
        self.embed_dim_out = embed_dim_out
        self.proj_t_cls = nn.Linear(embed_dim_in, embed_dim_out)
        self.proj_d_cls = nn.Linear(embed_dim_in, embed_dim_out)
        self.proj_v_cls = nn.Linear(embed_dim_in, embed_dim_out)
        self.proj_t_tok = nn.Linear(embed_dim_in, embed_dim_out)
        self.proj_d_tok = nn.Linear(embed_dim_in, embed_dim_out)
        self.proj_v_patch = nn.Linear(embed_dim_in, embed_dim_out)
        self.tgl_q = nn.Linear(embed_dim_in, embed_dim_out)
        self.tgl_k = nn.Linear(embed_dim_in, embed_dim_out)
        self.tgl_v = nn.Linear(embed_dim_in, embed_dim_out)

        self.tgl_layer_norm = nn.LayerNorm(embed_dim_out)

    def _TGG(self, z_t_cls: torch.Tensor, z_d_cls: torch.Tensor) -> torch.Tensor:
        t = l2_normalize(self.proj_t_cls(z_t_cls), dim=-1)  # [B,H]
        d = l2_normalize(self.proj_d_cls(z_d_cls), dim=-1)  # [B,H]
        return torch.matmul(t, d.t())  # [B,B] cosine via dot of normalized

    def _TGL(self, Z_d_tok: torch.Tensor, Z_t_tok: torch.Tensor, z_d_cls: torch.Tensor) -> torch.Tensor:
        Q = self.tgl_q(Z_d_tok)  # [B, Ld, H]
        K = self.tgl_k(Z_t_tok)  # [B, Lt, H]
        V = self.tgl_v(Z_t_tok)  # [B, Lt, H]
        scores = torch.einsum("bqh,ckh->bcqk", Q, K) / math.sqrt(self.embed_dim_out)  # [B,B,Ld,Lt]
        attn = torch.softmax(scores, dim=-1)  # over Lt
        context = torch.einsum("bcqk,ckh->bcqh", attn, V)  # [B,B,Ld,H]
        context = context.mean(dim=2)  # [B,B,H]
        context = self.tgl_layer_norm(context)
        d_cls = l2_normalize(self.proj_d_cls(z_d_cls), dim=-1)  # [B,H]
        ctx = l2_normalize(context, dim=-1)                      # [B,B,H]
        return torch.einsum("ih,ijh->ij", d_cls, ctx)  # [B,B]

    def _TIB(self, z_t_cls, Z_t_tok, z_d_cls, Z_d_tok) -> torch.Tensor:
        S_TGG = self._TGG(z_t_cls, z_d_cls)                        # [B,B]
        S_TGL = self._TGL(Z_d_tok, Z_t_tok, z_d_cls)               # [B,B]
        return 0.5 * (S_TGG + S_TGL)

    def _CGG(self, z_t_cls: torch.Tensor, z_v_cls: torch.Tensor) -> torch.Tensor:
        t = l2_normalize(self.proj_t_cls(z_t_cls), dim=-1)  # [B,H]
        v = l2_normalize(self.proj_v_cls(z_v_cls), dim=-1)  # [B,H]
        return torch.matmul(t, v.t())  # [B,B]

    def _CLL(self, Z_t_tok: torch.Tensor, Z_v_patch: torch.Tensor) -> torch.Tensor:
        t_tok = l2_normalize(self.proj_t_tok(Z_t_tok), dim=-1)        # [B,Lt,H]
        v_pat = l2_normalize(self.proj_v_patch(Z_v_patch), dim=-1)    # [B,Lv,H]
        sim = torch.einsum("bth,cph->bctp", t_tok, v_pat)  # [B,B,Lt,Lv]
        return sim.amax(dim=(-1, -2))  # max over (t,p) -> [B,B]

    def _CAB(self, z_t_cls, Z_t_tok, z_v_cls, Z_v_patch) -> torch.Tensor:
        S_CGG = self._CGG(z_t_cls, z_v_cls)                 # [B,B]
        S_CLL = self._CLL(Z_t_tok, Z_v_patch)               # [B,B]
        return 0.5 * (S_CGG + S_CLL)

    def forward(
        self,
        z_d_cls: torch.Tensor,
        Z_d_tok: torch.Tensor,
        z_t_cls: torch.Tensor,
        Z_t_tok: torch.Tensor,
        z_v_cls: torch.Tensor,
        Z_v_patch: torch.Tensor,
    ) -> torch.Tensor:
        S_TIB = self._TIB(z_t_cls, Z_t_tok, z_d_cls, Z_d_tok)         # [B,B]
        S_CAB = self._CAB(z_t_cls, Z_t_tok, z_v_cls, Z_v_patch)       # [B,B]
        S = S_TIB + S_CAB
        return S
