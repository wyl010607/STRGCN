import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from .STRGCN_Layer import STRGCNLayer, EfficientSTRGCNLayer
from .TimeEmbedding import TimeEmbedding
from .IrrRevin import IrrRevIN


class STRGCNCLS(nn.Module):
    """
    STRGCN for irregular multivariate time-series **classification** (Hi-Patch style).

    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        hid_dim: int,
        n_layer: int,
        node_dim: int,
        dropout: float = 0.0,
        gcn_type: str = "cheb",
        gcn_depth: int | None = None,
        use_bias: bool = True,
        use_norm: bool = True,
        time_embedding_type: str = "rope",
        load_adj: bool = False,
        use_efficient_gcn: bool = False,
        use_revin: bool = False,
        revin_affine: bool = False,
        use_static: bool = False,
        static_in_dim: int | None = None,
        global_args: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        global_args = (global_args or {})
        self.task_name = global_args.get("task_name", "classification")

        self.c_in = int(c_in)
        self.c_out = int(c_out)
        self.hid_dim = int(hid_dim)
        self.n_layer = int(n_layer)
        self.num_nodes = int(global_args.get("num_of_vals"))
        self.dropout = float(dropout)

        self.use_static = bool(use_static)
        self.static_in_dim = static_in_dim
        self.Ns = (self.static_in_dim or 0) if self.use_static else 0
        self.num_nodes_ext  = self.num_nodes + self.Ns + 1

        if self.Ns > 0:
            self.static_value_proj = nn.Linear(1, self.hid_dim)
            self.static_type_emb = nn.Embedding(self.Ns, self.hid_dim)

        self.class_token = nn.Parameter(torch.zeros(1, 1, self.hid_dim))
        nn.init.trunc_normal_(self.class_token, std=0.02)

        self.use_efficient_gcn = bool(use_efficient_gcn)
        self.use_info_agg = bool(kwargs.get("use_info_agg", False))
        self.num_hyper_nodes_per_var = global_args.get("num_hyper_nodes_per_var", None)

        self.gcn_type = gcn_type
        self.gcn_depth = gcn_depth
        self.use_bias = use_bias
        self.use_norm = use_norm

        self.time_embedding_type = time_embedding_type

        self.use_revin = use_revin
        if self.use_revin:
            self.revin = IrrRevIN(num_vars=self.num_nodes, eps=1e-6, learn_affine=revin_affine)

        if load_adj:
            warnings.warn("Loading adjacency matrix is not implemented; using learnable/implicit structure instead.")

        self.encoder = nn.Linear(self.c_in, self.hid_dim)

        self.time_embedding = TimeEmbedding(self.hid_dim, time_embedding_type=self.time_embedding_type)

        GCNLayer = EfficientSTRGCNLayer if self.use_efficient_gcn else STRGCNLayer
        self.gcn_layers = nn.ModuleList(
            [
                GCNLayer(
                    in_dim=self.hid_dim,
                    out_dim=self.hid_dim,
                    node_dim=node_dim,
                    dropout=self.dropout,
                    num_nodes=self.num_nodes,
                    num_nodes_ext=self.num_nodes_ext,
                    num_hyper_nodes_per_var=self.num_hyper_nodes_per_var,
                    gcn_type=self.gcn_type,
                    gcn_depth=self.gcn_depth,
                    use_bias=self.use_bias,
                    use_norm=self.use_norm,
                    time_embedding_type=self.time_embedding_type,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(self.n_layer)
            ]
        )

        self.lambda_ = nn.Parameter(torch.tensor(0.5))


        self.cls_head = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hid_dim, self.c_out),
        )


    def forward(
        self,
        batch_value: torch.Tensor,         # (B, L)
        batch_timestamp: torch.Tensor,     # (B, L)
        batch_var_idx: torch.Tensor,       # (B, L) -> 0..V-1
        batch_pad_mask: torch.Tensor | None = None,  # (B, L)
        batch_agg_mask: torch.Tensor | None = None,  # (B, Lout, L) 
        batch_static: torch.Tensor | None = None,    # (B, D)   
        **kwargs,
    ) -> torch.Tensor:
        """
        返回 logits: (B, C)
        """
        B, L = batch_value.shape
        device = batch_value.device

        if batch_pad_mask is None:
            batch_pad_mask = torch.ones_like(batch_value, dtype=torch.float32, device=device)
        else:
            batch_pad_mask = batch_pad_mask.to(batch_value.dtype)

        if self.use_revin:
            pred_mask = torch.zeros_like(batch_pad_mask, dtype=batch_pad_mask.dtype)  # 分类无 pred mask
            x_norm, _ = self.revin.normalize(
                batch_value, batch_var_idx, pad_mask=batch_pad_mask, pred_mask=pred_mask
            )
        else:
            x_norm = batch_value

        x_emb = self.encoder(x_norm.unsqueeze(-1))                       # (B,L,H)
        t_emb = self.time_embedding(batch_timestamp, batch_pad_mask)     # (B,L,H)
        h_evt = x_emb + t_emb                                                # (B,L,H)

        static_nodes = []
        if self.Ns > 0:

            for j in range(self.Ns):
                v = batch_static[:, j:j + 1]  # (B,1)
                feat = self.static_value_proj(v) + self.static_type_emb.weight[j]  # (B,H)
                static_nodes.append(feat.unsqueeze(1))
            h_static = torch.cat(static_nodes, dim=1)  # (B,Ns,H)
        else:
            h_static = h_evt.new_zeros(h_evt.size(0), 0, h_evt.size(2))

        h_cls0 = self.class_token.expand(h_evt.size(0), -1, -1)  # (B,1,H)

        h = torch.cat([h_evt, h_static, h_cls0], dim=1)

        B, L = batch_value.shape
        L_ext = h.shape[1]
        vars_evt = batch_var_idx  # (B,L) in [0..V-1]
        vars_static = (torch.arange(self.Ns, device=h.device) +  self.num_nodes).view(1, -1).expand(B, -1)  # (B,Ns)
        vars_cls = torch.full((B, 1),  self.num_nodes + self.Ns, device=h.device,
                              dtype=vars_evt.dtype)
        vars_idx_ext = torch.cat([vars_evt, vars_static, vars_cls], dim=1)  # (B, L_ext)

        tt_evt = batch_timestamp  # (B,L)
        tt_extra = torch.zeros(B, L_ext - L, device=h.device, dtype=tt_evt.dtype)  # 常数0时间
        tt_ext = torch.cat([tt_evt, tt_extra], dim=1)  # (B,L_ext)

        pad_evt = batch_pad_mask.to(h.dtype)  # (B,L)
        pad_extra = torch.zeros(B, L_ext - L, device=h.device, dtype=h.dtype)  # 关键：静态/分类不参与时间平均
        pad_ext = torch.cat([pad_evt, pad_extra], dim=1)  # (B,L_ext)

        agg_ext = None
        if self.use_info_agg and (batch_agg_mask is not None):
            B, Lp, L0 = batch_agg_mask.shape
            ones_extra = torch.ones(B, Lp, L_ext - L0, device=h.device, dtype=batch_agg_mask.dtype)
            agg_ext = torch.cat([batch_agg_mask, ones_extra], dim=2)


        for gcn_layer in self.gcn_layers:
            h = gcn_layer(h, vars_idx_ext, t_emb=None, tt=tt_ext, pad_mask=pad_ext, agg_mask=agg_ext)
        #h = h * self.lambda_ + (1.0 - self.lambda_) * (x_emb + t_emb)

        h_cls = h[:, -1, :]  # (B,H)
        logits = self.cls_head(h_cls)
        return logits
