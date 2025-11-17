import math
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from .TimeEmbedding import TimeEmbedding
from .Function import temperature_softmax

def masked_softmax(logits, mask=None, dim=-1, eps=1e-9):
    if mask is None:
        max_per_row = torch.amax(logits, dim=dim, keepdim=True)
        exps = torch.exp(logits - max_per_row)
        denom = exps.sum(dim=dim, keepdim=True).clamp_min(eps)
        return exps / denom

    mask = mask.to(torch.bool)
    masked_logits = logits.masked_fill(~mask, float('-inf'))
    max_per_row = torch.amax(masked_logits, dim=dim, keepdim=True)
    max_per_row = torch.where(torch.isfinite(max_per_row), max_per_row, torch.zeros_like(max_per_row))
    exps = torch.exp(masked_logits - max_per_row) * mask.to(logits.dtype)
    denom = exps.sum(dim=dim, keepdim=True).clamp_min(eps)
    probs = exps / denom
    return probs

class STRGCNLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        node_dim,
        dropout,
        num_node,
        gcn_type="cheb",
        gcn_depth=None,
        use_bias=True,
        use_norm=True,
        *args,
        **kwargs,
    ):
        super(STRGCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.gcn_depth = gcn_depth
        self.gcn_type = gcn_type  # cheb or gcn
        if gcn_type == "gcn":
            if gcn_depth is not None:
                warnings.warn(
                    "gcn_depth is not used in gcn_type='gcn', set gcn_depth=1."
                )
            self.gcn_depth = 1
        self.weight_T_Embedding = TimeEmbedding(
            in_dim, time_embedding_type="transformer"
        )
        self.weight_S_Embedding = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=num_node, embedding_dim=out_dim)
                for _ in range(gcn_depth)
            ]
        )
        self.bias = nn.Parameter(torch.FloatTensor(out_dim)) if use_bias else None
        self.lambda_ = nn.Parameter(torch.FloatTensor([1]))
        self.dropout_layer = nn.Dropout(dropout)
        self.use_norm = use_norm
        self.layer_norm = nn.LayerNorm(out_dim) if use_norm else None

    def forward(self, x, support, vars_idx, t_emb, tt, agg_mask=None):
        # x:(B, L, D); support: (L, L); vars_idx: (L); t_emb: (B, L, D)
        # distance_mx = torch.einsum("bld,bld->bll", t_emb, t_emb) # 注意这里的分布问题，是否应该除以根号l
        distance_mx = torch.einsum("bld,bmd->blm", t_emb, t_emb)
        distance_mx = distance_mx / (distance_mx.sum(dim=2, keepdim=True) + 1e-8)
        # distance_mx = temperature_softmax(distance_mx, dim=2, temperature=1.0, mask_val=0.0)
        adj_mx = distance_mx * support  # (B, L, L)
        adj_mx = adj_mx / (adj_mx.sum(dim=2, keepdim=True) + 1e-8)
        # adj_mx = temperature_softmax(adj_mx, dim=2, temperature=1.0, mask_val=0.0)
        weight_T = self.weight_T_Embedding(tt)  # (B, L, in_dim)
        weight_S_ = [
            self.weight_S_Embedding[v](vars_idx) for v in range(self.gcn_depth)
        ]
        weight_S = torch.stack(weight_S_, dim=0)  # (gcn_depth, L, out_dim)
        # weight = torch.einsum("kld,klq->kldq", weight_U, weight_V)
        if self.gcn_type == "cheb":
            out = low_rank_cheb_graph_conv(
                x, adj_mx, weight_T, weight_S, self.gcn_depth, self.bias
            )
        elif self.gcn_type == "gcn":
            out = low_rank_graph_conv(x, adj_mx, weight_T, weight_S, self.bias)
        else:
            raise ValueError(f"gcn_type={self.gcn_type} is not supported.")
        out = F.relu(out)
        out = self.dropout_layer(out) * self.lambda_ + x * (1 - self.lambda_)
        if self.use_norm:
            out = self.layer_norm(out)
        return out


class EfficientSTRGCNLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        node_dim,
        dropout,
        num_nodes,
        num_nodes_ext,
        num_hyper_nodes_per_var=12,
        gcn_type="cheb",
        gcn_depth=None,
        use_bias=True,
        use_norm=True,
        time_embedding_type="rbf",
        layer_idx=0,
        #temperature=1.0,
        *args,
        **kwargs,
    ):
        super(EfficientSTRGCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes
        self.num_nodes_ext = num_nodes_ext
        self.num_hyper_nodes_per_var = num_hyper_nodes_per_var
        self.dropout = dropout
        self.gcn_depth = gcn_depth
        self.gcn_type = gcn_type  # cheb or gcn
        #self.temperature = temperature
        self.time_embedding_type = time_embedding_type
        self.layer_idx = layer_idx
        self.learnable_adj_weight_U = nn.Embedding(
            num_embeddings=num_nodes_ext, embedding_dim=node_dim
        )
        self.learnable_adj_weight_V = nn.Embedding(
            num_embeddings=num_nodes_ext, embedding_dim=node_dim
        )
        self.time_embedding = TimeEmbedding(in_dim, time_embedding_type=time_embedding_type)
        self.bottom_weight_S_Embedding = nn.Embedding(
            num_embeddings=num_nodes_ext, embedding_dim=in_dim
        )
        self.middle_weight_S_Embedding = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=num_nodes_ext, embedding_dim=out_dim)
                for _ in range(gcn_depth)
            ]
        )
        self.top_weight_S_Embedding = nn.Embedding(
            num_embeddings=num_nodes_ext, embedding_dim=out_dim
        )
        self.bottom_bias = (
            nn.Parameter(torch.zeros(out_dim)) if use_bias else None
        )
        self.middle_bias = (
            nn.Parameter(torch.zeros(out_dim)) if use_bias else None
        )
        self.top_bias = nn.Parameter(torch.zeros(out_dim)) if use_bias else None
        self.lambda_ = nn.Parameter(torch.tensor(0.5))
        self.dropout_layer = nn.Dropout(dropout)
        self.use_norm = use_norm
        self.layer_norm = nn.LayerNorm(out_dim) if use_norm else None

        self.bottom_weight_T = nn.Linear(in_dim, out_dim)
        self.middle_weight_T = nn.Linear(out_dim, out_dim)
        self.top_weight_T = nn.Linear(out_dim, out_dim)

        # layer learnable scales and temps
        self.bottom_val_scale  = nn.Parameter(torch.tensor(1.0))  # for structural logits
        self.bottom_time_scale = nn.Parameter(torch.tensor(1.0))  # for temporal  logits
        self.bottom_val_temp   = nn.Parameter(torch.tensor(1.0))
        self.bottom_time_temp  = nn.Parameter(torch.tensor(1.0))

        self.middle_val_scale  = nn.Parameter(torch.tensor(1.0))
        self.middle_time_scale = nn.Parameter(torch.tensor(1.0))
        self.middle_val_temp   = nn.Parameter(torch.tensor(1.0))
        self.middle_time_temp  = nn.Parameter(torch.tensor(1.0))

        self.top_val_scale     = nn.Parameter(torch.tensor(1.0))
        self.top_time_scale    = nn.Parameter(torch.tensor(1.0))
        self.top_val_temp      = nn.Parameter(torch.tensor(1.0))
        self.top_time_temp     = nn.Parameter(torch.tensor(1.0))

    @torch.no_grad()
    def hyper_times_from_mask(
        self,
        tt: torch.Tensor,         # [B, L]
        pad_mask: torch.Tensor,   # [B, L], 1=valid, 0=pad
        agg_mask: torch.Tensor,   # [B, L', L]
        eps: float = 1e-8,
    ) -> torch.Tensor:
        B, L = tt.shape
        device = tt.device
        if agg_mask is None:
            tt_reg = torch.linspace(
                torch.masked_select(tt, tt != -1).min(),
                tt.max(),
                self.num_hyper_nodes_per_var,
                requires_grad=False,
                device=device,
            ).unsqueeze(0).repeat(B, self.num_nodes)
            return tt_reg

        Lp = agg_mask.shape[1]
        w = agg_mask * pad_mask.unsqueeze(1).to(agg_mask.dtype)  # [B, L', L]
        num = torch.einsum("bjl,bl->bj", w, tt)                 # [B,L']
        den = w.sum(dim=-1).clamp_min(eps)                      # [B,L']
        tt_reg = num / den                                      # [B,L']
        empty = (den <= eps)                                    # [B,L']
        if empty.any():
            tt_min = torch.where(pad_mask.bool(), tt, torch.full_like(tt, float("inf"))).amin(dim=1, keepdim=True)
            tt_max = torch.where(pad_mask.bool(), tt, torch.full_like(tt, float("-inf"))).amax(dim=1, keepdim=True)
            tt_min = torch.where(torch.isfinite(tt_min), tt_min, torch.zeros_like(tt_min))
            tt_max = torch.where(torch.isfinite(tt_max), tt_max, tt_min + 1.0)
            base = torch.linspace(0.0, 1.0, steps=Lp, device=device).unsqueeze(0)  # [1,L']
            uni = tt_min + (tt_max - tt_min) * base                               # [B,L']
            tt_reg = torch.where(empty, uni, tt_reg)
        return tt_reg

    def forward(self, x, vars_idx, t_emb, tt, pad_mask, agg_mask=None):
        if (self.layer_idx != 0):
            agg_mask = None
        device = x.device
        tt_reg = self.hyper_times_from_mask(tt, pad_mask, agg_mask)  # [B, L']
        vars_idx_reg = torch.arange(self.num_nodes, device=device).repeat_interleave(
            self.num_hyper_nodes_per_var
        )
        # t_emb
        t_emb = self.time_embedding(tt, pad_mask)  # (B, L, in_dim)
        t_emb_reg = self.time_embedding(tt_reg)

        # bottom layer
        bottom_adj_U = self.learnable_adj_weight_U(vars_idx_reg)
        bottom_adj_V = self.learnable_adj_weight_V(vars_idx)
        bottom_support = torch.einsum(
            "md,bld->bml", bottom_adj_U, bottom_adj_V
        ) / math.sqrt(bottom_adj_U.shape[-1]) * self.bottom_val_scale  # (B, L', L)
        #bottom_support = bottom_support / bottom_support.sum(dim=-1, keepdim=True)
        bottom_distance_mx = torch.einsum("bld,bmd->blm", t_emb_reg, t_emb) * self.bottom_time_scale
        bottom_adj_mx = self.bottom_val_temp * bottom_support + self.bottom_time_temp * bottom_distance_mx
        bottom_adj_mx = masked_softmax(bottom_adj_mx, agg_mask, dim=-1)
        #bottom_weight_T = self.time_embedding(tt, pad_mask)  # (B, L, in_dim) 这里应该加一个可学习的东西，因为这个timeemb又要学距离度量，又要学表的的信息，比较难
        bottom_weight_T = self.bottom_weight_T(t_emb)  # (B, L, in_dim)
        bottom_weight_S = self.bottom_weight_S_Embedding(vars_idx)  # (B, L, in_dim)

        # middle layer
        middle_adj_V = self.learnable_adj_weight_V(vars_idx_reg)
        middle_support = torch.einsum(
            "md,ld->ml", bottom_adj_U, middle_adj_V #note 这里可能要换成带B的
        ) * self.middle_val_scale # (L', L')
        middle_distance_mx = torch.einsum("bld,bmd->blm", t_emb_reg, t_emb_reg) * self.middle_time_scale # (L', L')
        middle_adj_mx = F.softmax(self.middle_val_temp * middle_support + self.middle_time_temp * middle_distance_mx, dim=-1)

        #middle_weight_T = self.time_embedding(tt_reg)  # (L', in_dim)
        middle_weight_T = self.middle_weight_T(t_emb_reg)  # (B, L', in_dim)
        middle_weight_S_ = [
            self.middle_weight_S_Embedding[v](vars_idx_reg)
            for v in range(self.gcn_depth)
        ]
        middle_weight_S = torch.stack(
            middle_weight_S_, dim=0
        )  # (gcn_depth, L', out_dim)

        # top layer
        top_adj_U = self.learnable_adj_weight_U(vars_idx)
        top_support = torch.einsum(
            "bld,md->blm", top_adj_U, middle_adj_V # (B, L, L')
        ) * self.top_val_scale # (L', L')
        top_distance_mx = torch.einsum("bld,bmd->blm", t_emb, t_emb_reg) * self.top_time_scale # (B, L, L')
        top_adj_mx = self.top_val_temp * top_support + self.top_time_temp * top_distance_mx
        top_adj_mx = masked_softmax(top_adj_mx, agg_mask.transpose(1, 2) if agg_mask is not None else None, dim=-1)

        #top_weight_T = middle_weight_T
        top_weight_T = self.top_weight_T(t_emb_reg)  # (L', in_dim)
        top_weight_S = self.top_weight_S_Embedding(vars_idx_reg)  # (L', out_dim)

        # gcn
        bottom_out = low_rank_graph_conv(
            x, bottom_adj_mx, bottom_weight_T, bottom_weight_S, self.bottom_bias
        )
        bottom_out = F.relu(bottom_out)
        if self.use_norm:
            bottom_out = self.layer_norm(bottom_out)

        middle_out = low_rank_cheb_graph_conv(
            bottom_out,
            middle_adj_mx,
            middle_weight_T,
            middle_weight_S,
            self.gcn_depth,
            self.middle_bias,
        )
        middle_out = F.relu(middle_out) #(B, L, c_out)
        if self.use_norm:
            middle_out = self.layer_norm(middle_out)

        top_out = low_rank_graph_conv(
            middle_out, top_adj_mx, top_weight_T, top_weight_S, self.top_bias
        )

        out = F.relu(top_out)
        out = self.dropout_layer(out) * self.lambda_ + x * (1 - self.lambda_)
        if self.use_norm:
            out = self.layer_norm(out)

        return out


def low_rank_cheb_graph_conv(h, adj_mx, weight_T, weight_S, Ks, bias=None):
    """
    Forward pass of the Chebyshev Graph Convolution layer with step-by-step computation.

    Parameters
    ----------
    h : torch.Tensor
        Input tensor of shape (B, L, c_in), where:
            B is the batch size,
            L is the number of nodes,
            c_in is the number of input features.
    adj_mx : torch.Tensor
        Adjacency matrix of the graph, with shape (L, L). or (B, L, L)
    weight_T : torch.Tensor
        Weight tensor with shape (B, L, c_in) or (L, c_in), where:
            B is the batch size,
            L is the number of nodes,
            c_in is the number of input features.
    weight_S : torch.Tensor
        Weight tensor with shape (K, L, c_out), where:
            K is the number of Chebyshev polynomials,
            L is the number of nodes,
            c_out is the number of output features.
    Ks : int
        Order of the Chebyshev polynomial.
    bias : torch.Tensor or None, optional
        Bias tensor with shape (c_out), if bias is not None. Default is None.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (B, L, c_out), where:
            B is the batch size,
            L is the number of nodes,
            c_out is the number of output features.

    """
    assert (len(weight_S.shape) == 3) and (
        weight_S.shape[0] == Ks
    ), "Weight_S tensor must have shape (K, L, c_out)"
    #weight_T = torch.ones_like(weight_T)
    #weight_S = torch.ones_like(weight_S)
    # Initialize Chebyshev polynomials
    T_k_minus_2 = h  # T_0(x) = x (B, L, c_in)
    if len(weight_T.shape) == 2: #(L, c_in)
        s = torch.einsum("bli,li->bl", T_k_minus_2, weight_T)  # Contribution from T_0
    else:
        s = torch.einsum("bli,bli->bl", T_k_minus_2, weight_T)
    out = torch.einsum("bl,lo->blo", s, weight_S[0])

    if Ks > 1:
        # Compute T_1(x)
        if len(adj_mx.shape) == 2:  # Shared adjacency matrix
            T_k_minus_1 = torch.einsum("ij,bjd->bid", adj_mx, h)
        else:  # Batch-specific adjacency matrix
            T_k_minus_1 = torch.einsum("bij,bjd->bid", adj_mx, h)
        if len(weight_T.shape) == 2:
            s = torch.einsum("bli,li->bl", T_k_minus_1, weight_T)
        else:
            s = torch.einsum(
                "bli,bli->bl", T_k_minus_1, weight_T
            )  # Contribution from T_1
        out += torch.einsum("bl,lo->blo", s, weight_S[1])

    # Iteratively compute higher-order Chebyshev polynomials and contributions
    for k in range(2, Ks):
        if len(adj_mx.shape) == 2:  # Shared adjacency matrix
            T_k = 2 * torch.einsum("ij,bjd->bid", adj_mx, T_k_minus_1) - T_k_minus_2
        else:  # Batch-specific adjacency matrix
            T_k = 2 * torch.einsum("bij,bjd->bid", adj_mx, T_k_minus_1) - T_k_minus_2

        # Add contribution from
        if len(weight_T.shape) == 2:
            s = torch.einsum("bli,li->bl", T_k, weight_T)
        else:
            s = torch.einsum("bli,bli->bl", T_k, weight_T)
        out += torch.einsum("bl,lo->blo", s, weight_S[k])

        # Update for next iteration
        T_k_minus_2 = T_k_minus_1
        T_k_minus_1 = T_k

    # Add bias if present
    if bias is not None:
        assert len(bias.shape) == 1, "Bias tensor must have shape (c_out)"
        out += bias

    return out


def low_rank_graph_conv(h, adj_mx, weight_T, weight_S, bias=None):
    """
    Forward pass of the Graph Convolution layer.
    Parameters
    ----------
    h : torch.Tensor
        Input tensor of shape (B, L, c_in), where:
            B is the batch size,
            L is the number of nodes,
            c_in is the number of input features.
    adj_mx : torch.Tensor
        Adjacency matrix of the graph with shape (B, L) or (B, L, L), where:
            B is the batch size,
            L is the number of nodes.
    weight_T : torch.Tensor
        Weight tensor with shape (B, L, c_in) or (L, c_in), where:
            B is the batch size,
            L is the number of nodes,
            c_in is the number of input features.
    weight_S : torch.Tensor
        Weight tensor with shape (B, L, c_out) or (L, c_out), where:
            L is the number of nodes,
            c_out is the number of output features.
    Returns
    -------
    torch.Tensor
        Output tensor of shape (B, L, c_out), where:
            B is the batch size,
            L is the number of nodes,
            c_out is the number of output features.
    """
    #if len(weight_S.shape) == 3:
    #    weight_S = weight_S.squeeze(0)

    # S has shape (B, L)
    if len(weight_T.shape) == 2:
        S = torch.einsum("bni,ni->bn", h, weight_T)
    else:
        S = torch.einsum("bni,bni->bn", h, weight_T)

    # hw has shape (B, L, c_out)
    if len(weight_S.shape) == 2:
        hw = torch.einsum("bn,no->bno", S, weight_S)
    else:
        hw = torch.einsum("bn,bno->bno", S, weight_S)

    # Shape of out: (B, L, c_out)
    out = torch.einsum("bmn,bno->bmo", adj_mx, hw)

    # Add bias if present
    if bias is not None:
        out += bias
    return out


def cheb_graph_conv(h, adj_mx, weight, Ks, bias=None):
    """
    Forward pass of the Chebyshev Graph Convolution layer with step-by-step computation.

    Parameters
    ----------
    h : torch.Tensor
        Input tensor of shape (B, L, c_in), where:
            B is the batch size,
            L is the number of nodes,
            c_in is the number of input features.
    adj_mx : torch.Tensor
        Adjacency matrix of the graph, with shape (L, L). or (B, L, L)
    weight : torch.Tensor
        Weight tensor with shape (K, c_in, c_out), where:
            K is the number of Chebyshev polynomials,
            c_in is the number of input features,
            c_out is the number of output features.
    Ks : int
        Order of the Chebyshev polynomial.
    bias : torch.Tensor or None, optional
        Bias tensor with shape (c_out), if bias is not None. Default is None.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (B, L, c_out), where:
            B is the batch size,
            L is the number of nodes,
            c_out is the number of output features.

    """
    assert len(weight.shape) == 3, "Weight tensor must have shape (K, c_in, c_out)"
    assert weight.shape[0] == Ks, "Weight tensor must have shape (K, c_in, c_out)"

    # Initialize Chebyshev polynomials
    T_k_minus_2 = h  # T_0(x) = x
    out = torch.einsum("bld,do->blo", T_k_minus_2, weight[0])  # Contribution from T_0

    if Ks > 1:
        # Compute T_1(x)
        if len(adj_mx.shape) == 2:  # Shared adjacency matrix
            T_k_minus_1 = torch.einsum("ij,bjd->bid", adj_mx, h)
        else:  # Batch-specific adjacency matrix
            T_k_minus_1 = torch.einsum("bij,bjd->bid", adj_mx, h)
        out += torch.einsum(
            "bld,do->blo", T_k_minus_1, weight[1]
        )  # Contribution from T_1

    # Iteratively compute higher-order Chebyshev polynomials and contributions
    for k in range(2, Ks):
        if len(adj_mx.shape) == 2:  # Shared adjacency matrix
            T_k = 2 * torch.einsum("ij,bjd->bid", adj_mx, T_k_minus_1) - T_k_minus_2
        else:  # Batch-specific adjacency matrix
            T_k = 2 * torch.einsum("bij,bjd->bid", adj_mx, T_k_minus_1) - T_k_minus_2

        # Add contribution from T_k
        out += torch.einsum("bld,do->blo", T_k, weight[k])

        # Update for next iteration
        T_k_minus_2 = T_k_minus_1
        T_k_minus_1 = T_k

    # Add bias if present
    if bias is not None:
        assert len(bias.shape) == 1, "Bias tensor must have shape (c_out)"
        out += bias

    return out


def graph_conv(h, adj_mx, weight, bias=None):
    """
    Forward pass of the Graph Convolution layer.
    Parameters
    ----------
    h : torch.Tensor
        Input tensor of shape (B, L, c_in), where:
            B is the batch size,
            L is the number of nodes,
            c_in is the number of input features.
    adj_mx : torch.Tensor
        Adjacency matrix of the graph with shape (B, L) or (B, L, L), where:
            B is the batch size,
            L is the number of nodes.
    Returns
    -------
    torch.Tensor
        Output tensor of shape (B, L, c_out), where:
            B is the batch size,
            L is the number of nodes,
            c_out is the number of output features.
    """
    # First multiplication: propagate features through the graph
    out = torch.einsum("bij,bjd->bid", adj_mx, h)  # Shape: (B, L, c_in)
    # Second multiplication: feature transformation
    out = torch.einsum("bid,dk->bik", out, weight)  # Shape: (B, L, c_out)
    # Add bias if present
    if bias is not None:
        out += bias
    return out
