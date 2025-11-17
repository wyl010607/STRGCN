from tkinter import NO
import warnings
import torch
import torch.nn as nn
from .STRGCN_Layer import STRGCNLayer, EfficientSTRGCNLayer
from .TimeEmbedding import TimeEmbedding
from .IrrRevin import IrrRevIN


class STRGCN(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        hid_dim,
        n_layer,
        node_dim,
        dropout,
        #num_hyper_nodes_per_var=None,
        gcn_type="cheb",
        gcn_depth=None,
        use_bias=True,
        use_norm=True,
        time_embedding_type="rope",
        load_adj=False,
        use_efficient_gcn=False,
        use_revin=False,
        revin_affine=False,
        global_args=None,
        **kwargs
    ):
        super(STRGCN, self).__init__()
        self.task_name = global_args.get("task_name")
        self.c_in = c_in
        self.c_out = c_out
        self.hid_dim = hid_dim
        self.n_layer = n_layer
        self.num_nodes = global_args.get("num_of_vals")
        self.use_efficient_gcn = use_efficient_gcn
        self.use_info_agg = kwargs.get("use_info_agg", False)
        self.num_hyper_nodes_per_var = global_args.get("num_hyper_nodes_per_var")  # number of hyper node per var.  not necessary when use_efficient_gcn is False
        # parameters for GCNLayer
        self.dropout = dropout
        self.gcn_type = gcn_type
        self.gcn_depth = gcn_depth
        self.use_bias = use_bias
        self.use_norm = use_norm
        # parameters for TimeEmbedding
        self.time_embedding_type = time_embedding_type
        # parameters for Adjacency Matrix
        self.node_dim = node_dim
        self.load_adj = load_adj
        if load_adj:
            warnings.warn("Loading adjacency matrix is not implemented yet.")
        # self.learnable_adj = learnable_adj

        # RevIN (min–max) for long-form
        self.use_revin = use_revin
        if self.use_revin:
            self.revin = IrrRevIN(num_vars=self.num_nodes, eps=1e-6, learn_affine=revin_affine)

        # special tokens
        self.pred_token = nn.Parameter(torch.randn(1, 1, hid_dim))

        # encoder (B,L,c_in) -> (B,L,hid_dim)
        self.encoder = nn.Linear(c_in, hid_dim)

        if use_efficient_gcn:
            STRGCN_layer = EfficientSTRGCNLayer
        else:
            STRGCN_layer = STRGCNLayer

        # gcn_layers
        self.gcn_layers = nn.ModuleList(
            [
                STRGCN_layer(
                    hid_dim,
                    hid_dim,
                    node_dim,
                    dropout,
                    self.num_nodes,
                    num_hyper_nodes_per_var=self.num_hyper_nodes_per_var,
                    gcn_type=gcn_type,
                    gcn_depth=gcn_depth,
                    use_bias=use_bias,
                    use_norm=use_norm,
                    time_embedding_type=time_embedding_type,
                    layer_idx = layer_idx,
                    #temperature=temperature,
                )
                for layer_idx in range(n_layer)
            ]
        )

        # positional encoding
        self.time_embedding = TimeEmbedding(
            hid_dim, time_embedding_type=time_embedding_type
        )

        # create adjacency matrix
        #self.learnable_adj_weight_U = nn.Embedding(
        #    num_embeddings=self.num_nodes, embedding_dim=node_dim
        #)
        #self.learnable_adj_weight_V = nn.Embedding(
        #    num_embeddings=self.num_nodes, embedding_dim=node_dim
        #)
        self.lambda_ = nn.Parameter(torch.tensor(0.5))

        # projection
        if self.task_name in ["prediction", "interpolate"]:
            self.projection = nn.Linear(hid_dim, c_out)

        else:
            raise ValueError(f"task_name={self.task_name} is not supported.")

    def forward(
        self,
        batch_value,
        batch_timestamp,
        batch_var_idx,
        batch_pred_mask=None,
        batch_pad_mask=None,
        batch_time_id=None,
        batch_agg_mask=None
    ):
        # batch_value (B, L); batch_timestamp (B, L); batch_var_id (L); batch_pred_mask (B, L)

        # -------- RevIN (min–max, long-form) --------
        if self.use_revin:
            x_norm, revin_stats = self.revin.normalize(
                batch_value, batch_var_idx, pad_mask=batch_pad_mask, pred_mask=batch_pred_mask
            )  # (B,L)
        else:
            x_norm = batch_value
            revin_stats = None
        # -------- Gaussian-based graph aggregation --------
        if self.use_info_agg:
            agg_mask = batch_agg_mask # (B,L,num_hyper_nodes_per_var)
        else:
            agg_mask = None
        # encoder
        x_emb = self.encoder(x_norm.unsqueeze(-1))  # (B, L, hid_dim)
        batch_pred_mask = batch_pred_mask.unsqueeze(-1).bool()
        # time embedding
        t_emb = self.time_embedding(batch_timestamp, batch_pad_mask)  # (B, L, hid_dim)
        h_0 = x_emb + t_emb

        h_0 = h_0 * batch_pad_mask.unsqueeze(-1) * (~batch_pred_mask) + self.pred_token * batch_pred_mask

        # adj
        #adj_U = self.learnable_adj_weight_U(batch_var_idx)
        #adj_V = self.learnable_adj_weight_V(batch_var_idx)
        #adj = torch.einsum("bld,bmd->blm", adj_U, adj_V)  # (B, L, L)
        #adj = adj / adj.sum(dim=-1, keepdim=True)
        # adj = temperature_softmax(adj, dim=1, temperature=1.0)

        # gcn layers
        h = h_0
        for gcn_layer in self.gcn_layers:
            h = gcn_layer(h, batch_var_idx, t_emb, batch_timestamp, batch_pad_mask, agg_mask=agg_mask)
        h = h * self.lambda_ + h_0 * (1 - self.lambda_)

        # projection
        y_pred = self.projection(h)

        # -------- RevIN inverse  --------
        if self.use_revin:
            y_pred = self.revin.denormalize(y_pred, batch_var_idx, revin_stats)  # (B,L,c_out)

        return y_pred
