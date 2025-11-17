import torch
import torch.nn as nn
import math


class TimeEmbedding(nn.Module):
    def __init__(
        self,
        hid_dim,
        time_embedding_type="transformer",
        padding_value = -1,
        # --- RBF-specific hyperparameters ---
        rbf_ell=1.0,              # Length scale ℓ for the RBF kernel (larger ℓ -> slower decay w.r.t. |Δt|)
        rbf_learnable=True,      # Whether to make the sampled frequencies (and phases) learnable
        rbf_seed=42,            # RNG seed for reproducible frequency sampling
        rbf_use_phase=True,       # Whether to add a random phase b to each frequency
        rbf_normalize=True       # Whether to L2-normalize the output features
    ):
        super(TimeEmbedding, self).__init__()
        self.hid_dim = hid_dim
        self.time_embedding_type = time_embedding_type.lower()
        assert self.time_embedding_type in [
            "transformer",
            "rope",
            "time_difference",
            "rbf"
        ]
        if self.time_embedding_type == "rope":
            assert hid_dim % 2 == 0, "hid_dim must be even when using ROPE"
        if self.time_embedding_type == "rbf":
            assert hid_dim % 2 == 0, "hid_dim must be even when using RBF"

        self.padding_value = padding_value
        self.frequencies = nn.Parameter(
            torch.Tensor(hid_dim // 2).fill_(1.0), requires_grad=True
        )
        # --- RBF members (initialized only if RBF is selected) ---
        self.rbf_ell = float(rbf_ell)
        self.rbf_learnable = bool(rbf_learnable)
        self.rbf_use_phase = bool(rbf_use_phase)
        self.rbf_normalize = bool(rbf_normalize)

        if self.time_embedding_type == "rbf":
            # Number of frequency pairs (cos/sin), total dim = 2 * m = hid_dim
            m = hid_dim // 2

            gen = torch.Generator()
            if rbf_seed is not None:
                gen.manual_seed(int(rbf_seed))

            # Frequencies: ω ~ N(0, 1/ℓ^2)
            omega = torch.randn(m, generator=gen) / self.rbf_ell

            # Optional random phase b ~ U[0, 2π]
            if self.rbf_use_phase:
                b = torch.rand(m, generator=gen) * (2 * math.pi)
            else:
                b = torch.zeros(m)

            if self.rbf_learnable:
                self.omega = nn.Parameter(omega)  # trainable frequencies
                self.b = nn.Parameter(b)         # trainable phase (usually not necessary)
            else:
                self.register_buffer("omega", omega)  # fixed frequencies
                self.register_buffer("b", b)          # fixed phase

            # RFF scaling factor to stabilize feature magnitudes: sqrt(2 / m)
            self.register_buffer("rbf_scale", torch.tensor(math.sqrt(2.0 / m)))


    def forward(self, tt, padding_mask=None):
        """
        tt: Tensor of shape (B, L), containing continuous time values.
        Returns:
            Tensor of shape (B, L, hid_dim), containing time embeddings.
        """
        # Mask padding values
        if padding_mask is None:
            padding_mask = tt != self.padding_value
        if self.time_embedding_type == "transformer":
            t_emb = self.transformer_time_embedding(tt)
        elif self.time_embedding_type == "rope":
            t_emb = self.rope_time_embedding(tt)
        elif self.time_embedding_type == "time_difference":
            t_emb = self.time_difference_embedding(tt)
        elif self.time_embedding_type == "rbf":
            t_emb = self.rbf_time_embedding(tt)
        else:
            raise ValueError(f"Unknown time_embedding_type: {self.time_embedding_type}")

        t_emb = t_emb * padding_mask.unsqueeze(-1)
        return t_emb

    def transformer_time_embedding(self, tt):
        """
        Sinusoidal positional encoding adapted for continuous time values.
        """
        if tt.dim() == 1:
            tt = tt.unsqueeze(0)

        batch_size, seq_len = tt.size()
        device = tt.device

        # Use continuous time values directly
        position = tt.unsqueeze(-1)  # Shape: (B, L, 1)

        # Compute the div_term (frequencies)
        dim = self.hid_dim
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device) * (-math.log(1000.0) / dim)
        )
        scale = 2 * math.pi
        # Compute positional encodings
        pe = torch.zeros(batch_size, seq_len, dim, device=device)
        pe[:, :, 0::2] = torch.sin(position * div_term * scale)
        pe[:, :, 1::2] = torch.cos(position * div_term * scale)

        if pe.size(0) == 1:
            pe = pe.squeeze(0)

        return pe  # Shape: (B, L, hid_dim)

    def rope_time_embedding(self, tt):
        """
        Rotary positional embedding adapted for continuous time values.
        """
        batch_size, seq_len = tt.size()
        device = tt.device

        # Half of hid_dim for sin and cos components
        dim = self.hid_dim // 2

        # Compute the frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, device=device).float() / dim))

        # Compute angles
        position = tt.unsqueeze(-1)  # Shape: (B, L, 1)
        angles = position * inv_freq  # Shape: (B, L, dim)

        # Compute sin and cos components
        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)

        # Concatenate sin and cos components
        pe = torch.cat([sin_angles, cos_angles], dim=-1)  # Shape: (B, L, hid_dim)

        return pe  # Shape: (B, L, hid_dim)

    def time_difference_embedding(self, tt):
        batch_size, seq_len = tt.size()
        device = tt.device

        # Compute embeddings
        tt_expanded = tt.unsqueeze(-1)  # Shape: (B, L, 1)
        pe = torch.cos(
            tt_expanded * self.frequencies.to(device)
        )  # Shape: (B, L, hid_dim)

        return pe  # Shape: (B, L, hid_dim)

    def rbf_time_embedding(self, tt):
        """
        Random Fourier Features for the RBF kernel.

        Inner product approximation:
            <φ(t1), φ(t2)> ≈ exp(- (t1 - t2)^2 / (2 * ℓ^2))

        Implementation:
            Sample m frequencies ω_i ~ N(0, 1/ℓ^2) and (optionally) phases b_i ~ U[0, 2π].
            Define φ(t) = sqrt(2/m) * concat_i [cos(ω_i t + b_i), sin(ω_i t + b_i)].
            Output dimension is 2m = hid_dim.

        Args:
            tt: (B, L) continuous timestamps

        Returns:
            (B, L, hid_dim) RBF embeddings
        """
        if tt.dim() == 1:
            tt = tt.unsqueeze(0)

        # Use buffers/parameters; they move with .to(device) on the module.
        omega = self.omega  # (m,)
        b = self.b          # (m,)

        # Broadcast t against frequencies: (B, L, 1) * (1, 1, m) -> (B, L, m)
        t = tt.unsqueeze(-1)                         # (B, L, 1)
        arg = t * omega.view(1, 1, -1)               # (B, L, m)
        if self.rbf_use_phase:
            arg = arg + b.view(1, 1, -1)            # (B, L, m)

        c = torch.cos(arg)                           # (B, L, m)
        s = torch.sin(arg)                           # (B, L, m)
        pe = torch.cat([c, s], dim=-1) * self.rbf_scale  # (B, L, 2m = hid_dim)

        if self.rbf_normalize:
            pe = pe / (pe.norm(dim=-1, keepdim=True) + 1e-8)

        if pe.size(0) == 1:
            pe = pe.squeeze(0)
        return pe