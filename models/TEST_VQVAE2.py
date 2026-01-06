import math
import random
import numpy as np

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

class BasisLayer(nn.Module):
    """
    Basis Layer: Polynomial + Fourier Basis
    """
    def __init__(self, window_size, poly_degree=2, num_harmonics=2):
        super().__init__()
        self.window_size = window_size
        self.poly_degree = poly_degree
        self.num_harmonics = num_harmonics
        
        t = torch.linspace(-1, 1, window_size)
        basis_list = []

        # Polynomial terms
        for d in range(poly_degree + 1):
            basis_list.append(t**d)
            
        # Fourier terms
        for h in range(1, num_harmonics + 1):
            basis_list.append(torch.sin(2 * np.pi * h * t))
            basis_list.append(torch.cos(2 * np.pi * h * t))

        B = torch.stack(basis_list, dim=1)  # [Length, Num_Basis]
        B_pinv = torch.linalg.pinv(B)       # [Num_Basis, Length]

        self.register_buffer("B", B)
        self.register_buffer("B_pinv", B_pinv)

    def forward(self, x):
        # x: [Batch, Length, Channels]
        # Calculate Coefficients (Analysis)
        coeffs = torch.einsum('kl, blc -> bkc', self.B_pinv, x)
        
        # Reconstruct (Synthesis)
        global_component = torch.einsum('lk, bkc -> blc', self.B, coeffs)
        
        return global_component, coeffs

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs):
        # inputs: [B*C, Hidden, Num_Patches]
        inputs = inputs.permute(0, 2, 1).contiguous() # [B*C, N, H]
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embeddings.weight.t())
        )
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        return quantized.permute(0, 2, 1).contiguous(), loss

class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, patch_size=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        )

    def forward(self, x):
        return self.net(x)

class SimpleDecoder(nn.Module):
    def __init__(self, hidden_dim, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.net = nn.Linear(hidden_dim, patch_size)

    def forward(self, x):
        # x: [B, H, N]
        B, H, N = x.shape
        x = x.permute(0, 2, 1) # [B, N, H]
        x = self.net(x)        # [B, N, Patch]
        x = x.reshape(B, 1, N * self.patch_size) # [B, 1, L]
        return x

class QueryBasedTransformer(nn.Module):
    """
    Transformer using Learnable Queries + Cross Attention to handle variable lengths.
    Past Tokens (Source) -> Cross Attn -> Future Tokens (Target)
    """
    def __init__(self, hidden_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Standard Transformer Decoder Layer (Self Attn + Cross Attn + FFN)
        # We only strictly need Cross Attn part if Query implies position, 
        # but Self Attn allows queries to communicate.
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Positional Encoding for Past Tokens (Input)
        self.pos_embedding = nn.Parameter(torch.randn(1, hidden_dim, 1000)) # Simple learnable PE
        
    def forward(self, src_tokens, query_embed):
        """
        src_tokens: [Batch, Hidden, N_in] (Past)
        query_embed: [Batch, Hidden, N_out] (Future Placeholders)
        """
        B, H, N_in = src_tokens.shape
        _, _, N_out = query_embed.shape
        
        # 1. Add Position Embedding to Source
        # PE Slice: [1, H, N_in]
        src_pos = src_tokens + self.pos_embedding[:, :, :N_in]
        
        # 2. Permute for Transformer [Batch, Seq, Feature]
        # Memory (Keys/Values): Past Tokens
        memory = src_pos.permute(0, 2, 1) # [B, N_in, H]
        
        # Target (Queries): Learnable Future Tokens
        tgt = query_embed.permute(0, 2, 1) # [B, N_out, H]
        
        # 3. Transformer Decoder (Cross Attention)
        # Tgt attends to Memory
        output = self.transformer_decoder(tgt, memory) # [B, N_out, H]
        
        # 4. Restore Shape
        output = output.permute(0, 2, 1) # [B, H, N_out]
        
        return output

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.patch_size = getattr(configs, 'patch_size', 4)
        
        # Validity Check
        assert self.seq_len % self.patch_size == 0, "Seq_Len must be divisible by Patch_Size"
        assert self.pred_len % self.patch_size == 0, "Pred_Len must be divisible by Patch_Size"

        poly_degree = getattr(configs, 'poly_degree', 2)
        num_harmonics = getattr(configs, 'num_harmonics', 2)
        num_embeddings = getattr(configs, 'num_embeddings', 1024)
        
        self.num_basis = (poly_degree + 1) + (2 * num_harmonics)
        
        # ==========================================
        # 1. Global Components (Mathematical Projection)
        # ==========================================
        # Decomposer: Extracts coeffs from Past (Length L)
        self.decomposer = BasisLayer(self.seq_len, poly_degree, num_harmonics)
        
        # Recomposer: Reconstructs signal for Future (Length P)
        self.recomposer = BasisLayer(self.pred_len, poly_degree, num_harmonics)
        
        # Coefficient Projector: Maps Past Coeffs -> Future Coeffs
        # Dimensions are independent of L and P (Fixed K)
        self.global_forecaster = nn.Linear(self.num_basis, self.num_basis)
        
        # ==========================================
        # 2. Local Residual Path (Transformer)
        # ==========================================
        self.encoder = SimpleEncoder(input_dim=1, hidden_dim=configs.d_model, patch_size=self.patch_size)
        self.vq = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=configs.d_model)
        
        # Latent Transformer (Cross Attention)
        self.transformer = QueryBasedTransformer(
            hidden_dim=configs.d_model, 
            num_heads=self.configs.n_heads, # Configurable
            num_layers=self.configs.d_layers # Configurable
        )
        
        # Learnable Queries for Future Patches
        # Shape: [1, Hidden, Num_Future_Patches]
        self.num_future_patches = self.pred_len // self.patch_size
        self.future_query = nn.Parameter(torch.randn(1, configs.d_model, self.num_future_patches))
        
        self.decoder = SimpleDecoder(hidden_dim=configs.d_model, patch_size=self.patch_size)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, C = x_enc.shape

        # ==========================================
        # 1. Global Path (Variable Length Supported)
        # ==========================================
        # Analysis: Get Past Coefficients [B, K, C]
        _, coeffs_past = self.decomposer(x_enc)
        
        # Projection: Past Coeffs -> Future Coeffs [B, K, C]
        # Linear layer applied to dimension K (dim=1)
        coeffs_past_perm = coeffs_past.permute(0, 2, 1) # [B, C, K]
        coeffs_future = self.global_forecaster(coeffs_past_perm) # [B, C, K]
        coeffs_future = coeffs_future.permute(0, 2, 1) # [B, K, C]
        
        # Synthesis: Future Coeffs + Future Basis -> Future Trend [B, P, C]
        # Uses self.recomposer.B (size P)
        x_global_forecast = torch.einsum('lk, bkc -> blc', self.recomposer.B, coeffs_future)


        # ==========================================
        # 2. Local Residual Path (Transformer)
        # ==========================================
        # Calculate Residual
        # Note: We need reconstructed past trend to get residual.
        past_global_recon = torch.einsum('lk, bkc -> blc', self.decomposer.B, coeffs_past)
        x_residual = x_enc - past_global_recon
        
        # CI Reshape: [B*C, 1, L]
        x_residual_ci = x_residual.permute(0, 2, 1).reshape(B * C, 1, L)
        
        # Encode -> [B*C, H, N_in]
        z_past = self.encoder(x_residual_ci)
        
        # VQ (Past)
        z_past_q, vq_loss_past = self.vq(z_past)
        
        # Prepare Queries: Expand [1, H, N_out] -> [B*C, H, N_out]
        batch_queries = self.future_query.repeat(B * C, 1, 1)
        
        # Transformer: Cross Attention (Query=Future, Key/Val=Past)
        # Output: [B*C, H, N_out]
        z_future = self.transformer(z_past_q, batch_queries)
        
        # Re-Quantize (Future)
        # z_future_q, vq_loss_future = self.vq(z_future)
        
        # Decode -> [B*C, 1, P]
        # y_residual_ci = self.decoder(z_past_q)
        y_residual_ci = self.decoder(z_future)
        
        # Reshape Back -> [B, P, C]
        y_residual = y_residual_ci.reshape(B, C, self.pred_len).permute(0, 2, 1)

        # ==========================================
        # 3. Combine
        # ==========================================
        forecast = x_global_forecast + y_residual
        # total_vq_loss = vq_loss_past + vq_loss_future
        total_vq_loss = vq_loss_past

        return forecast, total_vq_loss

    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        forecast, vq_loss = self.forecast(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return forecast, vq_loss