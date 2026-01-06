# Cell
import math
import random
import numpy as np

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Tuple
from functools import partial

class BasisLayer(nn.Module):
    """
    Basis Layer to extract global components (trend + seasonality).
    For now, hardcoded to polynomial + Fourier basis and supporting only single-channel input.

    #TODO: Extend to multi-channel if needed.
    #TODO: Extend to support other basis types.
    """

    def __init__(self, window_size, poly_degree=2, num_harmonics=2):
        super().__init__()
        self.window_size = window_size
        t = torch.linspace(-1, 1, window_size)
        basis_list = []

        # TODO: Here we can define whatever basis we want, e.g., add support for Legrande polynomials
        for d in range(poly_degree + 1):
            basis_list.append(t**d)
        for h in range(1, num_harmonics + 1):
            basis_list.append(torch.sin(2 * np.pi * h * t))
            basis_list.append(torch.cos(2 * np.pi * h * t))

        B = torch.stack(basis_list, dim=1)  # dims: [window_size, num_basis]

        B_pinv = torch.linalg.pinv(B)  # Compute Pseudo-inverse

        self.register_buffer("B", B)
        self.register_buffer("B_pinv", B_pinv)


    def forward(self, x):
            # x: [Batch, Seq_Len, Channels]
            
            # Coefficients : Coeffs = B_pinv * X
            # einsum: 
            #   x: (batch, len, channel) -> blc
            #   B_pinv: (num_basis, len) -> kl (k=num_basis, l=len)
            #   Calculated pinv shape is (num_basis, window_size) if B is (window_size, num_basis)
            
            # B_pinv shape: [num_basis, window_size]
            # x shape: [Batch, window_size, Channels]
            # coeffs shape: [Batch, num_basis, Channels]
            coeffs = torch.einsum('kl, blc -> bkc', self.B_pinv, x)
            
            # Reconstruction: Global = B * Coeffs
            # B shape: [window_size, num_basis]
            # global shape: [Batch, window_size, Channels]
            global_component = torch.einsum('lk, bkc -> blc', self.B, coeffs)

            return global_component, coeffs

class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class SimpleDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, output_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)

class SimpleTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.transformer = nn.Transformer(hidden_dim, num_heads, num_layers)

    def forward(self, x):
        return self.transformer(x)

class HybridDeStationaryVQVAE(nn.Module):
    """
    Basis Subtraction + VQ-VAE + Basis Addition.
    """

    def __init__(self, window_size, hidden_dim=64, num_embeddings=128):
        super().__init__()
        self.basis_layer = BasisLayer(window_size)
        self.encoder = SimpleEncoder(1, hidden_dim)
        self.vq = VectorQuantizer(num_embeddings, hidden_dim)
        self.decoder = SimpleDecoder(hidden_dim, 1)

    def forward(self, x):
        # 1. Physics Subtraction
        x_global = self.basis_layer(x)
        x_res = x - x_global

        # 2. Tokenize Residuals
        z = self.encoder(x_res)
        z_q, vq_loss = self.vq(z)
        x_res_recon = self.decoder(z_q)

        # 3. Physics Addition
        x_final = x_res_recon + x_global
        return x_final, vq_loss

class VectorQuantizer(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(
            -1 / self.num_embeddings, 1 / self.num_embeddings
        )

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embeddings.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        return quantized.permute(0, 2, 1).contiguous(), loss


class Model(nn.Module):
    """
    test model
    """
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.num_basis = (self.configs.poly_degree + 1) + (2 * self.configs.num_harmonics)
        self.decomposer = BasisLayer(self.configs.seq_len, self.configs.poly_degree, self.configs.num_harmonics)
        self.recomposer = BasisLayer(self.pred_len, self.configs.poly_degree, self.configs.num_harmonics)

        self.encoder = SimpleEncoder(1, self.configs.d_model)
        self.decoder = SimpleDecoder(self.configs.d_model, 1)
        self.global_forecaster = nn.Linear(self.num_basis, self.num_basis)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, C = x_enc.shape
        
        x_global, coeffs = self.decomposer(x_enc)
        # 2. Local Residual Path
        x_residual = x_enc - x_global
        
        # Reshape: (Batch, Length, Channel) -> (Batch * Channel, 1, Length)
        x_residual_ci = x_residual.permute(0, 2, 1).reshape(B * C, 1, L)
        
        # Encoder Pass
        z_residual = self.encoder(x_residual_ci) # Output: [B*C, d_ff, L]
        
        # Decoder Pass
        y_residual_ci = self.decoder(z_residual) # Output: [B*C, 1, L]
        
        # Length Adjustment (Seq_Len -> Pred_Len)
        if y_residual_ci.shape[-1] != self.pred_len:
             y_residual_ci = F.interpolate(y_residual_ci, size=self.pred_len, mode='linear')
             # Output: [B*C, 1, Pred_Len]
        # Reshape Back: (Batch * Channel, 1, Pred_Len) -> (Batch, Pred_Len, Channel)
        y_residual = y_residual_ci.reshape(B, C, self.pred_len).permute(0, 2, 1)

        # 4. Global Forecast (Coefficients Projection)
        # coeffs: [B, Num_Basis, Enc_In] -> Linear는 마지막 차원에 작용하므로 Transpose 필요
        # Linear: Num_Basis -> Num_Basis
        coeffs_perm = coeffs.permute(0, 2, 1) # [B, Enc_In, Num_Basis]
        x_global_forecast_coeffs = self.global_forecaster(coeffs_perm) # [B, Enc_In, Num_Basis]
        x_global_forecast_coeffs = x_global_forecast_coeffs.permute(0, 2, 1) # [B, Num_Basis, Enc_In]

        # Recompose Global Component for Prediction Horizon
        # Using recomposer's Basis Matrix (Pred_Len x Num_Basis)
        # Global_Pred = B_pred * Coeffs_Pred
        # einsum: lk (B_pred), bkc (coeffs) -> blc
        x_global_forecast = torch.einsum('lk, bkc -> blc', self.recomposer.B, x_global_forecast_coeffs)

        # 5. Combine
        forecast = x_global_forecast + y_residual

        return forecast

    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        # batch_x: [Batch, Seq_Len, Channel]
        return self.forecast(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    def forecast_decomposition(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        x_global, coeffs = self.decomposer(x_enc)
        x_residual = x_enc - x_global
        z_residual = self.encoder(x_residual)
        y_residual = self.decoder(z_residual)

        x_global_forecast = self.global_forecaster(coeffs)

        forecast = x_global_forecast + y_residual

        return forecast, x_global_forecast, y_residual  # (B*, H), (B*, H), (B*, H)

