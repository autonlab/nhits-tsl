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
    Supports multi-channel input (B, L, C).
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

        B = torch.stack(basis_list, dim=1)  # [window_size, num_basis]
        B_pinv = torch.linalg.pinv(B)       # [num_basis, window_size]

        self.register_buffer("B", B)
        self.register_buffer("B_pinv", B_pinv)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        # OLS Coeffs: B_pinv * X -> [Batch, Num_Basis, Channels]
        coeffs = torch.einsum('kl, blc -> bkc', self.B_pinv, x)
        
        # Reconstruction: B * Coeffs -> [Batch, Seq_Len, Channels]
        global_component = torch.einsum('lk, bkc -> blc', self.B, coeffs)
        
        return global_component, coeffs

class VectorQuantizer(nn.Module):
    """
    Standard VQ-VAE Layer to tokenize the residual signals.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook (Vocabulary of Shapes)
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs):
        # inputs: [Batch*Channel, Embedding_Dim, Length]
        # Permute for metric calculation: [B*C, L, Emb_Dim]
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embeddings.weight.t())
        )
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        
        # Losses
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Restore shape: [B*C, Emb_Dim, L]
        return quantized.permute(0, 2, 1).contiguous(), loss

class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, patch_size=4):
        super().__init__()
        # stride=patch_size -> Seq_len L -> L/patch_size -> [B*C, Hidden, L/patch_size]
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Downsampling Layer
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        )

    def forward(self, x):
        # x: [Batch * Channel, 1, Length]
        # output: [Batch * Channel, Hidden, Length / Patch_Size]
        return self.net(x)

class SimpleDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, patch_size=4):
        super().__init__()
        # Upsampling Layer (ConvTranspose1d)
        # self.net = nn.Sequential(
        #     nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, kernel_size=patch_size, stride=patch_size, padding=0),
        #     nn.ReLU(),
        #     nn.Conv1d(hidden_dim // 2, output_dim, kernel_size=3, stride=1, padding=1),
        # )
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.net = nn.Linear(hidden_dim, patch_size)

    def forward(self, x):
        # x: [Batch * Channel, Hidden, Length / Patch_Size]
        # output: [Batch * Channel, 1, Length]
        B, H, N = x.shape
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.reshape(B, 1, N * self.patch_size)
        return x

class LatentForecaster(nn.Module):
    def __init__(self, hidden_dim, num_patches):
        super().__init__()
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_patches, self.hidden_dim * self.num_patches),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * self.num_patches, self.hidden_dim * self.num_patches),
        )

    def forward(self, x):
        B, H, N = x.shape
        x = x.reshape(B, -1)
        x = self.net(x)
        x = x.reshape(B, H, N)
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        
class Model(nn.Module):
    """
    NHITS-style Foundation Model (CI + VQ Tokenization)
    """
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.patch_size = configs.patch_size

        poly_degree = configs.poly_degree
        num_harmonics = configs.num_harmonics
        num_embeddings = configs.num_embeddings
        
        self.num_basis = (poly_degree + 1) + (2 * num_harmonics)
        
        # 1. Global Components
        self.decomposer = BasisLayer(self.seq_len, poly_degree, num_harmonics)
        # self.recomposer = BasisLayer(self.pred_len, poly_degree, num_harmonics)
        
        # 2. Local Residual Path (Tokenization)
        # Encoder: Residual -> Continuous Latent
        self.encoder = SimpleEncoder(input_dim=1, hidden_dim=configs.d_ff, patch_size=self.patch_size)
        
        # Tokenizer: Continuous Latent -> Discrete Codebook
        self.vq = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=configs.d_ff)
        self.num_patches_count = self.seq_len // self.patch_size
        self.latent_forecaster = LatentForecaster(hidden_dim=configs.d_ff, num_patches=self.num_patches_count)        # Decoder: Discrete Codebook -> Forecast Residual
        self.decoder = SimpleDecoder(hidden_dim=configs.d_ff, output_dim=configs.pred_len, patch_size=self.patch_size)
        
        # Global Coefficients Predictor
        self.global_forecaster = nn.Linear(self.num_basis, self.num_basis)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [Batch, Seq_Len, Enc_In]
        B, L, C = x_enc.shape

        # --- 1. Global Path ---
        x_global, coeffs = self.decomposer(x_enc)
        
        coeffs_perm = coeffs.permute(0, 2, 1) 
        x_global_forecast_coeffs = self.global_forecaster(coeffs_perm)
        x_global_forecast_coeffs = x_global_forecast_coeffs.permute(0, 2, 1)
        
        x_global_forecast = torch.einsum('lk, bkc -> blc', self.decomposer.B, x_global_forecast_coeffs)

        # --- 2. Local Residual Path (Tokenization) ---
        x_residual = x_enc - x_global
        
        # CI Reshape: [B*C, 1, L]
        x_residual_ci = x_residual.permute(0, 2, 1).reshape(B * C, 1, L)
        
        # Encode -> [B*C, d_ff, L / Patch_Size]
        z_residual = self.encoder(x_residual_ci)
        
        # Tokenize (VQ) -> [B*C, d_ff, L / Patch_Size], loss
        z_quantized, vq_loss_seq = self.vq(z_residual)

        z_future = self.latent_forecaster(z_quantized)
        # z_future_quantized, vq_loss_pred = self.vq(z_future)
        # Decode -> [B*C, 1, L / Patch_Size] (or Pred_Len)
        y_residual_ci = self.decoder(z_future)
        
        # vq_loss = vq_loss_seq + vq_loss_pred
        vq_loss = vq_loss_seq
        # Restore Shape: [B*C, 1, Pred_Len] -> [B, Pred_Len, C]
        y_residual = y_residual_ci.reshape(B, C, self.pred_len).permute(0, 2, 1)

        # --- 3. Combine ---
        forecast = x_global_forecast + y_residual

        # Return both forecast and auxiliary loss
        return forecast, vq_loss

    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        # Model returns (forecast, vq_loss)
        # Training loop expects 'outputs' usually. 
        # You might need to adjust the training loop to handle the tuple return, 
        # or handle the loss aggregation here if the loop is fixed.
        # Assuming standard run.py, it takes outputs[0] as prediction usually, 
        # but we need to optimize vq_loss too.
        
        forecast, vq_loss = self.forecast(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
        return forecast, vq_loss