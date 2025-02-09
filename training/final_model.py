import numpy as np
import torch
import torch.nn as nn


class SinusoidalPositionalEncoder(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        pe = torch.zeros(seq_len, embed_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe) # (1, seq_len, embed_dim)

    def forward(self, x):
        return x + self.pe.to(x.device)


class TimeTransformer(nn.Module):
    def __init__(self, seq_len, feature_dim, embed_dim, num_heads, hidden_dim, dropout, num_layers, latent_dim):
        super().__init__()
        self.seq_len = seq_len
        self.input_fc = nn.Linear(feature_dim, embed_dim)
        self.positional_encoder = SinusoidalPositionalEncoder(seq_len, embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.fc_mu = nn.Linear(seq_len * embed_dim, seq_len * latent_dim)
        self.fc_logvar = nn.Linear(seq_len * embed_dim, seq_len * latent_dim)
        self.decoder_layer = nn.TransformerEncoderLayer(latent_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, activation="gelu", batch_first=True)
        self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers)
        self.output_fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, feature_dim)
        )

    def forward(self, x):
        out = self.input_fc(x) # (batch_size, seq_len, embed_dim)
        out = self.positional_encoder(out) # (batch_size, seq_len, embed_dim)
        out_encoder = self.encoder(out) # (batch_size, seq_len, embed_dim)
        out = out_encoder.view(out_encoder.shape[0], -1) # (batch_size, seq_len * embed_dim)
        mu = self.fc_mu(out) # (batch_size, seq_len * latent_dim)
        logvar = self.fc_logvar(out) # (batch_size, seq_len * latent_dim)
        std = torch.exp(0.5 * logvar) # (batch_size, seq_len * latent_dim)
        eps = torch.randn_like(std) # (batch_size, seq_len * latent_dim)
        z = mu + std * eps # (batch_size, seq_len * latent_dim)
        z = z.view(z.shape[0], self.seq_len, -1) # (batch_size, seq_len, latent_dim)
        out = self.decoder(z) # (batch_size, seq_len, latent_dim)
        out = self.output_fc(out) # (batch_size, seq_len, feature_dim)
        return out, out_encoder, mu, logvar


class FrequencyCNN(nn.Module):
    def __init__(self, seq_len, input_channels, hidden_dim, latent_dim):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.fc_mu = nn.Linear(hidden_dim // 2 * seq_len, latent_dim * seq_len)
        self.fc_logvar = nn.Linear(hidden_dim // 2 * seq_len, latent_dim * seq_len)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, input_channels)
        )

    def forward(self, x):
        out = x.permute(0, 2, 1) # (batch_size, seq_len, input_channels) -> (batch_size, input_channels, seq_len)
        out_cnn = self.cnn(out) # (batch_size, hidden_dim // 2, seq_len)
        out = out_cnn.view(out_cnn.shape[0], -1) # (batch_size, hidden_dim // 2 * seq_len)
        mu = self.fc_mu(out) # (batch_size, latent_dim * seq_len)
        logvar = self.fc_logvar(out) # (batch_size, latent_dim * seq_len)
        std = torch.exp(0.5 * logvar) # (batch_size, latent_dim * seq_len)
        eps = torch.randn_like(std) # (batch_size, latent_dim * seq_len)
        z = mu + std * eps # (batch_size, latent_dim * seq_len)
        z = z.view(z.shape[0], -1, self.seq_len) # (batch_size, latent_dim, seq_len)
        z = z.permute(0, 2, 1) # (batch_size, seq_len, latent_dim)
        out = self.decoder(z) # (batch_size, seq_len, input_channels)
        return out, out_cnn, mu, logvar


class DualStreamModel(nn.Module):
    def __init__(
        self, 
        seq_len, 
        feature_dim, 
        embed_dim, 
        num_heads, 
        hidden_dim_transformer, 
        dropout_transformer, 
        num_transformer_layers, 
        latent_dim_transformer, 
        hidden_dim_cnn, 
        latent_dim_cnn
    ):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.time_transformer = TimeTransformer(seq_len, feature_dim, embed_dim, num_heads, hidden_dim_transformer, dropout_transformer, num_transformer_layers, latent_dim_transformer)
        self.freq_cnn = FrequencyCNN(seq_len, 20, hidden_dim_cnn, latent_dim_cnn)
        self.final_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim_transformer), 
            nn.ReLU(), 
            nn.Linear(hidden_dim_transformer, hidden_dim_transformer),
            nn.ReLU(),
            nn.Linear(hidden_dim_transformer, feature_dim)
        )
    
    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    def forward(self, x):
        time_reconstruction, time_encoded, time_mu, time_logvar = self.time_transformer(x)
        stft_0 = torch.stft(
            x[:, :, 0], 
            n_fft=self.seq_len, 
            hop_length=self.seq_len // 4, 
            win_length=self.seq_len, 
            window=torch.hann_window(self.seq_len).to(x.device), 
            onesided=False,
            return_complex=True
        ) # (batch_size, seq_len, 5)
        stft_1 = torch.stft(
            x[:, :, 1], 
            n_fft=self.seq_len, 
            hop_length=self.seq_len // 4, 
            win_length=self.seq_len, 
            window=torch.hann_window(self.seq_len).to(x.device), 
            onesided=False,
            return_complex=True
        ) # (batch_size, seq_len, 5)
        stft = torch.cat([stft_0.real, stft_0.imag, stft_1.real, stft_1.imag], dim=2) # (batch_size, seq_len, 20)
        stft_reconstruction, stft_encoded, stft_mu, stft_logvar = self.freq_cnn(stft) # (batch_size, seq_len, 20)
        stft_dim = stft_0.shape[-1]
        stft_0_reconstruction = torch.complex(stft_reconstruction[:, :, :stft_dim], stft_reconstruction[:, :, stft_dim:2 * stft_dim])
        stft_1_reconstruction = torch.complex(stft_reconstruction[:, :, 2 * stft_dim:3 * stft_dim], stft_reconstruction[:, :, 3 * stft_dim:])
        freq_reconstruction_0 = torch.istft(
            stft_0_reconstruction, 
            n_fft=self.seq_len, 
            hop_length=self.seq_len // 4, 
            win_length=self.seq_len, 
            window=torch.hann_window(self.seq_len).to(x.device), 
            onesided=False,
            length=self.seq_len,
            return_complex=False)
        freq_reconstruction_1 = torch.istft(
            stft_1_reconstruction, 
            n_fft=self.seq_len, 
            hop_length=self.seq_len // 4, 
            win_length=self.seq_len, 
            window=torch.hann_window(self.seq_len).to(x.device), 
            onesided=False,
            length=self.seq_len,
            return_complex=False)
        freq_reconstruction = torch.cat([freq_reconstruction_0.unsqueeze(-1), freq_reconstruction_1.unsqueeze(-1)], dim=2)
        reconstructions = torch.cat([time_reconstruction, freq_reconstruction], dim=2)
        final_reconstruction = self.final_fusion(reconstructions)

        kl_loss = self.kl_divergence(time_mu, time_logvar) + self.kl_divergence(stft_mu, stft_logvar)
        return final_reconstruction, time_reconstruction, freq_reconstruction, time_encoded, stft_encoded, kl_loss, time_mu, stft_mu
