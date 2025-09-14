import torch
import torch.nn as nn
import torch.nn.functional as F

class HNetModel(nn.Module):
    def __init__(self, input_channels=8, latent_channels=64):
        """
        H-Net autoencoder: 
        - input_channels: number of EMG channels (input features per time-step)
        - latent_channels: number of feature maps at the bottleneck (depth of U-Net)
        """
        super(HNetModel, self).__init__()
        # Encoder (downsampling conv layers)
        self.enc1 = nn.Conv1d(input_channels, 32, kernel_size=5, stride=2, padding=2)  # downsample by 2
        self.enc2 = nn.Conv1d(32, latent_channels, kernel_size=5, stride=2, padding=2)  # downsample by 2
        # Decoder (upsampling conv transpose layers)
        self.dec1 = nn.ConvTranspose1d(latent_channels, 32, kernel_size=4, stride=2, padding=1)  # upsample by 2
        self.dec2 = nn.ConvTranspose1d(32 + 32, input_channels, kernel_size=4, stride=2, padding=1)  
        # (32 from dec1 output concatenated with 32 from enc1 skip, outputs original channels)
    
    def forward(self, x):
        # Encoder forward
        # x shape: [batch, C, L]
        enc1_out = F.relu(self.enc1(x))          # -> shape [batch, 32, L/2]
        enc2_out = F.relu(self.enc2(enc1_out))   # -> shape [batch, latent_channels, L/4]
        # Decoder forward
        dec1_out = F.relu(self.dec1(enc2_out))   # -> shape [batch, 32, L/2] (mirrors enc1_out length)
        # Concatenate skip connection from enc1_out
        combined = torch.cat((dec1_out, enc1_out), dim=1)  # concat along channel dim
        dec2_out = self.dec2(combined)          # -> shape [batch, input_channels, L] (reconstructed output)
        return dec2_out  # reconstructed signal
    
    def compute_loss(self, x, target=None):
        """
        Compute reconstruction loss (MSE) between output and input.
        """
        # For autoencoder, target is just the input itself (if not provided explicitly).
        recon = self.forward(x)
        if target is None:
            target = x
        loss = F.mse_loss(recon, target)
        return loss
    
    def get_embedding(self, x):
        """
        Get a latent embedding vector for the input sequence, for use in classification.
        We'll take the bottleneck features averaged over time as the sequence embedding.
        """
        # Compute encoder outputs without decoding
        enc1_out = F.relu(self.enc1(x))
        enc2_out = F.relu(self.enc2(enc1_out))   # shape [batch, latent_channels, L/4]
        # Take mean over time dimension to get a fixed-length representation
        # enc2_out.mean(dim=2) yields [batch, latent_channels]
        embedding = enc2_out.mean(dim=2)
        return embedding



class VQWav2VecModel(nn.Module):
    def __init__(self, input_channels=8, encoder_dim=64, codebook_size=128, code_dim=64):
        """
        VQ-wav2vec style model:
        - encoder_dim: dimension of continuous encoder output
        - codebook_size: number of discrete codebook vectors
        - code_dim: dimension of each code (usually same as encoder_dim for direct quantization)
        """
        super(VQWav2VecModel, self).__init__()
        # Encoder: 1D CNN to produce latent sequence
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, encoder_dim, kernel_size=5, stride=2, padding=2), nn.ReLU()
        )
        # Codebook: learnable embedding table for quantization
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.codebook = nn.Embedding(codebook_size, code_dim)
        # Initialize codebook embeddings
        nn.init.uniform_(self.codebook.weight, -0.1, 0.1)
        # Context network: GRU to produce context vectors from quantized sequence
        self.context_rnn = nn.GRU(input_size=code_dim, hidden_size=encoder_dim, batch_first=True)
        # The context vector (GRU hidden state) will be used to predict next code
        # We can use a simple linear layer to score similarity between context and code vectors
        self.predict_proj = nn.Linear(encoder_dim, code_dim)  # project context to code dimension
    
    def quantize(self, latents):
        """
        Quantize the continuous latent vectors to nearest codebook entries.
        latents: Tensor of shape [B, T, encoder_dim]
        Returns: quantized vectors (B, T, code_dim), code indices (B, T)
        """
        # Flatten B and T dims for distance computation
        B, T, D = latents.shape  # D = encoder_dim
        flat_latents = latents.reshape(B * T, D)  # shape [B*T, D]
        # Compute L2 distance to each code in the codebook
        # dist[i, j] = ||latents_i - code_j||^2
        with torch.no_grad():
            # Compute squared distances: ||z||^2 - 2 z·c + ||c||^2
            latent_norm = (flat_latents ** 2).sum(dim=1, keepdim=True)  # [B*T, 1]
            code_norm = (self.codebook.weight ** 2).sum(dim=1)         # [codebook_size]
            # Compute dot product term
            dot = flat_latents @ self.codebook.weight.T               # [B*T, codebook_size]
            # dist = ||z||^2 + ||c||^2 - 2 * dot
            dist = latent_norm + code_norm.unsqueeze(0) - 2 * dot
            # Find nearest code index for each latent
            _, indices = dist.min(dim=1)  # indices shape [B*T]
        codes = indices.reshape(B, T)  # shape [B, T]
        # Lookup quantized vectors from codebook
        quantized = self.codebook(codes)  # shape [B, T, code_dim]
        # Straight-through estimator: pass gradients from quantized to encoder
        # (Add and subtract the encoder output to treat it as identity in backward)
        quantized = latents + (quantized - latents).detach()
        return quantized, codes
    
    def forward(self, x):
        """
        Forward pass: returns quantized latent sequence and context representations.
        x shape: [B, C, L] raw input.
        """
        # 1. Continuous encoder features
        z = self.encoder(x)                     # shape [B, encoder_dim, L_enc]
        z = z.permute(0, 2, 1)                  # -> [B, L_enc, encoder_dim] for time-major
        # 2. Vector quantization
        quant_z, codes = self.quantize(z)       # both shape [B, L_enc, code_dim]
        # 3. Context network (GRU) over quantized sequence
        # We get output for each time-step and final hidden state
        context_outputs, _ = self.context_rnn(quant_z)  # context_outputs: [B, L_enc, hidden_size]
        # 4. Project context to code space for prediction
        pred = self.predict_proj(context_outputs)  # shape [B, L_enc, code_dim]
        return quant_z, pred, codes
    
    def compute_loss(self, x, target=None):
        """
        Compute the contrastive loss (InfoNCE) for predicting next code.
        """
        B = x.size(0)
        quant_z, pred, codes = self.forward(x)  # pred: [B, T, code_dim]
        T = pred.size(1)
        # We will predict the code for the next time-step (offset = 1).
        # For simplicity, we ignore the last time-step in each sequence for prediction.
        if T < 2:
            return torch.tensor(0.0)  # not enough steps to predict
        # Formulate InfoNCE loss:
        # For each time t (0 <= t < T-1) in each sequence, treat pred[:, t, :] vs quant_z[:, t+1, :] as positive pair.
        # Negatives: use quant_z[:, t+1, :] from other sequences in the batch as negatives.
        # Compute similarity (dot product) between pred_t (context at t) and all candidates for time t+1 in batch.
        # pred_t has shape [B, code_dim]; next_codes = quant_z[:, t+1, :] has shape [B, code_dim].
        loss_nce = 0.0
        for t in range(T-1):
            # Normalize vectors (optional, for stability)
            context_t = F.normalize(pred[:, t, :], dim=-1)       # [B, code_dim]
            target_next = F.normalize(quant_z[:, t+1, :], dim=-1)  # [B, code_dim]
            # Compute similarity matrix: sim[j, k] = context_t[j] · target_next[k]^T
            sim = torch.matmul(context_t, target_next.T)  # shape [B, B]
            # For each sample j, the positive is k=j (same index), negatives are k != j
            labels = torch.arange(B, device=x.device)  # target indices 0..B-1
            # Compute InfoNCE loss for this time-step (cross entropy with positives on diagonal)
            loss_t = F.cross_entropy(sim, labels)
            loss_nce += loss_t
        loss_nce = loss_nce / (T-1)
        # We can also add a commitment loss or codebook diversity loss if desired (not shown for brevity).
        return loss_nce
    
    def get_embedding(self, x):
        """
        Get a fixed-length embedding for classification from the model.
        We'll run the forward pass and use the final context vector (last GRU hidden state) as the sequence representation.
        """
        self.eval()  # ensure in eval mode (no dropout, etc.)
        with torch.no_grad():
            # Run forward to get context outputs
            quant_z, pred, codes = self.forward(x)
            # context_outputs from GRU are available as part of forward's return (we didn't explicitly return it, 
            # but we can easily get final hidden state by running GRU directly here).
            # Instead of modifying forward, let's directly use context_rnn to get final hidden:
            _, hidden = self.context_rnn(quant_z)
            # hidden shape: (num_layers, B, hidden_size). We use hidden_state of last layer for each batch.
            final_hidden = hidden[-1]  # shape [B, hidden_size]
            embedding = final_hidden  # use final GRU hidden state as embedding
        return embedding



class ContrastiveModel(nn.Module):
    def __init__(self, input_channels=8, embedding_dim=64):
        """
        Contrastive (SimCLR-style) encoder:
        - embedding_dim: dimension of the output representation
        """
        super(ContrastiveModel, self).__init__()
        # Simple convolutional encoder to get a single vector representation
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2), nn.ReLU()
        )
        # We will use adaptive pooling to get a fixed-length output (average pooling over time axis)
        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)
        # Projection head (linear layer) to output final embedding
        self.projection = nn.Linear(64, embedding_dim)
    
    def forward(self, x):
        # x: [B, C, L]
        feat_map = self.features(x)       # -> shape [B, 64, L_reduced]
        pooled = self.global_pool(feat_map)  # -> shape [B, 64, 1] (pooled over time)
        pooled = pooled.view(pooled.size(0), -1)  # -> [B, 64]
        emb = self.projection(pooled)     # -> [B, embedding_dim]
        return emb
    
    def _augment(self, x):
        # Apply a simple random augmentation: add small Gaussian noise
        noise = 0.01 * torch.randn_like(x)
        # (You could include other augmentations such as random scaling, cropping, etc.)
        return x + noise
    
    def compute_loss(self, x, target=None):
        """
        Compute contrastive loss for a batch of samples using SimCLR approach.
        """
        B = x.size(0)
        # Create two augmented views of each sample
        x_aug1 = self._augment(x)
        x_aug2 = self._augment(x)
        # Compute embeddings for both sets
        z1 = self.forward(x_aug1)  # [B, embedding_dim]
        z2 = self.forward(x_aug2)  # [B, embedding_dim]
        # Normalize embeddings (as is common in contrastive learning)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        # Compute similarity matrix between all pairs (2B x 2B matrix if we stack z1 and z2)
        # But we can compute pairwise similarities in a batched way for positives and negatives.
        # Let's form a big tensor Z of shape [2B, emb_dim] where first B are z1 and next B are z2.
        Z = torch.cat([z1, z2], dim=0)  # [2B, emb_dim]
        sim_matrix = torch.matmul(Z, Z.T)  # [2B, 2B] pairwise cosine similarities
        # We need to exclude self-similarity on the diagonal from negatives.
        # Create labels: for each i in 0..B-1 (for z1[i]), the positive match is i+B (z2 of same sample), and vice versa.
        labels = torch.arange(B, device=x.device)
        # Compute loss for z1 -> z2 and z2 -> z1 separately and average.
        loss = 0.0
        # Temperature for scaling similarities (optional, use 1.0 here for simplicity)
        tau = 0.1
        for i in range(B):
            # Positive similarity for pair (i, i+B)
            pos_sim = sim_matrix[i, i+B] / tau
            # Negatives: similarities of i (z1[i]) with all z2[j] for j != i
            neg_sims = torch.cat([sim_matrix[i, :i], sim_matrix[i, i+1:B], sim_matrix[i, B:]], dim=0) / tau
            # InfoNCE loss: -log(exp(pos_sim) / (exp(pos_sim) + sum(exp(neg_sims))))
            log_prob = pos_sim - torch.log(torch.exp(pos_sim) + torch.exp(neg_sims).sum())
            loss += -log_prob
            # Also do symmetric: z2[i] with z1 (where z2 index i corresponds to original sample i)
            j = i + B
            pos_sim2 = sim_matrix[j, i] / tau
            neg_sims2 = torch.cat([sim_matrix[j, :B], sim_matrix[j, B:j], sim_matrix[j, j+1:]], dim=0) / tau
            log_prob2 = pos_sim2 - torch.log(torch.exp(pos_sim2) + torch.exp(neg_sims2).sum())
            loss += -log_prob2
        loss = loss / (2 * B)
        return loss
    
    def get_embedding(self, x):
        """
        Get the embedding vector (after projection) for input x without augmentation.
        """
        self.eval()
        with torch.no_grad():
            emb = self.forward(x)  # [B, embedding_dim]
        return emb
