import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class InfoNCELoss(nn.Module):
    """
    Bi-directional InfoNCE loss as described in Section 2.
    Combines alignment (cosine similarity) and uniformity regularization.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, pred_embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_embeddings: (B, D) normalized predicted embeddings
            target_embeddings: (B, D) normalized target embeddings from Y-Encoder
        Returns:
            Scalar bi-directional InfoNCE loss
        """
        # Normalize embeddings
        pred = F.normalize(pred_embeddings, dim=-1)
        target = F.normalize(target_embeddings, dim=-1)

        # Cosine similarity matrix scaled by temperature
        logits = torch.matmul(pred, target.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        # Bi-directional: pred->target and target->pred
        loss_pred_to_target = F.cross_entropy(logits, labels)
        loss_target_to_pred = F.cross_entropy(logits.T, labels)

        return (loss_pred_to_target + loss_target_to_pred) / 2.0


class Predictor(nn.Module):
    """
    VL-JEPA Predictor: maps (S_V, X_Q) -> S_Y_hat
    Uses Llama-style Transformer layers with NO causal mask (Section 3.1).
    Outputs are average-pooled over non-PAD tokens then projected.
    """
    def __init__(
        self,
        hidden_dim: int = 2048,
        num_layers: int = 8,
        num_heads: int = 16,
        embedding_dim: int = 1536,
        max_query_tokens: int = 512,
        vocab_size: int = 128256,  # Llama-3.2 tokenizer vocab size
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_query_tokens = max_query_tokens

        # Text token embedding (shared with Llama-3.2-1B tokenizer)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        # Linear projection from vision encoder dim to predictor hidden dim
        # V-JEPA 2 ViT-L outputs 1024-dim tokens; adjust if different
        self.vision_proj = nn.Linear(1024, hidden_dim)

        # Non-causal Transformer decoder layers (no causal mask)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation=nn.SiLU(),  # FIX: Pass the nn.SiLU() module instead of the string "silu"
            batch_first=True,
            norm_first=True,  # Pre-norm like Llama
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection to shared embedding space
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)

        # Learnable query tokens could also be used; here we use direct pooling
        self.pad_token_id = 0  # Adjust based on actual tokenizer

    def forward(
        self,
        visual_tokens: torch.Tensor,   # (B, N_v, 1024)
        query_token_ids: torch.Tensor,  # (B, N_q) padded to max_query_tokens
    ) -> torch.Tensor:
        """
        Returns:
            predicted_embedding: (B, embedding_dim)
        """
        # Project visual tokens
        vis_emb = self.vision_proj(visual_tokens)  # (B, N_v, hidden_dim)

        # Embed query tokens
        txt_emb = self.token_embedding(query_token_ids)  # (B, N_q, hidden_dim)

        # Concatenate: [visual; textual] — no causal masking
        combined = torch.cat([vis_emb, txt_emb], dim=1)  # (B, N_v+N_q, hidden_dim)

        # Self-attention over combined sequence (non-causal)
        # Using transformer decoder with memory=None equivalent via self-attn only
        attended = self.transformer(combined, memory=combined)

        # Average pool over non-PAD positions only
        pad_mask = (query_token_ids != self.pad_token_id)  # (B, N_q)
        # Visual tokens are never padded
        vis_mask = torch.ones(
            vis_emb.size(0), vis_emb.size(1),
            dtype=torch.bool, device=vis_emb.device
        )
        full_mask = torch.cat([vis_mask, pad_mask], dim=1)  # (B, N_v+N_q)

        # Masked average pooling
        attended_masked = attended * full_mask.unsqueeze(-1)
        pooled = attended_masked.sum(dim=1) / full_mask.sum(dim=1, keepdim=True).clamp(min=1)

        # Project to embedding space
        return self.output_proj(pooled)  # (B, embedding_dim)


class YEncoder(nn.Module):
    """
    Y-Encoder: embeds textual target Y into continuous latent space S_Y.
    Initialized from EmbeddingGemma-300M (Section 3.1).
    During training, uses LR multiplier of 0.05.
    """
    def __init__(self, embedding_dim: int = 1536, hidden_dim: int = 1024, vocab_size: int = 256000):
        super().__init__()
        # Placeholder for EmbeddingGemma-300M backbone
        # In practice, load pretrained weights here
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=16, dim_feedforward=hidden_dim * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, target_token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target_token_ids: (B, N_t) tokenized target text
        Returns:
            target_embedding: (B, embedding_dim)
        """
        emb = self.token_embedding(target_token_ids)
        encoded = self.encoder(emb)
        # Mean pooling over sequence
        pooled = encoded.mean(dim=1)
        return self.output_proj(pooled)


class VLJEPA(nn.Module):
    """
    Full VL-JEPA model combining Predictor + Y-Encoder.
    X-Encoder (V-JEPA 2 ViT-L) is kept frozen externally.
    Y-Decoder is NOT part of training (used only at inference).
    """
    def __init__(self, embedding_dim: int = 1536, y_encoder_lr_multiplier: float = 0.05):
        super().__init__()
        self.predictor = Predictor(embedding_dim=embedding_dim)
        self.y_encoder = YEncoder(embedding_dim=embedding_dim)
        self.criterion = InfoNCELoss()
        self.y_encoder_lr_multiplier = y_encoder_lr_multiplier

    def forward(
        self,
        visual_tokens: torch.Tensor,
        query_token_ids: torch.Tensor,
        target_token_ids: torch.Tensor,
    ) -> dict:
        pred_emb = self.predictor(visual_tokens, query_token_ids)
        target_emb = self.y_encoder(target_token_ids)
        loss = self.criterion(pred_emb, target_emb)
        return {
            "loss": loss,
            "predicted_embedding": pred_emb,
            "target_embedding": target_emb,
        }

    def get_param_groups(self, base_lr: float) -> list:
        """Y-Encoder params use 0.05x learning rate (Section 3.1)."""
        y_enc_params = list(self.y_encoder.parameters())
        other_params = [p for n, p in self.named_parameters()
                        if not n.startswith("y_encoder")]
        return [
            {"params": other_params, "lr": base_lr},
            {"params": y_enc_params, "lr": base_lr * self.y_encoder_lr_multiplier},
        ]


class SelectiveDecodingMonitor:
    """
    Implements selective decoding from Section 2.
    Monitors the continuous stream of predicted embeddings S_Y_hat
    and triggers decoding only when semantic shift exceeds threshold.
    Reduces decoding operations by ~2.85x vs uniform decoding.
    """
    def __init__(self, window_size: int = 5, variance_threshold: float = 0.1):
        self.window_size = window_size
        self.variance_threshold = variance_threshold
        self.buffer: List[torch.Tensor] = []

    @torch.no_grad()
    def should_decode(self, embedding: torch.Tensor) -> bool:
        """
        Args:
            embedding: (D,) single predicted embedding from current frame/window
        Returns:
            True if y-decoder should be invoked
        """
        self.buffer.append(embedding.cpu())
        if len(self.buffer) < self.window_size:
            return True  # Always decode during warmup

        # Keep sliding window
        self.buffer = self.buffer[-self.window_size:]
        window = torch.stack(self.buffer)  # (W, D)

        # Local variance as semantic change indicator
        variance = window.var(dim=0).mean().item()
        return variance > self.variance_threshold

    def reset(self):
        self.buffer.clear()


# === Usage Example ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VLJEPA(embedding_dim=1536).to(device)
    optimizer = torch.optim.AdamW(model.get_param_groups(base_lr=5e-5))

    # Simulated batch
    B = 4
    visual_tokens = torch.randn(B, 196, 1024, device=device)      # 14x14 ViT patches
    query_ids = torch.randint(1, 1000, (B, 64), device=device)     # Short queries padded
    target_ids = torch.randint(1, 1000, (B, 128), device=device)   # Target captions

    out = model(visual_tokens, query_ids, target_ids)
    print(f"Training Loss: {out['loss'].item():.4f}")

    # Selective decoding demo
    monitor = SelectiveDecodingMonitor(window_size=5, variance_threshold=0.08)
    decode_count = 0
    for t in range(30):
        emb = out["predicted_embedding"][0] + torch.randn_like(out["predicted_embedding"][0]) * 0.01 * t
        if monitor.should_decode(emb):
            decode_count += 1
            # y_decoder(emb) would be called here
    print(f"Selective decoding triggered {decode_count}/30 times "
          f"({30/max(decode_count,1):.2f}x reduction)")