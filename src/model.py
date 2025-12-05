import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== MoE components =====

class Top2Router(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.n_experts = n_experts
        self.router = nn.Linear(d_model, n_experts)

    def forward(self, x):
        """
        x: [B, S, D]
        return:
          top2_val: [B, S, 2]
          top2_idx: [B, S, 2]
          gate:     [B, S, n_experts]
        """
        logits = self.router(x)              # [B, S, n_experts]
        gate = F.softmax(logits, dim=-1)
        top2_val, top2_idx = torch.topk(gate, k=2, dim=-1)
        return top2_val, top2_idx, gate


class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.ffn(x)


class MoELayer(nn.Module):
    def __init__(self, d_model=256, d_ff=1024, n_experts=8):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts

        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff) for _ in range(n_experts)]
        )
        self.router = Top2Router(d_model, n_experts)

    def forward(self, x):
        """
        x: [B, S, D]
        return:
          out: [B, S, D]
          aux_loss: scalar tensor
        """
        top2_val, top2_idx, gate = self.router(x)

        B, S, D = x.shape
        out = torch.zeros_like(x)

        aux_loss = self._aux_loss(gate)

        # Naive implementation, đủ cho batch nhỏ trên Kaggle
        for b in range(B):
            for s in range(S):
                token = x[b, s]  # [D]
                y = 0.0
                for k in range(2):
                    e_idx = top2_idx[b, s, k].item()
                    w = top2_val[b, s, k]
                    y = y + w * self.experts[e_idx](token)
                out[b, s] = y

        return out, aux_loss

    def _aux_loss(self, gate):
        """
        gate: [B, S, n_experts]
        Load balancing loss kiểu GShard rất đơn giản
        """
        # mean trên batch và sequence
        expert_prob = gate.mean(dim=(0, 1))        # [n_experts]
        balance_loss = torch.sum(expert_prob * torch.log(expert_prob + 1e-9))
        return balance_loss


# ===== Positional encoding =====

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [B, S, D]
        """
        S = x.size(1)
        return x + self.pe[:, :S]


# ===== Encoder / Decoder layers =====

class EncoderMoELayer(nn.Module):
    def __init__(self, d_model=256, nhead=4, d_ff=1024, n_experts=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True,
        )
        self.moe = MoELayer(d_model=d_model, d_ff=d_ff, n_experts=n_experts)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        src: [B, S, D]
        """
        attn_out, _ = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = src + self.dropout(attn_out)
        src = self.ln1(src)

        moe_out, aux_loss = self.moe(src)
        src = src + self.dropout(moe_out)
        src = self.ln2(src)

        return src, aux_loss


class DecoderMoELayer(nn.Module):
    def __init__(self, d_model=256, nhead=4, d_ff=1024, n_experts=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True,
        )

        self.moe = MoELayer(d_model=d_model, d_ff=d_ff, n_experts=n_experts)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        tgt: [B, T, D]
        memory: [B, S, D]  từ encoder
        """
        # Masked self attention (causal)
        self_attn_out, _ = self.self_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = tgt + self.dropout(self_attn_out)
        tgt = self.ln1(tgt)

        # Cross attention
        cross_attn_out, _ = self.cross_attn(
            tgt,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout(cross_attn_out)
        tgt = self.ln2(tgt)

        # MoE FFN
        moe_out, aux_loss = self.moe(tgt)
        tgt = tgt + self.dropout(moe_out)
        tgt = self.ln3(tgt)

        return tgt, aux_loss


# ===== Full Transformer MT with MoE FFN =====

class MoETransformerMT(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=1024,
        n_experts=8,
        max_len=128,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len+5)

        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            EncoderMoELayer(d_model, nhead, d_ff, n_experts, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            DecoderMoELayer(d_model, nhead, d_ff, n_experts, dropout)
            for _ in range(num_decoder_layers)
        ])

        self.ln_enc = nn.LayerNorm(d_model)
        self.ln_dec = nn.LayerNorm(d_model)

        # Final LM head
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

    def generate_square_subsequent_mask(self, sz: int, device=None):
        # causal mask cho decoder
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        if device is not None:
            mask = mask.to(device)
        return mask

    def forward(
        self,
        src_ids,
        tgt_ids_in,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
    ):
        """
        src_ids: [B, S]
        tgt_ids_in: [B, T]  (chuỗi dịch input shift right)
        return:
          logits: [B, T, vocab_tgt]
          total_aux_loss: scalar
        """
        device = src_ids.device
        B, S = src_ids.shape
        _, T = tgt_ids_in.shape

        # Embedding + positional
        src = self.src_emb(src_ids) * math.sqrt(self.d_model)
        src = self.pos_enc(src)

        tgt = self.tgt_emb(tgt_ids_in) * math.sqrt(self.d_model)
        tgt = self.pos_enc(tgt)

        total_aux_loss = 0.0

        # Encoder
        memory = src
        for layer in self.encoder_layers:
            memory, aux_loss = layer(
                memory,
                src_mask=None,
                src_key_padding_mask=src_key_padding_mask,
            )
            total_aux_loss = total_aux_loss + aux_loss
        memory = self.ln_enc(memory)

        # Decoder
        # causal mask cho decoder
        tgt_mask = self.generate_square_subsequent_mask(T, device=device)

        out = tgt
        for layer in self.decoder_layers:
            out, aux_loss = layer(
                out,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            total_aux_loss = total_aux_loss + aux_loss
        out = self.ln_dec(out)

        logits = self.output_proj(out)
        return logits, total_aux_loss


# ===== Helper: đếm tham số =====

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
