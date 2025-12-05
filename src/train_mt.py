# train_mt.py
import argparse
import math
import os
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model import MoETransformerMT, count_parameters
from data_module import create_dataloaders
from utils import get_device, set_seed, save_checkpoint

from prepare_vi_en_data import prepare_iwslt2015_en_vi


def read_lines(path: str):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def load_parallel_corpus_from_files(
    root: str = "data",
    train_prefix: str = "train",
    val_prefix: str = "val",
    test_prefix: str = "test",
    src_ext: str = "vi",
    tgt_ext: str = "en",
):
    train_src = read_lines(os.path.join(root, f"{train_prefix}.{src_ext}"))
    train_tgt = read_lines(os.path.join(root, f"{train_prefix}.{tgt_ext}"))

    val_src = read_lines(os.path.join(root, f"{val_prefix}.{src_ext}"))
    val_tgt = read_lines(os.path.join(root, f"{val_prefix}.{tgt_ext}"))

    test_src = read_lines(os.path.join(root, f"{test_prefix}.{src_ext}"))
    test_tgt = read_lines(os.path.join(root, f"{test_prefix}.{tgt_ext}"))

    assert len(train_src) == len(train_tgt)
    assert len(val_src) == len(val_tgt)
    assert len(test_src) == len(test_tgt)

    return train_src, train_tgt, val_src, val_tgt, test_src, test_tgt


# =========================
# 2. Tokenizer
# =========================

class SimpleVocab:
    def __init__(self, texts: List[str], min_freq: int = 1, specials=None):
        if specials is None:
            specials = ["<pad>", "<bos>", "<eos>", "<unk>"]

        self.specials = specials
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        freq = {}
        for line in texts:
            for tok in line.strip().split():
                freq[tok] = freq.get(tok, 0) + 1

        vocab_list = list(self.specials)
        for tok, c in freq.items():
            if c >= min_freq and tok not in vocab_list:
                vocab_list.append(tok)

        self.itos = vocab_list
        self.stoi = {t: i for i, t in enumerate(self.itos)}

        self.pad_id = self.stoi[self.pad_token]
        self.bos_id = self.stoi[self.bos_token]
        self.eos_id = self.stoi[self.eos_token]
        self.unk_id = self.stoi[self.unk_token]

    def encode(self, text: str, add_bos_eos: bool = False):
        toks = text.strip().split()
        ids = [self.stoi.get(t, self.unk_id) for t in toks]
        if add_bos_eos:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        toks = []
        for i in ids:
            if i == self.eos_id:
                break
            if i in (self.pad_id, self.bos_id):
                continue
            toks.append(self.itos[i])
        return " ".join(toks)

    def __len__(self):
        return len(self.itos)

    # thêm 2 hàm dưới để save - load state

    def to_state(self):
        return {
            "itos": self.itos,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
        }

    @classmethod
    def from_state(cls, state):
        obj = cls(texts=[])
        obj.itos = state["itos"]
        obj.stoi = {t: i for i, t in enumerate(obj.itos)}
        obj.pad_token = state["pad_token"]
        obj.bos_token = state["bos_token"]
        obj.eos_token = state["eos_token"]
        obj.unk_token = state["unk_token"]

        obj.pad_id = obj.stoi[obj.pad_token]
        obj.bos_id = obj.stoi[obj.bos_token]
        obj.eos_id = obj.stoi[obj.eos_token]
        obj.unk_id = obj.stoi[obj.unk_token]
        return obj

# =========================
# 3. Train, eval loop
# =========================

def train_one_epoch(
    model: MoETransformerMT,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    lambda_aux: float,
    grad_clip: float = 1.0,
):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in train_loader:
        src_ids, tgt_in, tgt_out, src_pad_mask, tgt_pad_mask = batch
        src_ids = src_ids.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)
        src_pad_mask = src_pad_mask.to(device)
        tgt_pad_mask = tgt_pad_mask.to(device)

        logits, aux_loss = model(
            src_ids=src_ids,
            tgt_ids_in=tgt_in,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )

        B, T, V = logits.size()
        loss_main = criterion(
            logits.view(B * T, V),
            tgt_out.view(B * T),
        )
        loss = loss_main + lambda_aux * aux_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # thống kê
        non_pad = tgt_out.ne(criterion.ignore_index).sum().item()
        total_loss += loss_main.item() * non_pad
        total_tokens += non_pad

    return total_loss / max(1, total_tokens)


@torch.no_grad()
def evaluate(
    model: MoETransformerMT,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    lambda_aux: float,
):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in val_loader:
        src_ids, tgt_in, tgt_out, src_pad_mask, tgt_pad_mask = batch
        src_ids = src_ids.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)
        src_pad_mask = src_pad_mask.to(device)
        tgt_pad_mask = tgt_pad_mask.to(device)

        logits, aux_loss = model(
            src_ids=src_ids,
            tgt_ids_in=tgt_in,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )

        B, T, V = logits.size()
        loss_main = criterion(
            logits.view(B * T, V),
            tgt_out.view(B * T),
        )
        loss = loss_main + lambda_aux * aux_loss

        non_pad = tgt_out.ne(criterion.ignore_index).sum().item()
        total_loss += loss_main.item() * non_pad
        total_tokens += non_pad

    return total_loss / max(1, total_tokens)


# =========================
# 4. Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_enc_layers", type=int, default=3)
    parser.add_argument("--num_dec_layers", type=int, default=3)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--n_experts", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lambda_aux", type=float, default=1e-2)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mt_moe.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("Using device:", device)
    
    
    # 4. Load dữ liệu, tạo DataLoader
    prepare_iwslt2015_en_vi(output_dir=args.data_dir)

    # 4.1 Load dữ liệu toy, bạn thay bằng load thật
    train_src, train_tgt, val_src, val_tgt, test_src, test_tgt = load_parallel_corpus_from_files(
        root=args.data_dir,          
        train_prefix="train",
        val_prefix="val",
        test_prefix="test",
        src_ext="vi",
        tgt_ext="en",
    )

    # 4.2 Xây vocab và tokenizer đơn giản
    src_vocab = SimpleVocab(train_src + val_src + test_src)
    tgt_vocab = SimpleVocab(train_tgt + val_tgt + test_tgt)
    
    os.makedirs("checkpoints", exist_ok=True)
    
    torch.save(src_vocab.to_state(), "checkpoints/src_vocab.pt")
    torch.save(tgt_vocab.to_state(), "checkpoints/tgt_vocab.pt")

    def src_tokenizer(text: str):
        return src_vocab.encode(text, add_bos_eos=False)

    def tgt_tokenizer(text: str):
        # BOS và EOS sẽ do collate thêm, nên ở đây không cần
        return tgt_vocab.encode(text, add_bos_eos=False)

    src_pad_id = src_vocab.pad_id
    tgt_pad_id = tgt_vocab.pad_id
    bos_id = tgt_vocab.bos_id
    eos_id = tgt_vocab.eos_id

    # 4.3 Tạo DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(
        train_src=train_src,
        train_tgt=train_tgt,
        val_src=val_src,
        val_tgt=val_tgt,
        test_src=test_src,
        test_tgt=test_tgt,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_pad_id=src_pad_id,
        tgt_pad_id=tgt_pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 4.4 Khởi tạo model
    model = MoETransformerMT(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_enc_layers,
        num_decoder_layers=args.num_dec_layers,
        d_ff=args.d_ff,
        n_experts=args.n_experts,
        max_len=args.max_len,
        dropout=args.dropout,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params}")

    # 4.5 Optimizer và loss
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_id)

    best_val_loss = float("inf")

    # 4.6 Vòng lặp train
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            lambda_aux=args.lambda_aux,
        )

        val_loss = evaluate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            lambda_aux=args.lambda_aux,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("  New best model, saving checkpoint...")
            save_checkpoint(
                path=args.checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
            )


if __name__ == "__main__":
    main()
