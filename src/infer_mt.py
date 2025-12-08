# infer_mt.py
import argparse
from typing import List

import torch

from model import MoETransformerMT
from utils import get_device, load_checkpoint
from train_mt import SimpleVocab
import os

from config import (
    D_MODEL, NHEAD, NUM_ENC_LAYERS, NUM_DEC_LAYERS,
    D_FF, N_EXPERTS, MAX_LEN, DROPOUT,
)


@torch.no_grad()
def greedy_decode(
    model: MoETransformerMT,
    src_ids: torch.Tensor,
    src_pad_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int = 60,
):
    """
    src_ids: [B, S]
    src_pad_mask: [B, S] boolean
    """
    model.eval()
    device = src_ids.device
    B = src_ids.size(0)

    tgt_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

    for _ in range(max_len):
        logits, _ = model(
            src_ids=src_ids,
            tgt_ids_in=tgt_ids,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=None,
        )
        next_token_logits = logits[:, -1, :]  # [B, V]
        next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)  # [B, 1]

        tgt_ids = torch.cat([tgt_ids, next_tokens], dim=1)

        if (next_tokens == eos_id).all():
            break

    return tgt_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Path tới best checkpoint",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="mt_moe.pt",
        help="MoE checkpoint file name",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=60,
        help="Độ dài tối đa câu dịch",
    )
    args = parser.parse_args()

    device = get_device()
    print("Using device:", device)

    # 1. Build vocab 
    src_state = torch.load(f"{args.checkpoint_dir}/src_vocab.pt", map_location="cpu")
    tgt_state = torch.load(f"{args.checkpoint_dir}/tgt_vocab.pt", map_location="cpu")

    src_vocab = SimpleVocab.from_state(src_state)
    tgt_vocab = SimpleVocab.from_state(tgt_state)

    src_pad_id = src_vocab.pad_id
    tgt_pad_id = tgt_vocab.pad_id
    bos_id = tgt_vocab.bos_id
    eos_id = tgt_vocab.eos_id
    
    # 2. Khởi tạo model
    model = MoETransformerMT(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENC_LAYERS,
        num_decoder_layers=NUM_DEC_LAYERS,
        d_ff=D_FF,
        n_experts=N_EXPERTS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    ).to(device)

    # 3. Load checkpoint tốt nhất
    print(f"Loading checkpoint from {args.checkpoint}")
    load_checkpoint(os.path.join(args.checkpoint_dir, args.checkpoint_name), model, optimizer=None)

    # 4. Hàm encode và decode sử dụng vocab đã lưu
    def encode_src(text: str):
        return src_vocab.encode(text, add_bos_eos=False)

    def decode_tgt(ids: List[int]) -> str:
        return tgt_vocab.decode(ids)

    # 5. Ví dụ vài câu tiếng Việt cần dịch
    examples = [
        "xin chao ban",
        "hom nay troi dep",
        "toi dang hoc mo hinh transformer",
    ]

    for sent in examples:
        ids = encode_src(sent)
        src_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        src_pad_mask = src_tensor.eq(src_pad_id)

        pred_ids = greedy_decode(
            model=model,
            src_ids=src_tensor,
            src_pad_mask=src_pad_mask,
            bos_id=bos_id,
            eos_id=eos_id,
            max_len=args.max_len,
        )

        pred_ids = pred_ids[0].tolist()
        translated = decode_tgt(pred_ids)
        print("VI:", sent)
        print("EN:", translated)
        print("-" * 40)


if __name__ == "__main__":
    main()
