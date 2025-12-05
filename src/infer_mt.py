# infer_mt.py
import argparse
from typing import List

import torch

from model import MoETransformerMT
from utils import get_device, load_checkpoint


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


def load_parallel_corpus_dummy():
    train_src = [
        "xin chao the gioi",
        "toi dang thu train mo hinh",
        "day la mot vi du don gian",
    ]
    train_tgt = [
        "hello world",
        "i am trying to train a model",
        "this is a simple example",
    ]
    val_src = train_src
    val_tgt = train_tgt
    test_src = train_src
    test_tgt = train_tgt
    return train_src, train_tgt, val_src, val_tgt, test_src, test_tgt


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
        "--checkpoint",
        type=str,
        default="checkpoints/mt_moe.pt",
        help="Path tới best checkpoint",
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
    train_src, train_tgt, val_src, val_tgt, test_src, test_tgt = load_parallel_corpus_dummy()

    src_vocab = SimpleVocab(train_src + val_src + test_src)
    tgt_vocab = SimpleVocab(train_tgt + val_tgt + test_tgt)

    src_pad_id = src_vocab.pad_id
    tgt_pad_id = tgt_vocab.pad_id
    bos_id = tgt_vocab.bos_id
    eos_id = tgt_vocab.eos_id

    # 2. Khởi tạo model với đúng cấu hình như lúc train
    d_model = 256
    nhead = 4
    num_enc_layers = 3
    num_dec_layers = 3
    d_ff = 1024
    n_experts = 8
    max_len_model = 128
    dropout = 0.1

    model = MoETransformerMT(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_enc_layers,
        num_decoder_layers=num_dec_layers,
        d_ff=d_ff,
        n_experts=n_experts,
        max_len=max_len_model,
        dropout=dropout,
    ).to(device)

    # 3. Load checkpoint tốt nhất
    print(f"Loading checkpoint from {args.checkpoint}")
    load_checkpoint(args.checkpoint, model, optimizer=None)

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
