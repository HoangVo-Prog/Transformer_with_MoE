import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Callable, Tuple, Optional


# 1. Dataset cho bài toán dịch vi-en
class TranslationDataset(Dataset):
    def __init__(
        self,
        src_texts: List[str],
        tgt_texts: List[str],
        src_tokenizer: Callable[[str], List[int]],
        tgt_tokenizer: Callable[[str], List[int]],
        max_len: int = 128
    ):
        assert len(src_texts) == len(tgt_texts)

        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tok = src_tokenizer
        self.tgt_tok = tgt_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src = self.src_texts[idx]
        tgt = self.tgt_texts[idx]

        src_ids = self.src_tok(src)[: self.max_len]
        tgt_ids = self.tgt_tok(tgt)[: self.max_len]

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


# 2. Collate function cho một batch dịch
def collate_translation_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    src_pad_id: int,
    tgt_pad_id: int,
    bos_id: int,
    eos_id: int
):
    src_seqs, tgt_seqs = zip(*batch)

    # Thêm BOS và EOS cho target
    tgt_seqs = [torch.cat([torch.tensor([bos_id]), t, torch.tensor([eos_id])]) for t in tgt_seqs]

    src_lens = [len(s) for s in src_seqs]
    tgt_lens = [len(t) for t in tgt_seqs]

    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    # Tạo batch padded
    src_batch = torch.full((len(batch), max_src), src_pad_id, dtype=torch.long)
    tgt_batch = torch.full((len(batch), max_tgt), tgt_pad_id, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
        src_batch[i, : len(s)] = s
        tgt_batch[i, : len(t)] = t

    # Chuẩn bị input và output cho decoder
    tgt_in = tgt_batch[:, :-1]
    tgt_out = tgt_batch[:, 1:]

    src_pad_mask = src_batch.eq(src_pad_id)
    tgt_pad_mask = tgt_in.eq(tgt_pad_id)

    return src_batch, tgt_in, tgt_out, src_pad_mask, tgt_pad_mask


# 3. Hàm tạo DataLoader
def create_dataloaders(
    train_src: List[str],
    train_tgt: List[str],
    val_src: List[str],
    val_tgt: List[str],
    test_src: List[str],
    test_tgt: List[str],
    src_tokenizer,
    tgt_tokenizer,
    src_pad_id: int,
    tgt_pad_id: int,
    bos_id: int,
    eos_id: int,
    max_len: int = 128,
    batch_size: int = 64,
    num_workers: int = 2
):
    train_ds = TranslationDataset(
        train_src,
        train_tgt,
        src_tokenizer,
        tgt_tokenizer,
        max_len=max_len
    )

    val_ds = TranslationDataset(
        val_src,
        val_tgt,
        src_tokenizer,
        tgt_tokenizer,
        max_len=max_len
    )

    test_ds = TranslationDataset(
        test_src,
        test_tgt,
        src_tokenizer,
        tgt_tokenizer,
        max_len=max_len
    )

    collate_fn = lambda batch: collate_translation_batch(
        batch,
        src_pad_id,
        tgt_pad_id,
        bos_id,
        eos_id
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader
