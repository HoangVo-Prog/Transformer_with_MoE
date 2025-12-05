# prepare_iwslt2015_en_vi.py

import os
from datasets import load_dataset


def write_list(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in lines:
            x = str(x).strip()
            if x:
                f.write(x + "\n")


def prepare_iwslt2015_en_vi(output_dir: str = "data"):
    """
    Tải dataset thainq107/iwslt2015-en-vi rồi xuất ra 6 file:
        data/train.vi
        data/train.en
        data/val.vi
        data/val.en
        data/test.vi
        data/test.en
    """

    print("Loading Hugging Face dataset thainq107/iwslt2015-en-vi")
    ds = load_dataset("thainq107/iwslt2015-en-vi")

    os.makedirs(output_dir, exist_ok=True)

    # Mappings: split Hugging Face -> prefix file
    split_map = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }

    for split_name, prefix in split_map.items():
        if split_name not in ds:
            continue

        split = ds[split_name]
        vi_list = split["vi"]
        en_list = split["en"]

        assert len(vi_list) == len(en_list), f"{split_name}: vi and en length mismatch before writing"

        out_vi = os.path.join(output_dir, f"{prefix}.vi")
        out_en = os.path.join(output_dir, f"{prefix}.en")

        kept = 0
        with open(out_vi, "w", encoding="utf-8") as f_vi, open(out_en, "w", encoding="utf-8") as f_en:
            for vi, en in zip(vi_list, en_list):
                vi = "" if vi is None else str(vi).strip()
                en = "" if en is None else str(en).strip()
                f_vi.write(vi + "\n")
                f_en.write(en + "\n")
                kept += 1

        print(f"{split_name}: original {len(vi_list)} pairs, wrote {kept} pairs to {prefix}.vi/.en")

    print(f"Saved text files into {output_dir}/")
    print("Done.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Thư mục để lưu các file train/val/test",
    )

    args = parser.parse_args()
    prepare_iwslt2015_en_vi(output_dir=args.output_dir)
