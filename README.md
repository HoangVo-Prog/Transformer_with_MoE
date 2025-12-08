# Transformer_with_MoE

## Environment setup

```bash
git clone https://github.com/HoangVo-Prog/Transformer_with_MoE.git
cd Transformer_with_MoE
pip install -r requirements.txt
```

## How to train the model

For more details about the config, please check the src/config

```bash
bash script/train.sh
```


## How to translate a specific sentence

Please make sure the checkpoint (model + vocal) you train is in ./checkpoints

```bash
python src/infer_mt.py \
    --checkpoint-dir /checkpoints \
    --sentences "Tôi sẽ cho bạn biết về công nghệ đó ." "Nhưng bây giờ , chúng ta có một công nghệ thực để làm việc này ."
```

Sample result

```cmd
Using device: cuda
Loading checkpoint from /kaggle/input/notebookf0049095da/Transformer_with_MoE/checkpoints
/kaggle/working/Transformer_with_MoE/src/model.py:69: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=False):
/kaggle/working/Transformer_with_MoE/src/model.py:69: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=False):
VI: Tôi sẽ cho bạn biết về công nghệ đó .
EN: I &apos;ll tell you about that technology .
----------------------------------------
VI: Nhưng bây giờ , chúng ta có một công nghệ thực để làm việc này .
EN: But now , we have a technology to do this .
----------------------------------------
```