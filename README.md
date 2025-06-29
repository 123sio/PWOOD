# xxx

## Installation
``` shell
conda create -n mm python==3.8 -y
conda activate mm
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -U openmim
mim install mmcv-full
mim install mmdet\<3.0.0
pip install scikit-learn
pip install prettytable
# For Point & Mix
cd mmrotate
pip install -v -e .
```

## Train
``` shell
#2 GPU
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --master_port=25510 \
train.py configs_dota15/xxx/xxx.py \
--launcher pytorch \
--work-dir work_dir/xxx/

```

## Test
``` shell
python test.py configs_dota15/xxx/xxx.py work_dir/xxx/xxx.pth 
```

## Weight

### DOTA- v1.0
｜Labeled Data | mAP | Config | Model | Log |
| :-----------: | :--: |:-----: | :----: | :-----:|
| 10% | 56.92 | [semi_h2rv2_adamw_dotav1_10p.py](https://github.com/123sio/PWOOD/blob/HBox/configs_dota15/pwood/dotav1/semi_h2rv2_adamw_dotav1_10p.py) | - | - | 
| 20% | 62.93 | [semi_h2rv2_adamw_dotav1_20p.py](https://github.com/123sio/PWOOD/blob/HBox/configs_dota15/pwood/dotav1/semi_h2rv2_adamw_dotav1_20p.py) | - | - | 
| 30% | 65.42 | [semi_h2rv2_adamw_dotav1_30p.py](https://github.com/123sio/PWOOD/blob/HBox/configs_dota15/pwood/dotav1/semi_h2rv2_adamw_dotav1_30p.py) | - | - | 

### DOTA- v1.5
｜Labeled Data | mAP | Config | Model | Log |
| :-----------: | :--: |:-----: | :----: | :-----:|
| 10% | 52.87 | [semi_h2rv2_adamw_dota15_10p.py](https://github.com/123sio/PWOOD/blob/HBox/configs_dota15/pwood/semi_h2rv2_adamw_dota15_10p.py) | - | - | 
| 20% | 59.36 | [semi_h2rv2_adamw_dota15_20p.py](https://github.com/123sio/PWOOD/blob/HBox/configs_dota15/pwood/semi_h2rv2_adamw_dota15_20p.py) | - | - | 
| 30% | 61.58 | [semi_h2rv2_adamw_dota15_30p.py](https://github.com/123sio/PWOOD/blob/HBox/configs_dota15/pwood/semi_h2rv2_adamw_dota15_30p.py) | - | - | 

### DOTA- v2.0
｜Labeled Data | mAP | Config | Model | Log |
| :-----------: | :--: |:-----: | :----: | :-----:|
| 10% | 31.03 | [semi_h2rv2_adamw_dota2_10p.py](https://github.com/123sio/PWOOD/blob/HBox/configs_dota15/pwood/dotav2/semi_h2rv2_adamw_dota2_10p.py) | - | - | 
| 20% | 36.39 | [semi_h2rv2_adamw_dota2_20p.py](https://github.com/123sio/PWOOD/blob/HBox/configs_dota15/pwood/dotav2/semi_h2rv2_adamw_dota2_20p.py) | - | - | 
| 30% | 40.27 | [semi_h2rv2_adamw_dota2_30p.py](https://github.com/123sio/PWOOD/blob/HBox/configs_dota15/pwood/dotav2/semi_h2rv2_adamw_dota2_30p.py) | - | - | 
### DIOR
｜Labeled Data | mAP | Config | Model | Log |
| :-----------: | :--: |:-----: | :----: | :-----:|
| 10% | 54.33 | [semi_h2rv2_adamw_dior_10p.py](https://github.com/123sio/PWOOD/blob/HBox/configs_dota15/pwood/dior/semi_h2rv2_adamw_dior_10p.py) | - | - | 
| 20% | 57.89 | [semi_h2rv2_adamw_dior_20p.py](https://github.com/123sio/PWOOD/blob/HBox/configs_dota15/pwood/dior/semi_h2rv2_adamw_dior_20p.py) | - | - | 
| 30% | 60.42 | [semi_h2rv2_adamw_dior_30p.py](https://github.com/123sio/PWOOD/blob/HBox/configs_dota15/pwood/dior/semi_h2rv2_adamw_dior_30p.py) | - | - | 
