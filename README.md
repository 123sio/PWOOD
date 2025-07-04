
<p align="center">
    <h1 align="center">Partial Weakly-Supervised Oriented Object Detection</h1>
    <p align="center">
    <a href='' style='text-decoration: none' >Mingxin Liu</a><sup></sup>&ensp;
    <a href='https://scholar.google.com/citations?user=rQbW67AAAAAJ' style='text-decoration: none' >Peiyuan Zhang</a><sup></sup>&ensp;
    <a href='' style='text-decoration: none' >Yuan Liu</a><sup></sup>&ensp;
    <a href='' style='text-decoration: none' >Wei Zhang</a><sup></sup>&ensp;
    <a href='https://scholar.google.com/citations?user=v-aQ8GsAAAAJ' style='text-decoration: none' >Yue Zhou</a><sup></sup>&ensp;
    <a href='' style='text-decoration: none' >Ning Liao</a><sup></sup>&ensp;
    <a href='' style='text-decoration: none' >Ziyang Gong</a><sup></sup>&ensp;
    <a href='https://scholar.google.com/citations?user=6XibZaYAAAAJ' style='text-decoration: none' >Junwei Luo</a><sup></sup>&ensp;
    <a href='' style='text-decoration: none' >Zhirui Wang</a><sup></sup>&ensp;
    <a href='https://scholar.google.com/citations?user=OYtSc4AAAAAJ' style='text-decoration: none' >Yi Yu</a><sup></sup>&ensp;
    <a href='https://yangxue.site/' style='text-decoration: none' >Xue Yang</a><sup></sup>&ensp;
    <div align="center">
      <a href='https://arxiv.org/abs/2507.02751'><img src='https://img.shields.io/badge/arXiv-2502.04268-brown.svg?logo=arxiv&logoColor=white'></a>
     <a href='https://huggingface.co/Xm4nQ8/weight'><img src='https://img.shields.io/badge/HuggingFace-Model-yellow.svg?logo=HuggingFace&logoColor=white'></a>
	  </div>
    </div>
     </p>
</p>

## Introduction
We propose the first Partial Weakly-Supervised Oriented Object Detection (PWOOD) framework based on partially weak annotations (horizontal boxes or single points), which can efficiently leverage large amounts of unlabeled data, significantly outperforming weakly supervised algorithms trained with partially weak annotations, and also offers a lower cost solution.

<img src="fig/pipline.png" alt="framework" width="100%" />

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

# For Point branch
cd mmrotate
pip install -v -e .
```

## Data Preparation
### DOTA
#### 1. Labeled/Unlabeled Data Division
To divide the DOTA- v1.0/v1.5 dataset into labeled and unlabeled data, please refer to [Data preparation of SOOD
](https://github.com/HamPerdredes/SOOD).

To divide the DOTA- v2.0 into labeled and unlabeled data, please refer to [data_list/dotav2](https://github.com/123sio/PWOOD/tree/HBox/data_list/dotav2).

#### 2. Data Split
For details on how to split the DOTA dataset into patches, please refer to the [official implementation](https://github.com/open-mmlab/mmrotate/blob/main/tools/data/dota/README.md) .

After split, the data folder should be organized as follows:
``` 
split_ss_dota_vxx
├── train
│   ├── images
│   └── annfiles
├── val
│   ├── images
│   └── annfiles
├── train_xx_labeled
│   ├── images
│   └── annfiles
└──train_xx_unlabeled
    ├── images
    └── annfiles
```

### DIOR
To divide the DIOR into labeled and unlabeled data, please refer to [data_list/dior](https://github.com/123sio/PWOOD/tree/HBox/data_list/dior).

## Train
```bash
#2 GPU
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --nproc_per_node=2 --master_port=25510 \
train.py configs_dota15/xxx/xxx.py \
--launcher pytorch \
--work-dir work_dir/xxx/

```

## Test
```bash
python test.py configs_dota15/xxx/xxx.py work_dir/xxx/xxx.pth 
```

## Weight

### DOTA- v1.0
Labeled Data | mAP | Config | Model | Log |
| :-----------: | :--: |:-----: | :----: | :-----:|
| 20% | 62.93 | [semi_h2rv2_adamw_dotav1_20p.py](https://github.com/123sio/PWOOD/blob/HBox/configs_dota15/pwood/dotav1/semi_h2rv2_adamw_dotav1_20p.py) | [best_0.629314_mAP.pth](https://huggingface.co/Xm4nQ8/weight/blob/main/work_dir_h/PWOOD/dota1_0/20p/best_0.629314_mAP.pth) | [dotav1_20p_log](https://huggingface.co/Xm4nQ8/weight/blob/main/work_dir_h/PWOOD/dota1_0/20p/20250303_111353.log.json) | 
| 30% | 65.42 | [semi_h2rv2_adamw_dotav1_30p.py](https://github.com/123sio/PWOOD/blob/HBox/configs_dota15/pwood/dotav1/semi_h2rv2_adamw_dotav1_30p.py) | [best_0.654153_mAP.pth](https://huggingface.co/Xm4nQ8/weight/blob/main/work_dir_h/PWOOD/dota1_0/30p/best_0.654153_mAP.pth) | [dotav1_30p_log](https://huggingface.co/Xm4nQ8/weight/blob/main/work_dir_h/PWOOD/dota1_0/30p/20250310_193742.log.json) | 

### DOTA- v1.5
Labeled Data | mAP | Config | Model | Log |
| :-----------: | :--: |:-----: | :----: | :-----:|
| 10% | 52.87 | [semi_h2rv2_adamw_dota15_10p.py](https://github.com/123sio/PWOOD/blob/HBox/configs_dota15/pwood/semi_h2rv2_adamw_dota15_10p.py) | [best_0.528748_mAP.pth](https://huggingface.co/Xm4nQ8/weight/blob/main/work_dir_h/PWOOD/gmm/10p_lr/best_0.528748_mAP.pth) | [dotav15_10p_log](https://huggingface.co/Xm4nQ8/weight/blob/main/work_dir_h/PWOOD/gmm/10p_lr/20250219_224359.log.json) | 
| 20% | 59.36 | [semi_h2rv2_adamw_dota15_20p.py](https://github.com/123sio/PWOOD/blob/HBox/configs_dota15/pwood/semi_h2rv2_adamw_dota15_20p.py) | [best_0.593614_mAP.pth](https://huggingface.co/Xm4nQ8/weight/blob/main/work_dir_h/PWOOD/gmm/best_0.593614_mAP.pth) | [dotav15_20p_log](https://huggingface.co/Xm4nQ8/weight/blob/main/work_dir_h/PWOOD/gmm/20250217_202030.log.json) | 
| 30% | 61.58 | [semi_h2rv2_adamw_dota15_30p.py](https://github.com/123sio/PWOOD/blob/HBox/configs_dota15/pwood/semi_h2rv2_adamw_dota15_30p.py) | [best_0.615836_mAP.pth](https://huggingface.co/Xm4nQ8/weight/blob/main/work_dir_h/PWOOD/gmm/30p_lr/best_0.615836_mAP.pth) | [dotav15_30p_log](https://huggingface.co/Xm4nQ8/weight/blob/main/work_dir_h/PWOOD/gmm/30p_lr/20250219_223950.log.json) | 

### DOTA- v2.0
Labeled Data | mAP | Config | Model | Log |
| :-----------: | :--: |:-----: | :----: | :-----:|
| 10% | 31.30 | [semi_h2rv2_adamw_dota2_10p.py](https://github.com/123sio/PWOOD/blob/HBox/configs_dota15/pwood/dotav2/semi_h2rv2_adamw_dota2_10p.py)| [best_0.310266_mAP.pth](https://huggingface.co/Xm4nQ8/weight/blob/main/work_dir_h/PWOOD/dotav2/10p/best_0.310266_mAP.pth)| [dotav2_10p_log](https://huggingface.co/Xm4nQ8/weight/blob/main/work_dir_h/PWOOD/dotav2/10p/20250313_224047.log.json)|
| 20% | 36.39 | [semi_h2rv2_adamw_dota2_20p.py](https://github.com/123sio/PWOOD/blob/HBox/configs_dota15/pwood/dotav2/semi_h2rv2_adamw_dota2_20p.py)| [best_0.363926_mAP.pth](https://huggingface.co/Xm4nQ8/weight/blob/main/work_dir_h/PWOOD/dotav2/best_0.363926_mAP.pth) | [dotav2_20p_log](https://huggingface.co/Xm4nQ8/weight/blob/main/work_dir_h/PWOOD/dotav2/20250304_174131.log.json) |
| 30% | 40.27 | [semi_h2rv2_adamw_dota2_30p.py](https://github.com/123sio/PWOOD/blob/HBox/configs_dota15/pwood/dotav2/semi_h2rv2_adamw_dota2_30p.py)| [best_0.402659_mAP.pth](https://huggingface.co/Xm4nQ8/weight/blob/main/work_dir_h/PWOOD/dotav2/30p/pro_data/best_0.402659_mAP.pth) | [dotav2_30p_log](https://huggingface.co/Xm4nQ8/weight/blob/main/work_dir_h/PWOOD/dotav2/30p/pro_data/20250321_142715.log.json) |

### DIOR
Labeled Data | mAP | Config | Model | Log |
| :-----------: | :--: |:-----: | :----: | :-----:|
| 10% | 54.33 | [semi_h2rv2_adamw_dior_10p.py](https://github.com/123sio/PWOOD/blob/HBox/configs_dota15/pwood/dior/semi_h2rv2_adamw_dior_10p.py) | [best_0.543296_mAP.pth](https://huggingface.co/Xm4nQ8/weight/blob/main/work_dir_h/PWOOD/dior/gmm/10p/best_0.543296_mAP.pth) | [doir_10p_log](https://huggingface.co/Xm4nQ8/weight/blob/main/work_dir_h/PWOOD/dior/gmm/10p/20250227_202752.log.json) | 
| 20% | 57.89 | [semi_h2rv2_adamw_dior_20p.py](https://github.com/123sio/PWOOD/blob/HBox/configs_dota15/pwood/dior/semi_h2rv2_adamw_dior_20p.py) | [best_0.578923_mAP.pth](https://huggingface.co/Xm4nQ8/weight/blob/main/work_dir_h/PWOOD/dior/gmm/20p/best_0.578923_mAP.pth) | [dior_20p_log](https://huggingface.co/Xm4nQ8/weight/blob/main/work_dir_h/PWOOD/dior/gmm/20p/20250227_205200.log.json) | 
| 30% | 60.42 | [semi_h2rv2_adamw_dior_30p.py](https://github.com/123sio/PWOOD/blob/HBox/configs_dota15/pwood/dior/semi_h2rv2_adamw_dior_30p.py) | [best_0.604248_mAP.pth](https://huggingface.co/Xm4nQ8/weight/blob/main/work_dir_h/PWOOD/dior/gmm/30p/best_0.604248_mAP.pth) | [dior_30p_log](https://huggingface.co/Xm4nQ8/weight/blob/main/work_dir_h/PWOOD/dior/gmm/30p/20250301_071406.log.json) | 

## Guide 
If you need Point version, please switch to the [Point branch](https://github.com/123sio/PWOOD/tree/Point).

## Citation
```bibtex
@misc{liu2025partialweaklysupervisedorientedobject,
      title={Partial Weakly-Supervised Oriented Object Detection}, 
      author={Mingxin Liu and Peiyuan Zhang and Yuan Liu and Wei Zhang and Yue Zhou and Ning Liao and Ziyang Gong and Junwei Luo and Zhirui Wang and Yi Yu and Xue Yang},
      year={2025},
      eprint={2507.02751},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.02751}, 
}

```


