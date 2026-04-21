# ACV project II info

## Basic Info
- the codes are tested using Python==3.10 Pytorch==2.8 
- codebase: https://github.com/yzou2/CRST (actually doesn't help much)

## Getting Started
### Prepare your dataset
- original dataset can be downloaded from the links in ./[original_datasets](original_datasets)/datasource
- make sure your data follows the structure like this:     
```tree
office_31
├── amazon/ (Source)
│   ├── back_pack/
│   └── ...
└── webcam/ (Target)
    ├── back_pack/
    └── ...
```
### Self training part
you can use [ST.py](ST.py) to start training, check args for more option

#### vanilla ST (example)
```shell
python ST.py --arch resnet50 --method ST --src_path ./original_datasets/office_31/amazon --tgt_path ./original_datasets/office_31/webcam --apply_aug --save_dir ./experiments/amazon_to_webcam_ST
```
#### CBST (example)
```shell
python ST.py --method CBST --src_path ./original_datasets/office_31/amazon --tgt_path ./original_datasets/office_31/webcam --apply_aug --init_portion 0.2 --portion_step 0.05 --max_portion 0.8 --lr 5e-4 --save_dir ./experiments/amazon_to_webcam_CBST
```

#### CRST (example)
```shell
   python ST.py --method CRST --src_path ./original_datasets/office_31/amazon --tgt_path ./original_datasets/office_31/webcam --apply_aug --init_portion 0.2 --portion_step 0.05 --max_portion 0.8 --lr 5e-4 --alpha 0.1 --beta 0.1 --save_dir ./experiments/amazon_to_webcam_CRST
```

代码实现我也不知道对不对基本内容我写在注释了      

我个人测试了ST部分 在默认args设置下显存占用大概4G (Resnet-50), 4轮 ST过程 UDA 有效，可以继续加epoch

