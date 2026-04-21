# ACV project II info

## Basic Info
- 代码经过以下测试环境测试 Python==3.10 Pytorch==2.8 
- codebase: https://github.com/yzou2/CRST (actually doesn't help much)
- 当使用Resnet-50作为backbones时，batch=32, 224*224分辨率，在office-31数据集上显存消耗大概4G

## Getting Started
### Prepare your dataset
- 原始的数据集可以从这里下载 ./[original_datasets](original_datasets)/datasource
- 请确保训练时数据结构如下所示:     
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
[ST.py](ST.py) 是主训练函数，可以查看ags来了解可调参数

#### vanilla ST 
- 源域损失 (Source Domain Loss)使用带标签的源域数据进行的标准交叉熵损失：

$$L_{source} = - \sum_{s=1}^S \sum_{n=1}^N \left[ y_{s,n} \cdot \log(p_n(w; I_s)) \right]$$
- 目标域损失 (Target Domain Loss)使用生成的伪标签 $\hat{y}$ 进行自监督训练的交叉熵损失：

$$L_{target} = - \sum_{t=1}^T \sum_{n=1}^N \left[ \hat{y}_{t,n} \cdot \log(p_n(w; I_t)) \right]$$
- 正则化项 (Regularization Term)通过 $L_1$ 范数控制伪标签的选择量：

$$L_{reg} = - \sum_{t=1}^T \sum_{n=1}^N \left[ k \cdot \|\hat{y}_{t,n}\|_1 \right]$$
- 总体损失函数 (Overall Loss Function)

$$L(w, \hat{y}) = L_{source}(w) + L_{target}(w, \hat{y}) + L_{reg}(\hat{y})$$
代码中并没有显式的正则化项操作，但通过“全局阈值截断”这一物理操作，在数学上等价于求解带正则项的最优解


```shell
#训练示例
python ST.py --arch resnet50 --method ST --src_path ./original_datasets/office_31/amazon --tgt_path ./original_datasets/office_31/webcam --apply_aug --num_rounds 20 --epochs_per_round 3 --init_portion 0.2 --portion_step 0.05 --max_portion 0.8 --lr 2e-4 --save_dir ./checkpoints/amazon_to_webcam_ST
```
#### CBST 
- 源域损失 (Source Domain Loss)与标准自训练一致，使用带标签源域数据的交叉熵损失:

$$L_{source} = - \sum_{s=1}^S \sum_{n=1}^N \left[ y_{s,n} \cdot \log(p_n(w; I_s)) \right]$$
- 类平衡目标域损失 (Class-Balanced Target Loss)通过为每个类别 $c$ 引入独立的参数 $k_c$，实现类间平衡:

$$L_{target\_balanced} = - \sum_{t=1}^T \sum_{n=1}^N \sum_{c=1}^C \left[ \hat{y}(c)_{t,n} \cdot \log(p(c | w; I_t)) + k_c \cdot \hat{y}(c)_{t,n} \right]$$
- 总体类平衡损失函数 (Overall Class-Balanced Loss):

$$L_{CB}(w, \hat{y}) = L_{source}(w) + L_{target\_balanced}(w, \hat{y})$$
CBST 并没有丢弃正则化，而是通过将正则化系数 $k$ “参数化”为与类别相关的 $k_c$，实现了类平衡选择。这种形式在数学表达上更直接地描述了伪标签 $\hat{y}$ 与预测概率 $p$ 之间的竞争关系


```shell
#训练示例
python python ST.py --method CBST --src_path ./original_datasets/office_31/amazon --tgt_path ./original_datasets/office_31/webcam --apply_aug --num_rounds 20 --epochs_per_round 3 --init_portion 0.2 --portion_step 0.05 --max_portion 0.8 --lr 2e-4 --save_dir ./checkpoints/amazon_to_webcam_CBST
```

#### CRST 
- 总体损失函数 (Combined Loss) CRST 将 CBST 的静态选择过程转化为了一个连续的正则化框架: 

$$L_{CRST}(w, \hat{y}) = \underbrace{L_{CB}(w, \hat{y})}_{\text{CBST 基础损失}} + \underbrace{\mathcal{R}_{MR}(w)}_{\text{模型正则化扩展}}$$
##### 可选扩展 (标签正则)
- `LRENT` (熵正则扩展) 在 CBST 中，伪标签 $\hat{y}$ 通常是硬标签（Hard Label）,CRST 可引入 LRENT 来实现标签软化:

$$\hat{y}_{t}^{(i)} = \frac{\left( \frac{p(i|\mathbf{x}_t)}{\lambda_i} \right)^{\frac{1}{\alpha}}}{\sum_{k=1}^{K} \left( \frac{p(k|\mathbf{x}_t)}{\lambda_k} \right)^{\frac{1}{\alpha}}}$$

扩展：通过 $\alpha$ 允许伪标签具有一定的概率分布，而非(1,0)
##### 可选扩展 (模型正则)
- `MRKLD` (KL 散度扩展): 

$$\mathcal{R}_{MRKLD} = -\sum_{k=1}^{K} \frac{1}{K} \log p(k|\mathbf{x}_t)$$
- `MRENT` (熵正则扩展):

$$\mathcal{R}_{MRENT} = \sum_{k=1}^{K} p(k|\mathbf{x}_t) \log p(k|\mathbf{x}_t)$$
- `MRL2` (L2 范数扩展):

$$\mathcal{R}_{MRL2} = \sum_{k=1}^{K} p(k|\mathbf{x}_t)^2$$

代码使用 alpha beta gamma delta 来控制各个正则化强度，原始论文使用了消融实验方法

$$L_{total} = L_{CE}(\mathbf{p}(\mathbf{x}_t), \hat{\mathbf{y}}_{t(\alpha)}) + \beta \mathcal{R}_{MRKLD} + \gamma \mathcal{R}_{MRENT} + \delta \mathcal{R}_{MRL2}$$

```shell
#训练示例 (运行时请修改 alpha beta gamma delta 我全上0.1只是为了检查报错)
   python ST.py --method CRST --src_path ./original_datasets/office_31/amazon --tgt_path ./original_datasets/office_31/webcam --apply_aug --num_rounds 20 --epochs_per_round 3 --init_portion 0.2 --portion_step 0.05 --max_portion 0.8 --lr 2e-4 --alpha 0.1 --beta 0.1 --gamma 0.1 --delta 0.1 --save_dir ./checkpoints/amazon_to_webcam_CRST_mixed
```

### Self training part Warning
代码没法直接用官方，因为公开的代码是做 img seg 任务，目前的代码大部分是参考官方过程手搓的，关键部分我都有参考论文并在代码中注释，但不保证正确, 请Review

目前我只测试过kc_value='conf' 'prob' 模式 未测试



