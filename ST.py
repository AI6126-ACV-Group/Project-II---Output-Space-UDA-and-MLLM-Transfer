import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn.functional as F
import os
import math
import time
import sys
from torch.utils.data import random_split
from torch.utils.data import Subset

class Logger(object):
    """
    Logger class to log training progress, the result will be saved as
    train_model_time.log file in --save_dir
    """
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 确保实时写入磁盘

    def flush(self):
        pass

class TargetUnlabeledDataset(datasets.ImageFolder):
    """
    读取图片并生成dataset：
    返回: (image_tensor, target_index, image_path)
    """
    def __init__(self, root, transform=None):
        super(TargetUnlabeledDataset, self).__init__(root, transform)

    def __getitem__(self, index):
        path, target = self.samples[index]  # target 是 ImageFolder 自动生成的 ID
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, path


class PseudoLabeledDataset(Dataset):
    """
    伪标签数据生成过程
    """
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        # 必须指定 loader
        self.loader = datasets.folder.default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        try:
            img = self.loader(path)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # 返回一个黑图或者抛出错误
            raise e
        if self.transform:
            img = self.transform(img)
        # 确保 label 是 Tensor
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label).long()
        return img, label

class SourceDatasetWrapper(Dataset):
    """
    非标转标过程，给int转统一的tensor
    """
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        img, label = self.dataset[i]
        # 将 int 转换为 one-hot 向量，形状为 [num_classes]
        one_hot_label = torch.zeros(self.num_classes)
        one_hot_label[label] = 1.0
        return img, one_hot_label

def kc_parameters(conf_dict, pred_cls_num, args, round_idx):
    """
    生成 ST/CB(R)ST 方法下目标域数据允许进入混合训练集的的置信度阈值
    :param conf_dict: 置信度字典
    :param pred_cls_num: 目标标签数量
    :param args: 传入 argparse
    :param round_idx: 迭代训练 index
    :return: cls_thresh (numpy 数组，长度与分类任务label数量一致，用来保存进入训练集的置信度阈值)
    """

    print(f'\n###### Round {round_idx}: Start KC Generation (Method: {args.method}) ######')
    start_time = time.time()
    # 初始化numpy数组
    cls_thresh = np.ones(args.num_classes, dtype=np.float32)
    cls_sel_size = np.zeros(args.num_classes, dtype=np.float32)

    # 当前轮次的选取比例 (Curriculum Learning) 用 args.max_portion 控制目标域数据进入混合训练集的最大上限
    portion = min(args.init_portion + round_idx * args.portion_step, args.max_portion)

    if args.method == 'ST':
        # (ST) with self-paced learning
        '''CBST 论文原文：
           A better strategy is to follow an easy-to-hard
           scheme via self-paced curriculum learning, where one seeks to generate pseudo-
           labels from the most confident predictions and hope they are mostly correct.
           Once the model is updated and better adapted to the target domain, the scheme
           then explores the remaining pseudo-labels with less confidence.
           
           L(w, y_hat) = L_source(w)  + L_target(w, y_hat) + L_reg(y_hat)
           L_source = - sum_{s=1}^S sum_{n=1}^N [ y_{s,n} * log(p_n(w; I_s))]     源域损失 (交叉熵)
           L_target = - sum_{t=1}^T sum_{n=1}^N [ y_hat_{t,n} * log(p_n(w; I_t))] 目标域损失,使用生成的伪标签 y_hat 进行自监督 (交叉熵)
           L_reg = - sum_{t=1}^T sum_{n=1}^N [ k * ||y_hat_{t,n}||_1 ]            正则化项, k是超参数，控制伪标签的选择量。k 越大，选中的样本越多 
           k 由代码 portion 隐式控制，  p_n 对应代码 probs
        '''
        all_scores = []
        for c in range(args.num_classes):
            all_scores.extend(conf_dict[c])
        if len(all_scores) > 0:
            all_scores.sort(reverse=True)
            total_sel = int(math.floor(len(all_scores) * portion))
            global_t = all_scores[total_sel - 1] if total_sel > 0 else 1.0
            cls_thresh[:] = global_t  # 所有类公用一个阈值，该值为当前portion下的最低score, self-paced learning
            print(f"ST Global Threshold: {global_t:.4f}")

    else:
        # CBST / CRST 逻辑：类平衡阈值
        """
        L_CB(w, y_hat) = L_source(w) + L_target_balanced(w, y_hat)
        L_source = - sum_{s=1}^S sum_{n=1}^N [ y_{s,n} * log(p_n(w; I_s)) ] 源域损失 (与 ST 一致):
        L_target_balanced = - sum_{t=1}^T sum_{n=1}^N sum_{c=1}^C [y_hat(c)_{t,n} * log(p(c | w; I_t)) + k_c * y_hat(c)_{t,n}] 类平衡目标域损失 
        k_c : 每一类独立的正则化参数 它决定了类 c 中被选为伪标签的比例 (相当于每一类都有自己的 portion) k_c > 0 对于所有类别 c 成立以保证所有类别都得到训练
        y_hat_{t,n} 属于 {e_1, ..., e_C} (one-hot 向量) 或 {0} (不选)。
        
        k_c 由代码 cls_sel_size[idx_cls] = int(math.floor(len(scores) * portion)) 控制 
        有点类似于 WCE 的思想，但在无监督任务中该weight用于目标域数据进入混合训练集的阈值
        """
        for idx_cls in range(args.num_classes):
            scores = conf_dict[idx_cls]
            if scores and len(scores) > 0:
                scores.sort(reverse=True)
                cls_sel_size[idx_cls] = int(math.floor(len(scores) * portion))
                len_sel = int(cls_sel_size[idx_cls])
                if len_sel > 0:
                    cls_thresh[idx_cls] = scores[len_sel - 1]

        print(f"CBST/CRST Thresholds per class: {np.round(cls_thresh, 4)}")

    # (Rare Class Mining) 这一部分继承于官方代码的 crst_seg_offical.py
    # 目前可以显示 top args.rare_cls_num 数量的置信度最低的label 但目前没有设置调整权重的代码，如后续训练出现hard sample, 可以添加调整rare_id cls_thresh 的代码
    cls_ratios = pred_cls_num / (np.sum(pred_cls_num) + 1e-6)
    rare_id = np.argsort(cls_ratios)[:args.rare_cls_num]
    # TODO: 当出现明显的hard sample时 （Thresholds 过低） 添加对其进行cls_thresh调整代码

    # 权重文件保存
    save_path = os.path.join(args.save_dir, f'round_{round_idx}')
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, 'cls_thresh.npy'), cls_thresh)
    np.save(os.path.join(save_path, 'rare_id.npy'), rare_id)
    print(f'Rarest IDs: {rare_id} | Time: {time.time() - start_time:.2f}s')
    return cls_thresh


def get_model_predictions(model, loader, device, args):
    model.eval()
    all_probs = []
    all_paths = []
    conf_dict = {i: [] for i in range(args.num_classes)}
    pred_cls_num = np.zeros(args.num_classes)
    with torch.no_grad():
        for imgs, _, paths in loader: #这里目标域的target被隐藏，不会被使用
            imgs = imgs.to(device)
            probs = F.softmax(model(imgs), dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            for i in range(len(preds)):
                p_label = preds[i].item()
                p_score = max_probs[i].item()
                pred_cls_num[p_label] += 1
                if args.kc_value == 'conf':
                    """
                    {
                    0: [0.9, 0.8...],      # 只有被认作是类0的图，其Top-1分数才在这, 可能小于真实的0类别图数量
                    1: [0.95...],          # 只有被认作是类1的图...
                    2: [0.7...],           # 只有被认作是类2的图...
                    ...
                    } 字典中所有概率的数量 = 训练集的图片数   
                    优点: 计算速度快，在类别平衡的数据集上效果会更好，判别性强
                    缺点: 如果模型对类 A 有偏见（预测数量极少），类 A 的统计池样本量会非常小。由于样本不足，计算出的阈值（Top k%）可能由于极端值的干扰而变得无意义
                        可能“死锁”,如果某一轮模型完全没有把任何样本预测为类 B，那么类 B 的池子就是空的。在 CBST 中，这会导致该类彻底失去被选中的机会
                    """
                    conf_dict[p_label].append(p_score)
                else:
                    """
                    {
                    0: [0.1, 0.23...],      # 所有训练图片被识别为1的概率
                    1: [0.2, 0.75...],      # 所有训练图片被识别为2的概率
                    2: [0.7, 0.05...],      # 所有训练图片被识别为3的概率
                    } 字典中所有概率的数量 = 训练集的图片数 * 类别数   即使类 A 预测得少，阈值也会相对平滑，
                    优点: 无论模型预测如何偏向，每个类都有充足的数据来计算阈值，不会出现“死锁”，对非平衡数据的稀缺类友好，阈值更平滑
                    缺点: 如果设置的 portion 较大，选出的阈值可能非常低，从而引入大量极其模糊、甚至完全错误的伪标签，让训练彻底崩溃
                         它的阈值不再代表“我认为它是类 A 的确信度”，而是代表“它在所有图里的类 A 激活排名”，判别性弱
                         计算压力大，速度慢
                    """
                    for c in range(args.num_classes):
                        conf_dict[c].append(probs[i, c].item())

            all_probs.append(probs.cpu())
            all_paths.extend(paths)

    return torch.cat(all_probs), all_paths, conf_dict, pred_cls_num


def select_sample(all_probs, all_paths, current_thresholds, args):
    """
    生成目标域被选出的 图片 和 假标签 (all_paths[i], target_label)
    """
    selected_samples = []
    num_classes = all_probs.size(1)
    lambdas = torch.from_numpy(current_thresholds).float()
    for i in range(len(all_probs)):
        prob = all_probs[i]
        max_score, hard_label = torch.max(prob, dim=0)
        # 1. 筛选逻辑：只保留超过当前类阈值的样本
        if max_score >= current_thresholds[hard_label.item()]:
            if args.method == 'CRST' and args.alpha > 0:
                # LRENT label: Tensor [num_classes]
                # 对应论文LRENT: Y_t_hat = [(p_i / lambda) ^ (1 / alpha)] / sum((p_k / lambda) ^ (1 / alpha))
                # 1e-6 是为了防止除以 0
                soft_label = (prob / (lambdas + 1e-6)) ** (1.0 / args.alpha)
                target_label = soft_label / soft_label.sum()
            else:
                # Hard label: 包装成 Tensor 标量，确保 dataset 拿到的是一致的类型
                target_label = torch.zeros(num_classes)
                target_label[hard_label.item()] = 1.0
            selected_samples.append((all_paths[i], target_label))

    return selected_samples


def loss_function(logits, labels, args):
    """
    损失函数设置 ST/CBST 使用交叉熵 ; CRST 会额外引入正则项 (论文中的 LRENT,MRKLD,MRENT,MRL2)
    - args.beta:  MRKLD (Model Regularization - KL) 权重
    - args.gamma: MRENT (Model Regularization - Entropy) 权重
    - args.delta: MRL2 (Model Regularization - L2) 权重
    """
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)

    # 基础交叉熵损失 (已经是 Tensor)
    ce_loss = -(labels * log_probs).sum(dim=1).mean()

    # 初始化正则项为 0 维 Tensor (这样它们依然是 Tensor 模式)
    # 使用 .to(logits.device) 确保它们在同一张显卡上
    mrkld_loss = torch.tensor(0.0).to(logits.device)
    mrent_loss = torch.tensor(0.0).to(logits.device)
    mrl2_loss = torch.tensor(0.0).to(logits.device)

    # 只有 CRST 模式才计算具体的正则项
    if args.method == 'CRST':
        if args.beta > 0:
            # MRKLD (Model Regularization via KL Divergence):
            # - sum_{k=1}^K [ (1 / K) * log(p(k | x_t)) ] (其实就是负的对数概率均值)
            # 物理意义: 最小化预测分布 p 与均匀分布 U(1/K) 之间的 KL 散度。
            # 效果: 强制模型预测向均匀分布靠拢，这是 CRST 论文中最推荐的正则化方式，能有效保持类别多样性。
            mrkld_loss = -log_probs.mean(dim=1).mean()
        if args.gamma > 0:
            # MRENT (Model Regularization via Entropy):
            # sum_{k=1}^K [ p(k | x_t) * log(p(k | x_t)) ]
            # 物理意义: 惩罚模型预测分布的负熵。
            # 效果: 直接鼓励模型输出具有更高熵（更不确定）的预测，避免模型陷入单一类别的自信陷阱。
            mrent_loss = (probs * log_probs).sum(dim=1).mean()
        if args.delta > 0:
            # MRL2 (Model Regularization via L2 Norm):
            #  sum_{k=1}^K [ p(k | x_t)^2 ]
            # 物理意义: 最小化预测概率向量的 L2 范数。
            # 效果: 防止预测分布中出现极大的概率值（如 0.999），迫使概率分布更加平滑。
            # 这里的 .mean() 会保持 Tensor 属性
            mrl2_loss = torch.norm(probs, p=2, dim=1).mean()

    # 2. 最终损失加权整合 (所有项都是 Tensor，相加结果也是 Tensor)
    total_loss = (ce_loss +
                  args.beta * mrkld_loss +
                  args.gamma * mrent_loss +
                  args.delta * mrl2_loss)

    # 始终返回 5 个值，保持调用处解包的一致性
    return total_loss, ce_loss, mrkld_loss, mrent_loss, mrl2_loss


def source_warmup(model, train_loader, src_val_loader, tgt_val_loader, device, args, warmup_model_path):
    print(f"==> Starting Warm-up...")
    patience = args.patience if hasattr(args, 'patience') else 5
    best_val_loss = float('inf')  # 修改为监控验证集 Loss
    counter = 0
    best_model_wts = None
    optimizer_wm = optim.AdamW(model.parameters(), lr=args.lr_warm, weight_decay=1e-2)
    scheduler_wm = optim.lr_scheduler.CosineAnnealingLR(optimizer_wm, T_max=args.warmup_epochs, eta_min=1e-6)
    for epoch in range(args.warmup_epochs):
        # --- 训练阶段 ---
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0  # 重命名变量以区分
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer_wm.zero_grad()
            outputs = model(imgs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer_wm.step()
            t_loss += loss.item()
            t_correct += (outputs.argmax(1) == labels).sum().item()
            t_total += labels.size(0)

        avg_train_loss = t_loss / len(train_loader)
        train_acc = 100. * t_correct / t_total
        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in src_val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                v_loss += F.cross_entropy(outputs, labels).item()
                v_correct += (outputs.argmax(1) == labels).sum().item()
                v_total += labels.size(0)

        avg_val_loss = v_loss / len(src_val_loader)
        val_acc = 100. * v_correct / v_total

        current_lr = optimizer_wm.param_groups[0]['lr']
        scheduler_wm.step()

        print(f"Epoch [{epoch + 1}/{args.warmup_epochs}] | LR: {current_lr:.6f} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        # 早停逻辑
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            best_model_wts = model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                print(f"==> Early stopping triggered at epoch {epoch + 1}")
                break

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    model.eval()
    correct_tgt = 0
    ## Source-only eval
    with torch.no_grad():
        for imgs, labels in tgt_val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct_tgt += (preds == labels).sum().item()
    print(f"\n==> Warm-up Complete! Initial Target Acc: {100. * correct_tgt / len(tgt_val_loader.dataset):.2f}%")
    torch.save(model.state_dict(), warmup_model_path)

    return model

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    log_name = f"train_{args.method}_{args.arch}_{time.strftime('%Y%m%d-%H%M%S')}.log"
    log_path = os.path.join(args.save_dir, log_name)
    sys.stdout = Logger(log_path, sys.stdout)
    sys.stderr = Logger(log_path, sys.stderr)

    print(f"========== Experiment Configuration ==========")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print(f"==============================================\n")

    checkpoint_path = os.path.join(args.save_dir, 'checkpoint.pth')
    warmup_model_path = os.path.join(args.save_dir, f'source_only_{args.arch}.pth')

    print(f"==> Initializing architecture: {args.arch}")
    model_func = getattr(models, args.arch)
    model = model_func(pretrained=True)

    # 第一次运行需要从 src_ds 获取 classes，先临时初始化
    # 如果是续训，num_classes 会被 args 覆盖
    num_ftrs = model.fc.in_features

    norm = transforms.Normalize(mean=args.norm_mean, std=args.norm_std)
    transform_eval = transforms.Compose([
        transforms.Resize((args.load_size, args.load_size)),
        transforms.CenterCrop((args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        norm
    ])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.crop_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm
    ]) if args.apply_aug else transform_eval

    full_src_ds = datasets.ImageFolder(args.src_path, transform=transform_train)
    full_src_ds = SourceDatasetWrapper(full_src_ds,args.num_classes) #转 tensor

    #args.num_classes = len(full_src_ds.dataset.classes)
    model.fc = nn.Linear(num_ftrs, args.num_classes)
    model.to(device)

    # 目标域数据集
    # 用于真实准确率评估 (使用标准 ImageFolder)
    tgt_eval_ds = datasets.ImageFolder(args.tgt_path, transform=transform_eval)
    tgt_eval_loader = DataLoader(tgt_eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # 用于自训练生成伪标签 (使用自定义类，获取路径)
    tgt_raw_ds = TargetUnlabeledDataset(args.tgt_path, transform=transform_eval)
    tgt_loader = DataLoader(tgt_raw_ds, batch_size=args.batch_size, shuffle=False, num_workers=1)

    assert full_src_ds.dataset.classes == tgt_eval_ds.classes, "Domain classes mismatch!"

    start_round = 0
    best_target_acc = 0.0
    st_patience = 8
    st_counter = 0

    # --- Source-only Warm-up ---
    if not (args.resume and os.path.exists(checkpoint_path)):
        if os.path.exists(warmup_model_path):
            print(f"==> Loading pre-trained source model: {warmup_model_path}")
            model.load_state_dict(torch.load(warmup_model_path))
        else:
            base_ds = datasets.ImageFolder(args.src_path)
            indices = np.arange(len(base_ds))
            np.random.shuffle(indices)
            train_size = int(0.8 * len(base_ds))
            train_idx, val_idx = indices[:train_size], indices[train_size:]
            # 创建两个独立的数据集实例，分别应用不同的 transform
            src_train_ds = datasets.ImageFolder(args.src_path, transform=transform_train)
            src_val_ds = datasets.ImageFolder(args.src_path, transform=transform_eval)
            # 使用 Subset 根据索引提取对应部分，防止数据泄露
            src_train_ds = Subset(src_train_ds, train_idx)
            src_val_ds = Subset(src_val_ds, val_idx)
            src_train_loader = DataLoader(src_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=1)
            src_val_loader = DataLoader(src_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=1)
            overlap = set(train_idx).intersection(set(val_idx))
            # 检查是否存在数据泄露问题
            print(f"==> Data Split Check: Train={len(train_idx)}, Val={len(val_idx)}, Overlap={len(overlap)}")
            assert len(overlap) == 0, "ERROR: Data Leakage detected in indices!"
            model = source_warmup(model, src_train_loader, src_val_loader, tgt_eval_loader, device, args, warmup_model_path)

    # --- Self-Training Rounds ---
    current_thresholds = np.zeros(args.num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_per_round * args.num_rounds, eta_min=0
    )

    if args.resume and os.path.exists(checkpoint_path):
        print(f"==> Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        # 恢复模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        # 恢复优化器状态（包含 momentum, velocity 等）
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 恢复调度器状态（包含当前步数 last_epoch）
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_round = checkpoint['round'] + 1
        best_target_acc = checkpoint.get('best_target_acc', 0.0)
        st_counter = checkpoint.get('st_counter', 0)
        print(f"==> Resumed from Round {start_round}. Best Acc: {best_target_acc:.2f}%")

    for r in range(start_round, args.num_rounds):
        
        raw_probs, paths, conf_dict, pred_num = get_model_predictions(model, tgt_loader, device, args)
        # 2. 算当前轮次的新阈值
        current_thresholds = kc_parameters(conf_dict, pred_num, args, r)
        # 这里不需要 GPU，速度极快
        selected_samples = select_sample(raw_probs, paths, current_thresholds, args)

        print(f"Round {r}: Selected {len(selected_samples)} target samples.")
        # 构建混合数据集进行再训练
        tgt_pseudo_ds = PseudoLabeledDataset(selected_samples, transform=transform_train)

        combined_loader = DataLoader(
            torch.utils.data.ConcatDataset([full_src_ds, tgt_pseudo_ds]),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=1,
            #collate_fn=lambda x: (torch.stack([item[0] for item in x]), # 我就不信转不成tensor了
            #                      torch.stack([item[1] if isinstance(item[1], torch.Tensor) else torch.tensor(item[1]).long() for item in x]))
        )

        model.train()

        for epoch in range(args.epochs_per_round):
            total_loss = 0
            total_ce_loss = 0
            total_mrkld_loss=0
            total_mrent_loss = 0
            total_mrl2_loss=0
            for imgs, labels in combined_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss, ce_loss, mrkld_loss, mrent_loss, mrl2_loss = loss_function(model(imgs), labels, args)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_mrkld_loss += mrkld_loss.item()
                total_mrent_loss += mrent_loss.item()
                total_mrl2_loss += mrl2_loss.item()

            scheduler.step()
            print(
                f"Round {r} | Epoch {epoch} | LR: {optimizer.param_groups[0]['lr']:.6f} "
                f"| Loss: {total_loss / len(combined_loader):.4f}"
                f"| CE loss: {total_ce_loss / len(combined_loader):.4f} "
                f"| mrkld loss: {total_mrkld_loss / len(combined_loader):.4f} "
                f"| mrent loss: {total_mrent_loss / len(combined_loader):.4f} "
                f"| mrl2 loss: {total_mrl2_loss / len(combined_loader):.4f}")

        # 验证 (使用带有真标的 tgt_eval_loader)
        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in tgt_eval_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()

        current_acc = 100 * correct / len(tgt_eval_ds)

        # 更新早停状态
        if current_acc > best_target_acc:
            best_target_acc = current_acc
            st_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_target_model.pth'))
            print(f"*** New Best Target Acc: {best_target_acc:.2f}%! ***")
        else:
            st_counter += 1

        print(f"\n>>> [Round {r} Summary] Acc: {current_acc:.2f}% | Best: {best_target_acc:.2f}%")
        print("-" * 50)

        # 保存 Checkpoint
        save_dict = {
            'round': r,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),  # 必须保存
            'scheduler_state_dict': scheduler.state_dict(),  # 必须保存
            'best_target_acc': best_target_acc,
            'st_counter': st_counter,
            'args': args
        }
        torch.save(save_dict, checkpoint_path)
        torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_round_{r}.pth'))

        if st_counter >= st_patience:
            print(f"==> Early stopping self-training at Round {r}.")
            break


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ResNet UDA with ST/CBST/CRST')
    # 模型图像加载设置
    parser.add_argument('--load_size', type=int, default=256, help='图像缩放后的基础尺寸')
    parser.add_argument('--crop_size', type=int, default=224, help='图像最终裁剪出的尺寸')
    parser.add_argument('--norm_mean', type=float, nargs=3, default=[0.485, 0.456, 0.406],help='归一化均值 (R, G, B)')
    parser.add_argument('--norm_std', type=float, nargs=3, default=[0.229, 0.224, 0.225],help='归一化标准差 (R, G, B)')
    # 源域模型设置
    parser.add_argument('--arch', type=str, default='resnet50',choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'], help='选择网络baseline')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='源域预训练轮数，如果没有源域模型输入，会重新训练')
    parser.add_argument('--lr_warm', type=float, default=0.0001, help='源域预训练初始学习率(cosine annealing)，如果没有源域模型输入，会重新训练')
    # 是否从断点继续训练
    parser.add_argument('--resume', action='store_true', help='是否从 checkpoint 恢复训练, 默认从文件夹中checkpoint.pth恢复训练')
    # ST模式设置
    parser.add_argument('--method', type=str, default='ST', choices=['ST', 'CBST', 'CRST'], help="自训练模式")
    parser.add_argument('--kc_value', type=str, default='conf', choices=['conf', 'prob'],help="kc_value 计算kc时使用top-1(硬) 还是概率分布(软)")
    ## CRST 正则方法设置 (如果训练模式是 ST 或 CBST 下面的参数将不会产生任何效果)
    parser.add_argument('--alpha', type=float, default=0.0, help="LRENT (Label Regularization) 权重")
    parser.add_argument('--beta', type=float, default=0.0, help="MRKLD (Model Regularization - KL) 权重")
    parser.add_argument('--gamma', type=float, default=0.0, help="MRENT (Model Regularization - Entropy) 权重")
    parser.add_argument('--delta', type=float, default=0.0, help="MRL2 (Model Regularization - L2) 权重")
    # 数据与保存路径
    parser.add_argument('--src_path', type=str, default='./original_datasets/office_31/amazon',  help='源域数据路径')
    parser.add_argument('--tgt_path', type=str, default='./original_datasets/office_31/webcam',  help='目标域数据路径')
    parser.add_argument('--apply_aug', action='store_true', help='是否在训练时应用 RandomCrop 和 Flip')
    parser.add_argument('--save_dir', type=str, default='./ST_test', help='log,npy,模型checkpoint输出位置')
    # ST超参数
    parser.add_argument('--num_classes', type=int, default=31, help='分类任务标签数量')
    parser.add_argument('--init_portion', type=float, default=0.2, help='初始选择比例')
    parser.add_argument('--portion_step', type=float, default=0.1, help='每轮增加比例')
    parser.add_argument('--max_portion', type=float, default=0.9, help='自训练允许的目标域数据最大比率')
    parser.add_argument('--rare_cls_num', type=int, default=3, help='统计top n 模型最不自信的类别')
    # 训练配置
    parser.add_argument('--num_rounds', type=int, default=5, help='自训练目标最大轮数（训练时使用了早停设计）')
    parser.add_argument('--epochs_per_round', type=int, default=5, help='每次自训练跑多少epoch')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率（训练使用cosine annealing）')

    args = parser.parse_args()

    main(args)

    # python ST.py --arch resnet50 --method ST --src_path ./original_datasets/office_31/amazon --tgt_path ./original_datasets/office_31/webcam --apply_aug --num_rounds 50 --epochs_per_round 2 --init_portion 0.1 --portion_step 0.02 --max_portion 0.8 --lr 1e-5 --save_dir ./checkpoints/amazon_to_webcam_ST
    # python ST.py --arch resnet50 --method CBST --src_path ./original_datasets/office_31/amazon --tgt_path ./original_datasets/office_31/webcam --apply_aug --num_rounds 50 --epochs_per_round 2 --init_portion 0.1 --portion_step 0.02 --max_portion 0.8 --lr 1e-5 --save_dir ./checkpoints/amazon_to_webcam_CBST
    # python ST.py --arch resnet50 --method CRST --src_path ./original_datasets/office_31/amazon --tgt_path ./original_datasets/office_31/webcam --apply_aug --num_rounds 50 --epochs_per_round 2 --init_portion 0.1 --portion_step 0.02 --max_portion 0.8 --lr 1e-5 --alpha 0.05 --beta 0 --gamma 0 --delta 0 --save_dir ./checkpoints/amazon_to_webcam_CRST_LRENT
    # python ST.py --arch resnet50 --method CRST --src_path ./original_datasets/office_31/amazon --tgt_path ./original_datasets/office_31/webcam --apply_aug --num_rounds 50 --epochs_per_round 2 --init_portion 0.1 --portion_step 0.02 --max_portion 0.8 --lr 1e-5 --alpha 0 --beta 1e-3 --gamma 0 --delta 0 --save_dir ./checkpoints/amazon_to_webcam_CRST_MRKLD
    # python ST.py --arch resnet50 --method CRST --src_path ./original_datasets/office_31/amazon --tgt_path ./original_datasets/office_31/webcam --apply_aug --num_rounds 50 --epochs_per_round 2 --init_portion 0.1 --portion_step 0.02 --max_portion 0.8 --lr 1e-5 --alpha 0 --beta 0 --gamma 1e-3 --delta 0 --save_dir ./checkpoints/amazon_to_webcam_CRST_MRENT
    # python ST.py --arch resnet50 --method CRST --src_path ./original_datasets/office_31/amazon --tgt_path ./original_datasets/office_31/webcam --apply_aug --num_rounds 50 --epochs_per_round 2 --init_portion 0.1 --portion_step 0.02 --max_portion 0.8 --lr 1e-5 --alpha 0 --beta 0 --gamma 0 --delta 1e-3  --save_dir ./checkpoints/amazon_to_webcam_CRST_MRL2
    # python ST.py --arch resnet50 --method CRST --src_path ./original_datasets/office_31/amazon --tgt_path ./original_datasets/office_31/webcam --apply_aug --num_rounds 50 --epochs_per_round 2 --init_portion 0.1 --portion_step 0.02 --max_portion 0.8 --lr 1e-5 --alpha 0.05 --beta 1e-3  --gamma 0 --delta 0  --save_dir ./checkpoints/amazon_to_webcam_CRST_LRENT_MRKLD