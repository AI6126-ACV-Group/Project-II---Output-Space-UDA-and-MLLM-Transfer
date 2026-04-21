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
    def __init__(self, root, transform=None):
        super(TargetUnlabeledDataset, self).__init__(root, transform)

    def __getitem__(self, index):
        """
        覆盖父类方法：
        返回: (图片张量, 原始索引, 图片绝对路径)
        """
        path, target = self.samples[index]  # target 是 ImageFolder 自动生成的 ID
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, path


class PseudoLabeledDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label


def kc_parameters(conf_dict, pred_cls_num, args, round_idx):

    print(f'\n###### Round {round_idx}: Start KC Generation (Method: {args.method}) ######')
    start_time = time.time()
    cls_thresh = np.ones(args.num_classes, dtype=np.float32)
    cls_sel_size = np.zeros(args.num_classes, dtype=np.float32)
    # 当前轮次的选取比例 (Curriculum Learning)
    portion = min(args.init_portion + round_idx * args.portion_step, args.max_portion)

    if args.method == 'ST':
        # (ST) with self-paced learning
        '''
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
        有点类似于 WCE 的思想，但在无监督任务中该weight被赋给了伪标签的阈值
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

    # (Rare Class Mining)
    cls_ratios = pred_cls_num / (np.sum(pred_cls_num) + 1e-6)
    rare_id = np.argsort(cls_ratios)[:args.rare_cls_num]
    save_path = os.path.join(args.save_dir, f'round_{round_idx}')
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, 'cls_thresh.npy'), cls_thresh)
    np.save(os.path.join(save_path, 'rare_id.npy'), rare_id)

    print(f'Rarest IDs: {rare_id} | Time: {time.time() - start_time:.2f}s')
    return cls_thresh

def generate_pseudo_data(model, loader, device, args):
    model.eval()
    conf_dict = {i: [] for i in range(args.num_classes)}
    pred_cls_num = np.zeros(args.num_classes)
    all_raw_results = []
    with torch.no_grad():
        for imgs, _, paths in loader:
            imgs = imgs.to(device)
            probs = F.softmax(model(imgs), dim=1)
            # 获取 Top-1 预测
            max_probs, preds = torch.max(probs, dim=1)

            for i in range(len(preds)):
                p_label = preds[i].item()
                p_score = max_probs[i].item()
                pred_cls_num[p_label] += 1
                # 根据 kc_value 决定统计池的内容
                if args.kc_value == 'conf':
                    #硬标签，此时conf_dict返回的是 top-1
                    """
                    {
                     0: [0.92, 0.81, 0.75, ...], # 只有被预测为类别 0 的 Top-1 分数
                     1: [0.99, 0.88, ...],       # 只有被预测为类别 1 的 Top-1 分数
                      ...
                    }
                    """
                    conf_dict[p_label].append(p_score)
                elif args.kc_value == 'prob':
                    # 软标签
                    """
                    {
                    0: [0.92, 0.01, 0.05, ...], # 既包含预测为0的高分，也包含预测为其他类时，分类器给类0的低分
                    1: [0.03, 0.99, 0.02, ...], 
                    ...
                    }
                    """
                    for c in range(args.num_classes):
                        conf_dict[c].append(probs[i, c].item())
                all_raw_results.append({'path': paths[i], 'label': p_label, 'score': p_score})

    return all_raw_results, conf_dict, pred_cls_num


def loss_function(logits, labels, args):
    """
    - args.alpha: LRENT (Label Regularization) 权重
    - args.beta:  MRKLD (Model Regularization - KL) 权重
    - args.gamma: MRENT (Model Regularization - Entropy) 权重
    - args.delta: MRL2 (Model Regularization - L2) 权重
    """
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    num_classes = logits.size(1)

    # 1. 基础交叉熵损失 (Standard Cross Entropy)
    ce_loss = F.cross_entropy(logits, labels)

    if args.method != 'CRST':
        return  ce_loss

    # --- 正则化项初始化 ---
    lrent_loss = 0.0
    mrkld_loss = 0.0
    mrent_loss = 0.0
    mrl2_loss = 0.0

    # 2. LRENT (Label Regularization via Entropy)
    # LRENT (Label Regularization via Entropy):
    # sum_{k=1}^K [ y_hat_k * log(y_hat_k) ]
    # 物理意义: 惩罚伪标签的确定性。由于代码中 labels 通常是 one-hot 或平滑分布，
    # 训练时通过最小化预测分布的负熵（即最大化熵）来缓解过拟合，防止模型过快收敛到错误的硬标签。
    if args.alpha > 0:
        lrent_loss = -(probs * torch.log(probs + 1e-6)).sum(dim=1).mean()

    # MRKLD (Model Regularization via KL Divergence):
    # - sum_{k=1}^K [ (1 / K) * log(p(k | x_t)) ]
    # 物理意义: 最小化预测分布 p 与均匀分布 U(1/K) 之间的 KL 散度。
    # 效果: 强制模型预测向均匀分布靠拢，这是 CRST 论文中最推荐的正则化方式，能有效保持类别多样性。
    if args.beta > 0:
        mrkld_loss = -log_probs.mean(dim=1).mean()

    # MRENT (Model Regularization via Entropy):
    # sum_{k=1}^K [ p(k | x_t) * log(p(k | x_t)) ]
    # 物理意义: 惩罚模型预测分布的负熵。
    # 效果: 直接鼓励模型输出具有更高熵（更不确定）的预测，避免模型陷入单一类别的自信陷阱。
    if args.gamma > 0:
        mrent_loss = (probs * log_probs).sum(dim=1).mean()  # 惩罚低熵

    # MRL2 (Model Regularization via L2 Norm):
    #  sum_{k=1}^K [ p(k | x_t)^2 ]
    # 物理意义: 最小化预测概率向量的 L2 范数。
    # 效果: 防止预测分布中出现极大的概率值（如 0.999），迫使概率分布更加平滑。
    if args.delta > 0:
        mrl2_loss = torch.norm(probs, p=2, dim=1).mean()

    # --- 最终损失加权整合 ---
    total_loss = (ce_loss +
                  args.alpha * lrent_loss +
                  args.beta * mrkld_loss +
                  args.gamma * mrent_loss +
                  args.delta * mrl2_loss)

    return total_loss


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
        t_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer_wm.zero_grad()
            outputs = model(imgs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer_wm.step()
            t_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * correct / total
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for imgs, labels in src_val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                v_loss += F.cross_entropy(outputs, labels).item()

        avg_val_loss = v_loss / len(src_val_loader)
        current_lr = optimizer_wm.param_groups[0]['lr']
        scheduler_wm.step()
        print(
            f"Epoch [{epoch + 1}] | LR: {current_lr:.6f} | Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f}")
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
    args.num_classes = len(full_src_ds.classes)
    model.fc = nn.Linear(num_ftrs, args.num_classes)
    model.to(device)

    # 目标域数据集
    # 用于真实准确率评估 (使用标准 ImageFolder)
    tgt_eval_ds = datasets.ImageFolder(args.tgt_path, transform=transform_eval)
    tgt_eval_loader = DataLoader(tgt_eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 用于自训练生成伪标签 (使用自定义类，获取路径)
    tgt_raw_ds = TargetUnlabeledDataset(args.tgt_path, transform=transform_eval)
    tgt_loader = DataLoader(tgt_raw_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    assert full_src_ds.classes == tgt_eval_ds.classes, "Domain classes mismatch!"

    start_round = 0
    best_target_acc = 0.0
    st_patience = 3
    st_counter = 0

    if args.resume and os.path.exists(checkpoint_path):
        print(f"==> Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_round = checkpoint['round'] + 1
        best_target_acc = checkpoint.get('best_target_acc', 0.0)
        st_counter = checkpoint.get('st_counter', 0)
        print(f"==> Resumed from Round {start_round}. Best Acc: {best_target_acc:.2f}%")

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
    for r in range(start_round, args.num_rounds):
        # 生成伪标签 (使用 tgt_loader 获取路径)
        all_raw, conf_dict, pred_cls_num = generate_pseudo_data(model, tgt_loader, device, args)
        # 更新类阈值 (KC)
        current_thresholds = kc_parameters(conf_dict, pred_cls_num, args, r)
        # 筛选样本
        selected_samples = [(s['path'], s['label']) for s in all_raw if s['score'] >= current_thresholds[s['label']]]
        print(f"Round {r}: Selected {len(selected_samples)} target samples.")
        # 构建混合数据集进行再训练
        tgt_pseudo_ds = PseudoLabeledDataset(selected_samples, transform=transform_train)
        combined_loader = DataLoader(
            torch.utils.data.ConcatDataset([full_src_ds, tgt_pseudo_ds]),
            batch_size=args.batch_size, shuffle=True, num_workers=4
        )
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs_per_round * len(combined_loader), eta_min=1e-6
        )

        model.train()

        for epoch in range(args.epochs_per_round):
            total_loss = 0
            for imgs, labels in combined_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = loss_function(model(imgs), labels, args)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            print(
                f"Round {r} | Epoch {epoch} | LR: {optimizer.param_groups[0]['lr']:.6f} | Loss: {total_loss / len(combined_loader):.4f}")

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
    parser.add_argument('--method', type=str, default='ST', choices=['ST', 'CBST', 'CRST'])
    parser.add_argument('--kc_value', type=str, default='conf', choices=['conf', 'prob'],help="kc_value 计算使用硬标签还是软标签")
    ## CRST 正则方法设置 (如果训练模式是 ST 或 CBST 下面的参数将不会产生任何效果)
    parser.add_argument('--alpha', type=float, default=0.0, help="LRENT (Label Regularization) 权重")
    parser.add_argument('--beta', type=float, default=0.0, help="MRKLD (Model Regularization - KL) 权重")
    parser.add_argument('--gamma', type=float, default=0.0, help="MRENT (Model Regularization - Entropy) 权重")
    parser.add_argument('--delta', type=float, default=0.0, help="MRL2 (Model Regularization - L2) 权重")
    # 数据与保存路径
    parser.add_argument('--src_path', type=str, default='./original_datasets/office_31/amazon',  help='源域数据路径')
    parser.add_argument('--tgt_path', type=str, default='./original_datasets/office_31/webcam',  help='目标域无标签数据路径')
    parser.add_argument('--apply_aug', action='store_true', help='是否在训练时应用 RandomCrop 和 Flip')
    parser.add_argument('--save_dir', type=str, default='./ST_test')
    # ST超参数
    parser.add_argument('--num_classes', type=int, default=31)
    parser.add_argument('--init_portion', type=float, default=0.2, help='初始选择比例')
    parser.add_argument('--portion_step', type=float, default=0.1, help='每轮增加比例')
    parser.add_argument('--max_portion', type=float, default=0.9)
    parser.add_argument('--reg_weight', type=float, default=0.1, help='CRST正则项权重')
    parser.add_argument('--rare_cls_num', type=int, default=3)
    # 训练配置
    parser.add_argument('--num_rounds', type=int, default=5)
    parser.add_argument('--epochs_per_round', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)

    args = parser.parse_args()

    main(args)

    #python ST.py --method CBST --src_path ./original_datasets/office_31/amazon --tgt_path ./original_datasets/office_31/webcam --apply_aug --num_rounds 20 --epochs_per_round 5 --init_portion 0.2 --portion_step 0.05 --max_portion 0.8 --lr 5e-4 --save_dir ./checkpoints/amazon_to_webcam_CBST