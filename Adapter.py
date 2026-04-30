## code base: https://github.com/gaopengcuhk/CLIP-Adapter/blob/main
## code base: https://github.com/gaopengcuhk/Tip-Adapter/blob/main
## 该版本去除了 dassl 依赖
import random
from torch.utils.data import Subset
import logging
import os
from datetime import datetime
import json
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch.nn.functional as F

_tokenizer = _Tokenizer()

# 扩展模板库，你可以根据需要继续添加
CUSTOM_TEMPLATES = {
    'Office31': 'a photo of a {}',
    'Officehome': 'a photo of a {}',
    'PACS': 'an image of a {}.'
}


class TipAdapterF(nn.Module):
    def __init__(self, cache_keys, cache_values, clip_model, text_features, args):
        super().__init__()
        # cache_keys: [Dim, N]
        dim, n_samples = cache_keys.shape
        self.adapter = nn.Linear(dim, n_samples, bias=False).to(torch.float32)
        # 核心：使用 Cache Keys 初始化权重
        self.adapter.weight.data = cache_keys.t().clone().to(torch.float32)
        self.cache_values = cache_values  # [N, C]
        self.clip_model = clip_model
        self.text_features = text_features  # [C, Dim]
        self.logit_scale = clip_model.logit_scale.exp().item()
        self.beta = args.init_beta
        self.alpha = args.init_alpha

    def forward(self, image):
        # 提取原始 CLIP 特征
        clip_dtype = self.clip_model.visual.conv1.weight.dtype
        with torch.no_grad():
            image_features = self.clip_model.visual(image.type(clip_dtype)).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
        #  CLIP 分支
        clip_logits = self.logit_scale * image_features @ self.text_features.t()

        # Tip-Adapter 分支 (使用可学习的 adapter 层计算 affinity)
        affinity = self.adapter(image_features)  # [Batch, N]
        cache_logits = ((-1) * (self.beta - self.beta * affinity)).exp() @ self.cache_values

        # 4融合
        logits = clip_logits + self.alpha * cache_logits
        return logits

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False)
            #nn.ReLU(inplace=True)  ## ZY 260424 官方代码中的 adapter 居然接最后了1个 relu让所有输出都是大于0的，非常的神秘，这里给注释掉了
        )
        # 0 初始化权重，使其在初期不至于破坏 CLIP 特征 让人想起controlnet
        nn.init.constant_(self.fc[2].weight, 0)

    def forward(self, x):
        return self.fc(x)

class TextEncoder(nn.Module):
    def __init__(self, dataset_name, classnames, clip_model):
        super().__init__()
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        self.dataset_name = dataset_name

    def forward(self):
        temp = CUSTOM_TEMPLATES.get(self.dataset_name, 'an image of a {}.')
        prompts = [temp.format(c.replace('_', ' ').lower()) for c in self.classnames]
        device = next(self.clip_model.parameters()).device
        prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        text_features = self.clip_model.encode_text(prompts)
        return text_features


class CustomCLIP(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(args.dataset, classnames, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.ratio = args.ratio
        self.dtype = torch.float32

        if "RN50" in args.backbone and "RN101" not in args.backbone:
            fc_dim = 1024
        else:
            fc_dim = 512
        self.adapter = Adapter(fc_dim, args.reduction).to(self.dtype)
        with torch.no_grad():
            # 这里的 text_encoder 会处理好 bike_helmet -> bike helmet
            t_feat = self.text_encoder()
            t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
            self.register_buffer("text_features_cache", t_feat.float())

    def forward(self, image):
        # 提取图像特征（保持 CLIP 原生精度，通常是 float16）
        clip_dtype = self.image_encoder.conv1.weight.dtype
        image_features = self.image_encoder(image.type(clip_dtype))
        # 转换到 float32 进行后续所有计算
        image_features = image_features.type(self.dtype)
        # 先对原始特征做一次归一化（确保基准一致）
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)
        # 残差增强
        x = self.adapter(image_features)
        image_features = self.ratio * x + (1 - self.ratio) * image_features
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)
        # 直接使用缓存的、预先归一化过的文本特征
        text_features = self.text_features_cache.type(self.dtype)
        # 计算 Logits
        # 冻结 scale 的 exp 值，防止其在训练/测试模式间跳变
        scale = self.logit_scale.exp().type(self.dtype)
        logits = scale * image_features @ text_features.t()

        return logits

def get_few_shot_dataset(dataset, k=16):
    if k <= 0: return None
    indices = []
    cls_to_indices = {i: [] for i in range(len(dataset.classes))}
    for idx, (_, label) in enumerate(dataset.samples):
        cls_to_indices[label].append(idx)
    for label in cls_to_indices:
        # 如果样本不够 k 个，则取全部
        available = cls_to_indices[label]
        sample_count = min(k, len(available))
        indices.extend(random.sample(available, sample_count))
    return Subset(dataset, indices)


def build_cache_model(args, clip_model, train_loader_cache, device):
    cache_keys = []
    cache_values = []
    augment_epoch = getattr(args, 'augment_epoch', 10)
    num_classes = len(train_loader_cache.dataset.dataset.classes)

    clip_model.eval()
    with torch.no_grad():
        for augment_idx in range(augment_epoch):
            train_features = []
            print(f'Tip-Cache Augment Epoch: {augment_idx + 1} / {augment_epoch}')
            for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                images = images.to(device)
                clip_dtype = clip_model.visual.conv1.weight.dtype
                image_features = clip_model.visual(images.type(clip_dtype))
                train_features.append(image_features)
                if augment_idx == 0:
                    cache_values.append(target.to(device))
            cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

    cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
    cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
    cache_keys = cache_keys.permute(1, 0).float()
    cache_values = F.one_hot(torch.cat(cache_values, dim=0), num_classes).float()
    return cache_keys, cache_values

def evaluate_tip_adapter(model, loader, cache_keys, cache_values, device, beta=1.0, alpha=1.0):
    model.eval()
    correct, total = 0, 0
    # 预取的文本特征
    text_features = model.text_features_cache.type(torch.float32)
    scale = model.logit_scale.exp().item()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            clip_dtype = model.image_encoder.conv1.weight.dtype
            image_features = model.image_encoder(images.type(clip_dtype)).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)

            clip_logits = scale * image_features @ text_features.t()
            affinity = image_features @ cache_keys.float()  # [Batch, N]

            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values.float()

            tip_logits = clip_logits + alpha * cache_logits
            correct += (tip_logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

def train (args, model, train_loader, val_loader, optimizer, criterion, device, logger, output_dir,
                stage_name="Train"):
    """
    :param stage_name: 用于区分日志输出（如 "Source-Pretrain" 或 "Target-FineTune"）
    """
    best_acc = 0.0
    counter = 0

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    stage_history = {"loss": [], "acc": []}
    for epoch in range(args.epochs):
        model.train()
        model.image_encoder.eval() #freeze CLIP
        model.text_encoder.eval() #freeze CLIP
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"[{stage_name}] Epoch {epoch + 1}/{args.epochs}")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        scheduler.step()

        current_acc = evaluate(model, val_loader, device)
        avg_loss = epoch_loss / len(train_loader)
        stage_history["loss"].append(avg_loss)
        stage_history["acc"].append(current_acc)

        logger.info(f"[{stage_name}] Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f} | Test Acc: {current_acc:.2f}%")
        if current_acc > best_acc:
            best_acc = current_acc
            counter = 0

            save_path = os.path.join(output_dir, f"best_{stage_name.lower()}.pth")
            torch.save(model.adapter.state_dict(), save_path)
            logger.info(f">>> New Best Acc in {stage_name}: {best_acc:.2f}%! Saved to {save_path}")
        else:
            counter += 1
            if counter >= args.patience:
                logger.info(f"[{stage_name}] Early stopping at epoch {epoch + 1}. Best Acc: {best_acc:.2f}%")
                break
    return best_acc


def run_tip_adapter_F_stage(args, clip_model, cache_keys, cache_values, text_features, train_loader, test_loader,
                            device, logger, output_dir):
    model = TipAdapterF(cache_keys, cache_values, clip_model, text_features, args).to(device)
    for param in model.clip_model.parameters():  # freeze CLIP
        param.requires_grad = False
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=args.lr, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        model.clip_model.eval()
        epoch_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Tip-F Training Epoch {epoch + 1}"):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        logger.info(f"[Tip-F] Epoch {epoch + 1} | Loss: {epoch_loss / len(train_loader):.4f} | Acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.adapter.state_dict(), os.path.join(output_dir, "best_tip_f.pth"))
    return best_acc

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"FULL_PIPELINE_{args.dataset}_{timestamp}"
    output_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "pipeline.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    logger = logging.getLogger()
    logger.info("=" * 20 + " Arguments " + "=" * 20)
    args_dict = vars(args)
    for key, value in args_dict.items():
        logger.info(f"{key:20s}: {value}")
    logger.info("=" * 51 + "\n")

    clip_model, preprocess = clip.load(args.backbone, device=device)
    source_dataset = datasets.ImageFolder(root=args.source_path, transform=preprocess)
    target_dataset = datasets.ImageFolder(root=args.target_path, transform=preprocess)

    source_loader_full = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True)
    target_loader_full = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)
    if source_dataset.classes != target_dataset.classes:
        logger.error("Source and Target dataset classes do not match!")
        logger.error(f"Source classes: {source_dataset.classes}")
        logger.error(f"Target classes: {target_dataset.classes}")
        # 抛出异常防止后续错误训练
        raise ValueError("Dataset class mismatch! Please ensure folder names are identical.")
    else:
        logger.info(f"Dataset consistency check passed: {len(source_dataset.classes)} classes found.")

    # 获取 B 域 Few-shot 数据
    target_few_shot_set = get_few_shot_dataset(target_dataset, k=args.shot)
    target_few_shot_loader = DataLoader(target_few_shot_set, batch_size=min(args.batch_size, args.shot), shuffle=True)

    # 重置模型与优化器
    def reset_experiment():
        model = CustomCLIP(args, source_dataset.classes, clip_model).to(device)
        model.image_encoder.eval()
        model.text_encoder.eval()
        for name, param in model.named_parameters():
            param.requires_grad = True if "adapter" in name else False

        optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        return model, optimizer, criterion

    temp_clip = CustomCLIP(args, source_dataset.classes, clip_model).to(device)
    text_features = temp_clip.text_features_cache.clone()
    results_summary = {}

    # B-Domain Zero-shot
    logger.info("\n>>> EXPERIMENT 1: B-Domain Zero-shot")
    model, _, _ = reset_experiment()
    acc = evaluate(model, target_loader_full, device)
    results_summary['Zero-shot'] = acc
    logger.info(f"Zero-shot Acc: {acc:.2f}%")

    #  B-Domain Few-shot
    logger.info("\n>>> EXPERIMENT 2: B-Domain Few-shot")
    model, optimizer, criterion = reset_experiment()
    acc = train(args, model, target_few_shot_loader, target_loader_full,
                optimizer, criterion, device, logger, output_dir, "B-FewShot")
    results_summary['Few-shot'] = acc

    # A-to-B Transfer (A训练后直接用于B)
    logger.info("\n>>> EXPERIMENT 3: A-to-B Transfer")
    model, optimizer, criterion = reset_experiment()
    acc = train(args, model, source_loader_full, target_loader_full,
                optimizer, criterion, device, logger, output_dir, "A-to-B-Transfer")
    results_summary['Transfer'] = acc

    # A-Pretrain + B-Few-shot-FineTune
    logger.info("\n>>> EXPERIMENT 4: A-Pretrain + B-FineTune")
    # 注意：这里不需要手动 reset，要保留 A 的权重
    model, optimizer, criterion = reset_experiment()
    # A-Domain Pretrain
    train(args, model, source_loader_full, target_loader_full,
          optimizer, criterion, device, logger, output_dir, "Pretrain-Stage1")
    model.adapter.load_state_dict(torch.load(os.path.join(output_dir, "best_pretrain-stage1.pth")))
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay)
    # Stage 2: B-Domain Fine-tune
    acc = train(args, model, target_few_shot_loader, target_loader_full,
                optimizer, criterion, device, logger, output_dir, "FineTune-Stage2")
    results_summary['A-Pretrain-B-FineTune'] = acc

    # Tip-Adapter
    logger.info("\n>>> EXPERIMENT 5: Tip-Adapter (Non-parametric)")
    # 使用 target_few_shot_loader 构建缓存
    args.augment_epoch = 10
    # 注意这里最好重新建一个不 shuffle 的 loader 用于构建 cache
    cache_build_loader = DataLoader(target_few_shot_set, batch_size=args.batch_size, shuffle=False)
    ckey, cval = build_cache_model(args, clip_model, cache_build_loader, device)

    acc_tip = evaluate_tip_adapter(temp_clip, target_loader_full, ckey, cval, device,
                                   beta=args.init_beta, alpha=args.init_alpha)

    results_summary['Tip-Adapter'] = acc_tip
    logger.info(f"Tip-Adapter 0-shot Acc: {acc_tip:.2f}%")

    # Tip-Adapter-F
    logger.info("\n>>> EXPERIMENT 6: Tip-Adapter-F (Fine-tuning)")
    # 直接利用刚刚生成的 ckey, cval 进行微调
    acc_tip_f = run_tip_adapter_F_stage(args, clip_model, ckey, cval, text_features,
                                        target_few_shot_loader, target_loader_full,
                                        device, logger, output_dir)
    results_summary['Tip-Adapter-F'] = acc_tip_f

    logger.info("\n" + "=" * 40)
    logger.info("FINAL PIPELINE RESULTS SUMMARY")
    logger.info("=" * 40)
    for k, v in results_summary.items():
        logger.info(f"{k:25s}: {v:.2f}%")
    logger.info("=" * 40)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Universal CLIP-Adapter Trainer")
    # 路径与标识
    parser.add_argument('--source_path', type=str, default='./original_datasets/office_31/amazon', help='Path to source domain images')
    parser.add_argument('--target_path', type=str, default='./original_datasets/office_31/webcam', help='Path to target domain images')
    parser.add_argument('--dataset', type=str, default='Office31', help='Dataset name for template matching (Office31, Officehome, PACS)')
    # 模型架构
    parser.add_argument('--backbone', type=str, default='RN50', help='CLIP backbone (RN50, ViT-B/16, etc.)')
    parser.add_argument('--ratio', type=float, default=0.01, help='Residual ratio (alpha)')
    parser.add_argument('--reduction', type=int, default=4, help='Adapter bottleneck reduction')
    parser.add_argument('--init_beta', type=float, default=5.5, help='Beta for Tip-Adapter')
    parser.add_argument('--init_alpha', type=float, default=1.0, help='Alpha for Tip-Adapter')
    # few-shot 设置
    parser.add_argument('--shot', type=int, default=3, help='K-shot per class')
    # 优化参数
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--patience', type=int, default=6, help='Early stopping patience')
    # 结果保存
    parser.add_argument('--save_dir', type=str, default='checkpoints/amazon_to_webcam_Adapter', help='checkpoint and logs save directory')

    args = parser.parse_args()
    main(args)

    # python  Adapter.py --source_path ./original_datasets/PACS/photo --target_path ./original_datasets/PACS/sketch --dataset PACS --save_dir checkpoints/photo_to_sketch_Adapter