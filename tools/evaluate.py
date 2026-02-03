"""模型评估脚本

对训练好的模型进行评估，计算综合性能指标。
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import sys
import json
from datetime import datetime

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.unetplusplus import NestedUNet
from data.dataloader import DataLoaderFactory
from utils.metrics import compute_metrics, compute_confusion_matrix, print_metrics


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    save_dir: Path = None
) -> dict:
    """评估模型在测试集上的性能
    
    Args:
        model: 分割模型
        test_loader: 测试数据加载器
        device: 计算设备
        num_classes: 类别数
        save_dir: 结果保存目录
        
    Returns:
        包含所有评估指标的字典
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    print("评估中...")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(images)
            if isinstance(outputs, list):
                outputs = outputs[0]
            
            # 获取预测
            preds = torch.argmax(outputs, dim=1)  # (B, H, W)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  已处理 {batch_idx + 1} 个批次")
    
    # 合并结果
    import numpy as np
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    print(f"\n总计评估样本数: {len(all_preds)}")
    
    # 计算指标
    print("计算评估指标...")
    metrics = compute_metrics(all_preds, all_targets, num_classes)
    
    # 计算混淆矩阵
    confusion_matrix = compute_confusion_matrix(all_preds, all_targets, num_classes)
    metrics['confusion_matrix'] = confusion_matrix
    
    # 打印指标
    print_metrics(metrics)
    
    # 保存结果
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存指标到 JSON
        metrics_to_save = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in metrics.items() if k != 'confusion_matrix'
        }
        metrics_to_save['confusion_matrix'] = confusion_matrix.tolist()
        metrics_to_save['timestamp'] = datetime.now().isoformat()
        
        with open(save_dir / 'evaluation_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)
        
        # 保存混淆矩阵为 CSV
        import csv
        with open(save_dir / 'confusion_matrix.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入类别名称
            class_names = ['background', 'cable', 'tape', 'bulge_defect', 
                          'loose_defect', 'burr_defect', 'thin_defect']
            writer.writerow([''] + class_names[:num_classes])
            # 写入混淆矩阵
            for i, row in enumerate(confusion_matrix):
                writer.writerow([class_names[i]] + row.tolist())
        
        print(f"\n评估结果已保存到 {save_dir}")
    
    return metrics


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device
) -> nn.Module:
    """加载模型检查点
    
    Args:
        model: 模型实例
        checkpoint_path: 检查点路径
        device: 计算设备
        
    Returns:
        加载权重后的模型
    """
    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("检查点加载完成")
    return model


def main():
    parser = argparse.ArgumentParser(description='模型评估脚本')
    
    # 数据参数
    parser.add_argument('--test-img-dir', type=str, 
                       default='dataset/processed/test/images',
                       help='测试图像目录')
    parser.add_argument('--test-mask-dir', type=str,
                       default='dataset/processed/test/masks',
                       help='测试 mask 目录')
    
    # 模型参数
    parser.add_argument('--model-path', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--num-classes', type=int, default=7,
                       help='类别数')
    
    # 评估参数
    parser.add_argument('--batch-size', type=int, default=8,
                       help='批次大小')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='数据加载工作进程数')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备 (cuda/cpu)')
    
    # 保存参数
    parser.add_argument('--save-dir', type=str, default='results/evaluation',
                       help='结果保存目录')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    print("\n加载测试数据...")
    test_loader = DataLoaderFactory.create_val_loader(
        images_dir=args.test_img_dir,
        masks_dir=args.test_mask_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 构建模型
    print("\n构建模型...")
    model = NestedUNet(
        num_classes=args.num_classes,
        input_channels=3,
        deep_supervision=False  # 评估时关闭深监督
    ).to(device)
    
    # 加载检查点
    model = load_checkpoint(model, args.model_path, device)
    
    # 评估模型
    print("\n开始评估...\n")
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        num_classes=args.num_classes,
        save_dir=args.save_dir
    )
    
    print("\n" + "="*50)
    print("评估完成！")
    print("="*50)


if __name__ == '__main__':
    main()
