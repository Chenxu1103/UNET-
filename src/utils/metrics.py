"""语义分割评价指标计算模块

计算 mIoU (平均交并比)、Precision (查准率)、Recall (查全率) 等指标
"""
import numpy as np
from typing import Dict, Tuple


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = -1
) -> Tuple[float, Dict[int, float], Dict[int, float], Dict[int, float]]:
    """计算单张或单批次预测的 mIoU、各类别Precision和Recall
    
    Args:
        pred: 预测mask，值为类别ID，shape (H,W) 或 (N,H,W)
        target: 实际mask，同尺寸
        num_classes: 类别总数
        ignore_index: 忽略的类别ID（默认-1表示不忽略）
        
    Returns:
        tuple: (mIoU, precision_dict, recall_dict, iou_dict)
            - mIoU: 平均交并比 (float)
            - precision_dict: 各类别查准率字典 {class_id: precision}
            - recall_dict: 各类别查全率字典 {class_id: recall}
            - iou_dict: 各类别交并比字典 {class_id: iou}
    """
    # 如果pred/target是批次，展平到一维
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    
    # 初始化累加
    IoUs = []
    precision = {}
    recall = {}
    iou_dict = {}
    
    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        # 计算该类的真正例、假正例、假负例
        true_cls = target_flat == cls
        pred_cls = pred_flat == cls

        # 判断是否为背景类
        is_background = (cls == 0)

        # 如果该类在GT中不存在
        if true_cls.sum() == 0:
            # GT没有，预测也没有 -> 算正确(IoU=1.0)
            # GT没有，但预测有   -> 算错误(IoU=0.0)
            if pred_cls.sum() == 0:
                iou = 1.0
            else:
                iou = 0.0
            iou_dict[cls] = iou
            precision[cls] = 1.0 if pred_cls.sum() == 0 else 0.0
            recall[cls] = 1.0
            # 背景类不计入mIoU平均（避免主导），其他类计入
            if not is_background:
                IoUs.append(iou)
            continue

        # Intersection and Union
        inter = (pred_cls & true_cls).sum()
        union = (pred_cls | true_cls).sum()

        # IoU 计算
        if union == 0:
            iou = 1.0
        else:
            iou = inter / float(union)

        iou_dict[cls] = iou

        # 只对非背景类计算mIoU（避免背景主导）
        if not is_background:
            IoUs.append(iou)
        
        # Precision: 预测为cls中正确的比例
        if pred_cls.sum() == 0:
            precision_val = 1.0 if true_cls.sum() == 0 else 0.0
        else:
            precision_val = inter / float(pred_cls.sum())
        precision[cls] = precision_val
        
        # Recall: 实际为cls中被正确预测的比例
        if true_cls.sum() == 0:
            recall_val = 1.0 if pred_cls.sum() == 0 else 0.0
        else:
            recall_val = inter / float(true_cls.sum())
        recall[cls] = recall_val
    
    mIoU = sum(IoUs) / len(IoUs) if IoUs else 0.0
    
    return mIoU, precision, recall, iou_dict


def compute_confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int
) -> np.ndarray:
    """计算混淆矩阵
    
    Args:
        pred: 预测mask，值为类别ID，shape (H,W) 或 (N,H,W)
        target: 实际mask，同尺寸
        num_classes: 类别总数
        
    Returns:
        混淆矩阵，shape (num_classes, num_classes)
    """
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    
    # 初始化混淆矩阵
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # 填充混淆矩阵
    for t, p in zip(target_flat, pred_flat):
        confusion[t, p] += 1
    
    return confusion


def print_metrics(
    mIoU: float,
    precision: Dict[int, float],
    recall: Dict[int, float],
    iou_dict: Dict[int, float],
    class_names: Dict[int, str] = None
) -> None:
    """打印格式化的指标信息
    
    Args:
        mIoU: 平均交并比
        precision: 各类别查准率
        recall: 各类别查全率
        iou_dict: 各类别交并比
        class_names: 类别ID到名称的映射，如None则使用ID
    """
    if class_names is None:
        class_names = {cls: f"class_{cls}" for cls in iou_dict.keys()}
    
    print("\n" + "="*60)
    print(f"{'Metrics Summary':^60}")
    print("="*60)
    print(f"{'mIoU (Mean IoU):':<30} {mIoU:.4f}")
    print("-"*60)
    
    print(f"{'Class':<15} {'IoU':<12} {'Precision':<12} {'Recall':<12}")
    print("-"*60)
    
    for cls_id in sorted(iou_dict.keys()):
        cls_name = class_names.get(cls_id, f"class_{cls_id}")
        iou = iou_dict[cls_id]
        prec = precision.get(cls_id, 0.0)
        rec = recall.get(cls_id, 0.0)
        print(f"{cls_name:<15} {iou:<12.4f} {prec:<12.4f} {rec:<12.4f}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # 示例用法
    pred = np.array([[0, 1, 2], [1, 2, 2], [0, 1, 2]], dtype=np.int32)
    target = np.array([[0, 1, 2], [0, 2, 2], [0, 1, 1]], dtype=np.int32)
    
    mIoU, precision, recall, iou_dict = compute_metrics(pred, target, num_classes=3)
    print_metrics(mIoU, precision, recall, iou_dict)
