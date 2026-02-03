"""ONNX 模型导出脚本

将训练好的 PyTorch 模型导出为 ONNX 格式，用于跨平台推理和TensorRT加速
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import argparse
import torch

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.unetplusplus import NestedUNet


def export_onnx(
    model_path: str,
    onnx_path: str,
    num_classes: int = 7,
    input_size: int = 512,
    pretrained_encoder: bool = True,
    opset_version: int = 11
) -> bool:
    """导出PyTorch模型为ONNX格式
    
    Args:
        model_path: PyTorch 模型检查点路径
        onnx_path: 输出 ONNX 文件路径
        num_classes: 分类数
        input_size: 输入图像尺寸
        pretrained_encoder: 是否使用预训练编码器
        opset_version: ONNX opset 版本
        
    Returns:
        导出是否成功
    """
    print("="*70)
    print("ONNX Model Export")
    print("="*70)
    print(f"Model path: {model_path}")
    print(f"ONNX output: {onnx_path}")
    print()
    
    device = torch.device("cpu")
    
    # 加载模型
    print("[1] Loading model...")
    model = NestedUNet(
        num_classes=num_classes,
        deep_supervision=False,  # 导出时禁用深度监督，只保留主输出
        pretrained_encoder=pretrained_encoder
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"  ✓ Model loaded from {model_path}")
    except Exception as e:
        print(f"  ERROR: Failed to load model: {e}")
        return False
    
    model.eval()
    
    # 生成示例输入
    print(f"\n[2] Creating dummy input (1, 3, {input_size}, {input_size})...")
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
    
    # 导出为ONNX
    print(f"\n[3] Exporting to ONNX (opset={opset_version})...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=opset_version,
            do_constant_folding=True,
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            },
            verbose=False
        )
        print(f"  ✓ ONNX model exported to {onnx_path}")
    except Exception as e:
        print(f"  ERROR: Failed to export ONNX: {e}")
        return False
    
    # 验证ONNX模型
    print("\n[4] Validating ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("  ✓ ONNX model validation passed")
    except ImportError:
        print("  WARNING: ONNX not installed, skipping validation")
    except Exception as e:
        print(f"  WARNING: ONNX validation failed: {e}")
    
    print("\n" + "="*70)
    print("✓ ONNX export completed successfully!")
    print("="*70)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("model_path", type=str, help="Path to PyTorch model checkpoint")
    parser.add_argument("--output", type=str, default="model/unetpp.onnx",
                       help="Output ONNX file path")
    parser.add_argument("--num_classes", type=int, default=7,
                       help="Number of classes")
    parser.add_argument("--input_size", type=int, default=512,
                       help="Input image size")
    parser.add_argument("--no-pretrained", action="store_true",
                       help="Don't use pretrained encoder")
    parser.add_argument("--opset", type=int, default=11,
                       help="ONNX opset version")
    
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # 导出模型
    success = export_onnx(
        args.model_path,
        args.output,
        num_classes=args.num_classes,
        input_size=args.input_size,
        pretrained_encoder=not args.no_pretrained,
        opset_version=args.opset
    )
    
    sys.exit(0 if success else 1)
    print(f"Exported: {onnx_path}")


if __name__ == "__main__":
    export_onnx(
        pt_path="runs/unetpp_resnet18/best.pt",
        onnx_path="runs/unetpp_resnet18/best.onnx",
        encoder="resnet18",
        num_classes=4,
        input_size=512,
    )
