"""TensorRT 引擎构建脚本

将 ONNX 模型转换为 TensorRT 引擎以实现边缘设备上的高速推理
"""
import sys
from pathlib import Path
import argparse


def build_engine(onnx_file: str, engine_file: str, fp16: bool = True):
    """将ONNX模型构建为TensorRT引擎
    
    Args:
        onnx_file: ONNX 模型文件路径
        engine_file: 输出的 TensorRT 引擎文件路径
        fp16: 是否使用 FP16 半精度加速（需硬件支持）
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("ERROR: TensorRT not installed. Please install TensorRT to build engines.")
        print("  https://developer.nvidia.com/tensorrt")
        return False
    
    print("="*70)
    print("TensorRT Engine Builder")
    print("="*70)
    print(f"ONNX file: {onnx_file}")
    print(f"Output engine: {engine_file}")
    print(f"FP16 enabled: {fp16}")
    print()
    
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    
    # EXPLICIT_BATCH 允许指定动态batch维
    network = builder.create_network(
        flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    
    # 解析ONNX模型
    print("[1] Parsing ONNX model...")
    with open(onnx_file, "rb") as f:
        success = parser.parse(f.read())
        if not success:
            print("ERROR: Failed to parse ONNX model!")
            for i in range(parser.num_errors):
                print(f"  Error {i}: {parser.get_error(i)}")
            return False
    print("  ✓ ONNX parsed successfully")
    
    # 配置builder
    print("\n[2] Configuring builder...")
    config = builder.create_builder_config()
    
    # 设置最大工作空间（GPU内存）
    workspace_size = 1 << 30  # 1GB
    config.max_workspace_size = workspace_size
    print(f"  Max workspace size: {workspace_size / (1024**3):.1f} GB")
    
    # 启用FP16精度（如果硬件支持）
    if fp16 and builder.platform_has_fast_fp16:
        config.flags |= 1 << int(trt.BuilderFlag.FP16)
        print("  ✓ FP16 mode enabled")
    elif fp16:
        print("  WARNING: FP16 not supported on this platform, using FP32")
    else:
        print("  FP32 mode")
    
    # 创建推理配置文件（可选）
    # profile = builder.create_optimization_profile()
    # profile.set_shape("input", (1, 3, 512, 512), (1, 3, 512, 512), (4, 3, 512, 512))
    # config.add_optimization_profile(profile)
    
    # 构建TensorRT引擎
    print("\n[3] Building engine (this may take a few minutes)...")
    engine = builder.build_engine(network, config)
    
    if engine is None:
        print("ERROR: Failed to build engine!")
        return False
    
    # 保存引擎
    print("\n[4] Saving engine...")
    with open(engine_file, "wb") as f:
        f.write(engine.serialize())
    print(f"  ✓ Engine saved: {engine_file}")
    
    # 打印引擎信息
    print("\n[5] Engine Information:")
    print(f"  Bindings: {engine.num_bindings}")
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        shape = engine.get_binding_shape(i)
        dtype = engine.get_binding_dtype(i)
        print(f"    [{i}] {name}: shape={shape}, dtype={dtype}")
    
    print("\n" + "="*70)
    print("✓ TensorRT engine built successfully!")
    print("="*70)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX model")
    parser.add_argument("onnx_file", type=str, help="Path to ONNX model file")
    parser.add_argument("--output", type=str, default=None, 
                       help="Output TensorRT engine file (default: replace .onnx with .engine)")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 precision")
    
    args = parser.parse_args()
    
    # 确定输出文件名
    if args.output is None:
        args.output = args.onnx_file.replace(".onnx", ".engine")
    
    # 构建引擎
    success = build_engine(args.onnx_file, args.output, fp16=not args.no_fp16)
    
    sys.exit(0 if success else 1)
