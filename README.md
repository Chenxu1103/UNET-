# 电缆绕包质量检测系统

基于深度学习的电缆绕包质量实时检测系统，支持电缆、胶带分割和毛刺缺陷检测。

## 项目特点

- ✅ **高精度分割**：基于UNet++的电缆/胶带语义分割（mIoU 79.97%）
- ✅ **毛刺检测**：多尺度边缘融合算法，准确识别电缆表面毛刺
- ✅ **图像增强**：CLAHE+去噪+锐化，适应低光照环境
- ✅ **ROI检测**：支持自定义ROI，减少背景干扰
- ✅ **多分辨率支持**：自动归一化不同分辨率视频
- ✅ **实时处理**：GPU加速，10-15 FPS

## 系统架构

```
电缆绕包检测系统
├── 阶段1: 电缆/胶带分割（UNet++模型）
├── 阶段2: 毛刺检测（规则法+边缘检测）
└── 阶段3: 结果可视化与统计
```

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (GPU加速)
- OpenCV 4.5+

## 安装

```bash
# 克隆项目
git clone https://github.com/YOUR_USERNAME/cable-wrapping-detection.git
cd cable-wrapping-detection

# 创建虚拟环境
python -m venv gpu_env
source gpu_env/bin/activate  # Windows: gpu_env\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 标准分辨率视频检测（800x448）

```bash
python infer_two_stage_burr.py \
  --video data/raw/video.mp4 \
  --model checkpoints_3class_advanced/best_model.pth \
  --output log/output \
  --device cuda
```

### 2. 高分辨率视频检测（自动归一化）

```bash
python infer_two_stage_burr.py \
  --video data/raw/high_res_video.avi \
  --model checkpoints_3class_advanced/best_model.pth \
  --output log/output \
  --device cuda \
  --rotate \
  --normalize-resolution \
  --target-width 800 \
  --target-height 448
```

### 3. 增强版毛刺检测（图像增强）

```bash
python infer_enhanced_burr.py \
  --video data/raw/video.avi \
  --model checkpoints_3class_advanced/best_model.pth \
  --output log/output \
  --device cuda \
  --enhance
```

### 4. 自定义ROI检测

```bash
python infer_high_res_custom_roi.py \
  --video data/raw/video.avi \
  --model checkpoints_3class_advanced/best_model.pth \
  --output log/output \
  --device cuda
```

## 项目结构

```
cable-wrapping-detection/
├── src/                          # 源代码
│   ├── models/                   # 模型定义
│   │   ├── unetpp.py            # UNet++模型
│   │   └── losses.py            # 损失函数
│   └── utils/                    # 工具函数
│       └── geometry_enhanced.py  # 几何测量
├── data/                         # 数据目录
│   ├── raw/                     # 原始视频
│   └── labelme/                 # 标注数据
├── tools/                        # 工具脚本
│   ├── extract_frames.py       # 帧提取
│   ├── calibrate_roi.py        # ROI标定
│   └── train.py                # 训练脚本
├── infer_two_stage_burr.py      # 主检测脚本
├── infer_enhanced_burr.py       # 增强版检测
├── infer_high_res_custom_roi.py # 自定义ROI检测
└── README.md                    # 项目说明
```

## 核心功能

### 1. 电缆/胶带分割

使用UNet++模型进行语义分割：
- 输入：512x512 RGB图像
- 输出：3类分割（背景、电缆、胶带）
- 性能：mIoU 79.97%

### 2. 毛刺检测

多尺度边缘融合算法：
- Canny边缘检测
- Sobel边缘检测
- Laplacian边缘检测
- 形态学过滤
- 连通域分析

### 3. 图像增强

针对低光照环境的增强：
- CLAHE（对比度限制自适应直方图均衡化）
- 非局部均值去噪
- 锐化滤波

### 4. ROI检测

支持自定义ROI配置：
- 固定ROI坐标
- 自动分辨率映射
- ROI外区域遮罩

## 检测结果

### 参考视频（800x448）

- 电缆检测：0%-59%
- 胶带检测：0%-60%
- 毛刺检测：0帧（无误检）
- 处理速度：14.59 FPS

### 高分辨率视频（2448x2048 → 800x448）

- 电缆检测：0%-10%
- 胶带检测：0%-0.1%
- 毛刺检测：22帧（0.9%）
- 处理速度：10.66 FPS

## 模型训练

### 数据准备

```bash
# 提取视频帧
python tools/extract_frames.py --video data/raw/video.mp4 --output data/frames

# 使用labelme标注
labelme data/frames

# 转换标注数据
python data/prepare_dataset_cli.py --input data/labelme --output dataset/processed
```

### 训练模型

```bash
python tools/train.py \
  --dataset dataset/processed \
  --output checkpoints \
  --epochs 100 \
  --batch-size 8 \
  --lr 0.001
```

## 参数说明

### 主要参数

- `--video`: 输入视频路径
- `--model`: 模型权重路径
- `--output`: 输出目录
- `--device`: 计算设备（cuda/cpu）
- `--rotate`: 旋转视频90度
- `--normalize-resolution`: 启用分辨率归一化
- `--enhance`: 启用图像增强
- `--burr-sensitivity`: 毛刺检测灵敏度（low/medium/high）

### ROI配置

在脚本中修改ROI坐标：

```python
CUSTOM_ROI = {
    'x1': 250,   # 左边界
    'y1': 0,     # 上边界
    'x2': 550,   # 右边界
    'y2': 448    # 下边界
}
```

## 性能优化

- GPU加速：使用CUDA进行模型推理
- 批处理：支持批量处理多个视频
- 多线程：视频读取与推理并行
- 模型量化：支持INT8量化（RV1126部署）

## 部署

### RV1126环境部署

```bash
# 转换模型为ONNX格式
python tools/export_onnx.py --model checkpoints/best_model.pth --output model.onnx

# 转换为RKNN格式
python tools/convert_rknn.py --onnx model.onnx --output model.rknn
```

## 常见问题

### Q: 检测效果差怎么办？

A: 检查以下几点：
1. ROI是否正确框住电缆区域
2. 视频分辨率是否与训练数据匹配
3. 光照条件是否相似
4. 尝试启用图像增强（--enhance）

### Q: 如何调整ROI？

A: 使用ROI标定工具：
```bash
python tools/calibrate_roi.py --video data/raw/video.mp4 --output roi.json
```

### Q: 如何提高检测速度？

A:
1. 使用GPU加速（--device cuda）
2. 降低输入分辨率
3. 减少帧率（--frame-stride 2）

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue或Pull Request。

## 更新日志

### v1.0.0 (2026-02-03)

- ✅ 实现UNet++电缆/胶带分割
- ✅ 实现多尺度毛刺检测算法
- ✅ 支持图像增强处理
- ✅ 支持多分辨率视频
- ✅ 支持自定义ROI配置
- ✅ 完整的检测流程和可视化
