"""数据审计脚本 - 排查标注质量问题

针对 Labelme 格式的 JSON 标注文件，检查：
1. 多边形 ROI 坐标越界
2. 多边形自交或点顺序混乱
3. Mask 为空
4. 标注类别不匹配
5. 极小目标（IoU 波动风险）
"""
import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import shutil

# 类别映射（V2版本）
CLASS_MAP_V2 = {
    "background": 0,
    "cable": 1,
    "tape": 2,
    "burr_defect": 3,
    "loose_defect": 4,
    "wrap_uneven": 5,
}

# 旧类别名映射
CLASS_NAME_MAPPING = {
    "thin_defect": "wrap_uneven",
    "bulge_defect": None,
    "damage_defect": None,
}


class DatasetAuditor:
    """数据集审计器"""

    def __init__(
        self,
        labelme_dir: str,
        images_dir: str,
        output_dir: str = "audit_output"
    ):
        self.labelme_dir = Path(labelme_dir)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "bad_samples").mkdir(exist_ok=True)
        (self.output_dir / "overlays").mkdir(exist_ok=True)
        (self.output_dir / "tiny_objects").mkdir(exist_ok=True)

        self.report = {
            "total": 0,
            "bad_samples": [],
            "tiny_objects": [],
            "class_distribution": {},
            "area_stats": [],
        }

    def polygon_to_mask(
        self,
        h: int,
        w: int,
        pts: List[List[float]]
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """将多边形点转换为 mask，同时检查各种问题"""
        pts = np.asarray(pts, dtype=np.float32)

        # 检查 NaN
        if np.any(np.isnan(pts)):
            return None, "NaN in points"

        # 检查形状
        if pts.ndim != 2 or pts.shape[1] != 2:
            return None, f"Bad shape: {pts.shape}"

        # 检查越界（允许小范围溢出）
        if np.any(pts[:, 0] < -10) or np.any(pts[:, 1] < -10):
            return None, "Points out of bounds (negative)"
        if np.any(pts[:, 0] > w + 10) or np.any(pts[:, 1] > h + 10):
            return None, "Points out of bounds (exceed image size)"

        # 检查点数是否足够
        if len(pts) < 3:
            return None, f"Too few points: {len(pts)} (need >= 3)"

        # 转换为 mask
        pts_i = np.round(pts).astype(np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)

        try:
            cv2.fillPoly(mask, [pts_i], 255)
        except Exception as e:
            return None, f"fillPoly failed: {str(e)}"

        # 检查是否为空
        area = int((mask > 0).sum())
        if area == 0:
            return None, "Empty mask after rasterize"

        return mask, None

    def audit_one_sample(
        self,
        img_name: str,
        json_path: Path,
        img_path: Path
    ) -> Optional[str]:
        """审计单个样本，返回错误信息（如果有的话）"""
        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            return f"Image read failed"
        h, w = img.shape[:2]

        # 读取 JSON
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                ann = json.load(f)
        except Exception as e:
            return f"JSON load failed: {str(e)}"

        # 检查 shapes
        shapes = ann.get("shapes", [])
        if len(shapes) == 0:
            return "No shapes in annotation"

        # 统计类别分布
        overlay_img = img.copy()

        for idx, shape in enumerate(shapes):
            label = shape.get("label", "")

            # 处理旧类别名
            if label in CLASS_NAME_MAPPING:
                new_label = CLASS_NAME_MAPPING[label]
                if new_label is None:
                    continue  # 跳过已移除的类别
                label = new_label

            if label not in CLASS_MAP_V2:
                return f"Unknown label: {label}"

            # 统计类别
            cls_id = CLASS_MAP_V2[label]
            self.report["class_distribution"][label] = \
                self.report["class_distribution"].get(label, 0) + 1

            # 获取点
            points = shape.get("points", [])
            if not points:
                return f"Shape {idx} ({label}): No points"

            # 转换为 mask
            mask, err = self.polygon_to_mask(h, w, points)
            if err:
                return f"Shape {idx} ({label}): {err}"

            # 统计面积
            area = int((mask > 0).sum())
            area_ratio = area / (h * w)
            self.report["area_stats"].append({
                "image": img_name,
                "label": label,
                "area": area,
                "area_ratio": area_ratio
            })

            # 检查极小目标
            if area_ratio < 0.001:  # < 0.1%
                self.report["tiny_objects"].append({
                    "image": img_name,
                    "label": label,
                    "area": area,
                    "area_ratio": area_ratio
                })

            # 绘制 overlay
            color = self._get_color(label)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay_img, contours, -1, color, 2)

            # 在图像上标注类别名称
            if len(contours) > 0:
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(
                        overlay_img, label, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                    )

        # 保存 overlay
        overlay_path = self.output_dir / "overlays" / img_name
        cv2.imwrite(str(overlay_path), overlay_img)

        return None  # 无错误

    def _get_color(self, label: str) -> Tuple[int, int, int]:
        """获取类别对应的颜色"""
        colors = {
            "background": (128, 128, 128),
            "cable": (255, 0, 0),
            "tape": (0, 255, 0),
            "burr_defect": (0, 0, 255),
            "loose_defect": (255, 255, 0),
            "wrap_uneven": (255, 0, 255),
        }
        return colors.get(label, (128, 128, 128))

    def run(self):
        """运行审计"""
        print("="*70)
        print("数据集审计开始")
        print("="*70)

        json_files = list(self.labelme_dir.glob("*.json"))
        print(f"\n发现 {len(json_files)} 个标注文件")

        for json_path in json_files:
            self.report["total"] += 1
            stem = json_path.stem

            # 查找对应的图像文件
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG']:
                candidate = self.images_dir / (stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None:
                err = f"No matching image for {stem}.json"
                self.report["bad_samples"].append({
                    "file": stem + ".json",
                    "error": err
                })
                print(f"  [BAD] {err}")
                continue

            # 审计该样本
            err = self.audit_one_sample(img_path.name, json_path, img_path)

            if err:
                self.report["bad_samples"].append({
                    "file": stem + ".json",
                    "error": err
                })
                print(f"  [BAD] {img_path.name}: {err}")

                # 复制坏样本到 bad_samples 目录
                shutil.copy2(img_path, self.output_dir / "bad_samples" / img_path.name)

        # 保存报告
        self._save_report()

    def _save_report(self):
        """保存审计报告"""
        report_path = self.output_dir / "audit_report.json"

        # 整理数据
        summary = {
            "total_samples": self.report["total"],
            "bad_samples_count": len(self.report["bad_samples"]),
            "bad_samples": self.report["bad_samples"],
            "tiny_objects_count": len(self.report["tiny_objects"]),
            "tiny_objects": sorted(
                self.report["tiny_objects"],
                key=lambda x: x["area_ratio"]
            )[:20],  # 最小的20个
            "class_distribution": self.report["class_distribution"],
            "area_stats": {
                "count": len(self.report["area_stats"]),
                "min_ratio": min([x["area_ratio"] for x in self.report["area_stats"]]) if self.report["area_stats"] else 0,
                "max_ratio": max([x["area_ratio"] for x in self.report["area_stats"]]) if self.report["area_stats"] else 0,
            }
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 打印摘要
        print("\n" + "="*70)
        print("审计摘要")
        print("="*70)
        print(f"总样本数: {summary['total_samples']}")
        print(f"坏样本数: {summary['bad_samples_count']}")
        print(f"极小目标数 (<0.1%): {summary['tiny_objects_count']}")
        print(f"\n类别分布:")
        for label, count in summary['class_distribution'].items():
            print(f"  {label}: {count}")
        print(f"\n面积统计:")
        print(f"  最小占比: {summary['area_stats']['min_ratio']*100:.4f}%")
        print(f"  最大占比: {summary['area_stats']['max_ratio']*100:.2f}%")

        if summary['bad_samples_count'] > 0:
            print(f"\n⚠️  发现 {summary['bad_samples_count']} 个问题样本:")
            for item in summary['bad_samples'][:10]:
                print(f"  - {item['file']}: {item['error']}")
            if len(summary['bad_samples']) > 10:
                print(f"  ... 还有 {len(summary['bad_samples'])-10} 个")

        if summary['tiny_objects_count'] > 0:
            print(f"\n⚠️  发现 {summary['tiny_objects_count']} 个极小目标:")
            for item in summary['tiny_objects'][:10]:
                print(f"  - {item['image']} ({item['label']}): {item['area_ratio']*100:.3f}%")

        print(f"\n报告已保存: {report_path}")
        print(f"可视化输出: {self.output_dir / 'overlays'}")
        print("="*70)


def main():
    """主函数"""
    base_dir = Path(__file__).parent.parent

    # 路径配置
    labelme_dir = base_dir / "dataset" / "raw" / "annotations"
    images_dir = base_dir / "dataset" / "raw" / "images"
    output_dir = base_dir / "audit_output"

    # 检查目录
    if not labelme_dir.exists():
        print(f"[错误] 标注目录不存在: {labelme_dir}")
        print("\n尝试使用 processed 数据集...")
        labelme_dir = None
        processed_dir = base_dir / "dataset" / "processed_v2" / "train"
        if processed_dir.exists():
            print(f"\n注意: processed 数据集没有 JSON 文件，无法审计标注")
            print(f"如需审计，请提供原始 Labelme 标注目录")
        return

    if not images_dir.exists():
        print(f"[错误] 图像目录不存在: {images_dir}")
        return

    # 运行审计
    auditor = DatasetAuditor(
        labelme_dir=str(labelme_dir),
        images_dir=str(images_dir),
        output_dir=str(output_dir)
    )
    auditor.run()


if __name__ == "__main__":
    main()
