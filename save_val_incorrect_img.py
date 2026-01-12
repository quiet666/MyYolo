#!/usr/bin/env python3
import os
from pathlib import Path
import shutil
import cv2
import torch
from ultralytics import YOLO

# ===================== 配置区 =====================
MODEL_PATH = "/root/autodl-tmp/ultralytics-main/runs/classify/train8/weights/best.pt"  # 你的分类模型 .pt
VAL_DIR = Path("/root/autodl-tmp/ultralytics-main/datas/WX_class/defect_all2_split/val")  # 形如 val/0_xxx, val/1_xxx ...
INCORRECT_DIR = Path("/root/autodl-tmp/ultralytics-main/runs/classify/predict_defect_all/save_incorrect")  # 修改路径

if INCORRECT_DIR.exists():
    shutil.rmtree(INCORRECT_DIR)
    print(f"目录 {INCORRECT_DIR} 已删除")
INCORRECT_DIR.mkdir(parents=True, exist_ok=True)  # 支持的图片扩展名
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# =================================================
def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def main():
    # 加载模型（分类模型）
    model = YOLO(MODEL_PATH)
    names = model.names  # 类别名字典，如 {0: '0_normal', 1: '3_py', ...}

    # 错误图片数量统计
    error_counts = {name: 0 for name in names.values()}  # 初始化为0

    # 遍历验证集的每个类别文件夹
    for label_folder in VAL_DIR.iterdir():
        if not label_folder.is_dir():
            continue

        # 从文件夹名中解析真实标签
        folder_name = label_folder.name
        try:
            true_label = int(folder_name.split("_")[0])
        except Exception:
            try:
                true_label = int(folder_name)
            except Exception:
                continue

        # 真实标签名（如果 model.names 提供）
        true_label_name = names.get(true_label, str(true_label))

        # 为该真实标签创建保存目录
        save_dir = INCORRECT_DIR / f"{true_label_name}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # 收集该文件夹内的图片
        images = [p for p in label_folder.rglob("*") if is_image(p)]
        if not images:
            continue

        for img_path in images:
            # 推理（单张图片）
            with torch.no_grad():
                results = model(str(img_path))  # 获取推理结果
            r = results[0]

            # ---------------- 关键点：直接使用r.probs进行归一化 ----------------
            probs = r.probs.data.cpu()  # 获取原始概率
            probs_normalized = probs / probs.sum()  # 归一化，确保所有概率和为 1
            # ------------------------------------------------------

            # 仅保存预测错误的图片
            if probs_normalized.argmax() != true_label:
                img = cv2.imread(str(img_path))
                if img is None:  # 读图失败则直接拷贝原图
                    shutil.copy2(str(img_path), str(save_dir / img_path.name))
                    continue

                # 获取图片尺寸
                h, w = img.shape[:2]

                # 文字绘制设置：改进字体颜色和行间距
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.3  # 更小的字体
                font_thickness = 1  # 更细的字体
                color = (255, 255, 255)  # 绿色字体，使其更清晰

                # 按置信度从高到低排序
                sorted_probs = sorted(enumerate(probs_normalized), key=lambda x: x[1], reverse=True)

                # 标记所有类别的预测置信度
                y_offset = 30
                for i, prob in sorted_probs:
                    class_name = names.get(i, str(i))
                    conf = prob.item()  # 归一化后的概率值
                    text = f"{class_name} {conf:.2f}"

                    # 如果文本超出图片宽度，则换行
                    if len(text) > 20:  # 超过20个字符的文本换行
                        text = f"{class_name}\n{conf:.2f}"

                    # 在图片上绘制文字
                    cv2.putText(img, text, (10, y_offset), font, font_scale, color, font_thickness, cv2.LINE_AA)
                    y_offset += 20  # 每行文字间隔更小

                # 保存结果图片
                out_path = save_dir / img_path.name
                cv2.imwrite(str(out_path), img)

                # 更新错误图片数量
                error_counts[true_label_name] += 1

    # 输出每个类别的错误图片数量
    print("每个类别错误的图片数量：")
    for class_name, count in error_counts.items():
        print(f"{class_name}: {count} 张")

    print(f"已处理并保存错误分类的图片，路径：{INCORRECT_DIR}")

if __name__ == "__main__":
    main()
