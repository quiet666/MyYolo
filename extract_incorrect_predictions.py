#!/usr/bin/env python3
"""
从预测结果中提取分类错误的图片并保存到指定目录
此脚本假设 PREDICT_VAL_DIR 的目录结构与 VAL_DIR 完全相同，
并且已经包含了带有预测标注的图片（例如由 'yolo predict' 生成）。
"""

import os
from pathlib import Path
import shutil
import cv2
from collections import defaultdict

# ===================== 配置区 =====================
# 原始验证集目录
VAL_DIR = Path("/root/autodl-tmp/ultralytics-main/datas/WX_class/defect_all2_split/val")

# 预测结果目录 (结构应该与val一致)
PREDICT_VAL_DIR = Path("/root/autodl-tmp/ultralytics-main/runs/classify/predict_defect_all_2")

# 错误分类图片保存目录
INCORRECT_DIR = Path("/root/autodl-tmp/ultralytics-main/runs/classify/predict_defect_all_2/incorrect")

# 支持的图片扩展名
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# =================================================

def is_image(p: Path) -> bool:
    """判断是否为图片文件"""
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def get_true_label(folder_name: str) -> int:
    """从文件夹名中解析真实标签"""
    try:
        return int(folder_name.split("_")[0])
    except Exception:
        try:
            return int(folder_name)
        except Exception:
            return -1

def extract_class_from_folder_name(folder_name: str) -> str:
    """从文件夹名中提取类别名"""
    parts = folder_name.split("_")
    if len(parts) > 1:
        return "_".join(parts[1:])  # 返回除了数字前缀外的部分
    return folder_name

def main():
    print("开始处理预测结果...")
    
    # 如果目标目录已存在，则删除并重新创建
    if INCORRECT_DIR.exists():
        shutil.rmtree(INCORRECT_DIR)
        print(f"已删除已有目录: {INCORRECT_DIR}")
    INCORRECT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"创建目录: {INCORRECT_DIR}")
    
    # 统计每个类别的错误图片数量
    error_counts = defaultdict(int)
    
    # 遍历验证集的每个类别文件夹
    for true_label_folder in VAL_DIR.iterdir():
        if not true_label_folder.is_dir():
            continue
            
        # 获取真实标签
        true_label = get_true_label(true_label_folder.name)
        if true_label == -1:
            print(f"警告: 无法解析真实标签 {true_label_folder.name}")
            continue
            
        true_label_name = str(true_label) + "_" + extract_class_from_folder_name(true_label_folder.name)
        print(f"处理真实标签类别: {true_label_name} (来自文件夹 {true_label_folder.name})")
        
        # 对应的预测结果目录
        predict_label_dir = PREDICT_VAL_DIR / true_label_folder.name
        
        if not predict_label_dir.exists():
            print(f"警告: 预测目录不存在 {predict_label_dir}")
            continue
            
        # 为该真实标签创建保存目录
        save_dir = INCORRECT_DIR / f"{true_label_name}_classified_as"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 收集该文件夹内的图片
        images = [p for p in true_label_folder.rglob("*") if is_image(p)]
        if not images:
            print(f"警告: {true_label_folder} 中未找到图片")
            continue
            
        print(f"  处理 {len(images)} 张图片")
        
        for img_path in images:
            # 构建预测结果图片路径
            predict_img_path = predict_label_dir / img_path.name
            
            # 检查预测结果图片是否存在
            if not predict_img_path.exists():
                print(f"  警告: 预测图片不存在 {predict_img_path}")
                continue
                
            # 读取预测结果图片
            img = cv2.imread(str(predict_img_path))
            if img is None:
                print(f"  警告: 无法读取图片 {predict_img_path}")
                continue
                
            # 直接复制图片到目标目录
            out_path = save_dir / img_path.name
            shutil.copy2(str(predict_img_path), str(out_path))
            error_counts[true_label_name] += 1
                
    # 输出统计结果
    print("\n每个类别错误的图片数量:")
    total_errors = 0
    for class_name, count in error_counts.items():
        print(f"  {class_name}: {count} 张")
        total_errors += count
        
    print(f"\n总共错误分类图片: {total_errors} 张")
    print(f"错误分类的图片保存在: {INCORRECT_DIR}")

if __name__ == "__main__":
    main()