import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import datetime
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def setup_logging_start():
    """设置并打印程序开始日志."""
    start_time = datetime.datetime.now()
    print(f"程序开始运行时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("开始训练 YOLO 模型...")
    print("-" * 50)
    return start_time


def setup_logging_end(start_time):
    """打印程序结束日志和总耗时."""
    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print("-" * 50)
    print(f"训练完成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总训练耗时: {total_time}")


def delete_ipynb_checkpoints(root_dir):
    """删除 .ipynb_checkpoints / .cache."""
    print("【执行 delete_ipynb_checkpoints()】")
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        if os.path.basename(dirpath) == ".ipynb_checkpoints":
            print(f"【Deleting】: {dirpath}")
            shutil.rmtree(dirpath)
        elif os.path.basename(dirpath) == ".cache":
            print(f"【Deleting】: {dirpath}")
            shutil.rmtree(dirpath)


# ---------------------------------------------------------------------------------------------


def run_detection_training(model, data, save_name="train"):
    """封装检测训练代码."""
    print("------------------------------------------- 【运行检测-训练】 -------------------------------------------")
    model = YOLO(model=f"ultralytics/cfg/models/11/{model}.yaml").load(f"{model}.pt")
    print("【在v7.3基础上进行微调】")

    results = model.train(
        data=data,
        epochs=200,
        patience=50,
        batch=16,
        imgsz=640,
        val=True,
        workers=16,
        name=save_name,
        resume=False,
        cache=True,
    )
    print("检测训练代码已执行。")
    print("-" * 50)
    return results


def run_detection_validation(checkpoint, data, save_name="val", split="val"):
    """封装检测验证代码."""
    print("------------------------------------------- 【运行检测-验证】 -------------------------------------------")
    print(f"checkpoint 路径为：{checkpoint}")

    model = YOLO(checkpoint)
    model.val(
        data=data,
        imgsz=640,
        split=split,
        name=save_name,
        save_conf=True,
    )
    print("检测验证代码已执行（当前注释状态）")
    print("-" * 50)


def run_detection_prediction(checkpoint, predict_data, save_name="predict", conf=0.1):
    print("------------------------------------------- 【运行检测-预测】 -------------------------------------------")
    model = YOLO(checkpoint)
    model.predict(
        predict_data,
        save=True,
        line_width=3,
        name=save_name,
    )
    print("检测-预测代码已执行（当前注释状态）")
    print("-" * 50)


def run_detection_prediction_power(checkpoint, predict_data, conf=0.1, low_conf_threshold=0.5, save_path="./save"):
    print("--------------【运行检测预测】--------------")
    model = YOLO(checkpoint)

    # 类别优先级（高 → 低）
    priority_list = ["fz", "cl", "py", "undefined", "normal"]

    # 构建保存根目录：xxx_split/xxx-results/
    base_name = os.path.basename(save_path).split("_split")[0]
    save_root = Path(save_path) / f"{base_name}-results"
    orig_root = save_root / "orig"
    marked_root = save_root / "marked"
    orig_root.mkdir(parents=True, exist_ok=True)
    marked_root.mkdir(parents=True, exist_ok=True)

    # 统计结果容器
    stats = defaultdict(int)

    # YOLO 保存带框图到：xxx_split/xxx-predict/
    results = model.predict(predict_data, save=True, conf=conf, project=save_path, name=f"{base_name}-predict")

    # 把 predict_data 当作“原图所在目录”
    predict_data_dir = Path(predict_data)

    for result in tqdm(results, desc="Processing images"):
        result_path = Path(result.path)
        save_dir = Path(result.save_dir) if getattr(result, "save_dir", None) else None

        # ---------------------【关键修正点：区分原图/带框图】--------------------
        # 情况1：result.path 在 save_dir 下面 → 它是“带框图”
        if save_dir is not None and save_dir in result_path.parents:
            marked_img = result_path
            orig_img = predict_data_dir / result_path.name
        else:
            # 情况2：result.path 是原图路径
            orig_img = result_path
            marked_img = save_dir / result_path.name if save_dir is not None else None

        boxes = result.boxes

        # ---------------------【1. 无检测框】--------------------
        if boxes is None or len(boxes) == 0:
            # 原图 no_det
            dst_orig = orig_root / "no_det"
            dst_orig.mkdir(exist_ok=True, parents=True)
            if orig_img.exists():
                shutil.copy2(orig_img, dst_orig / orig_img.name)

            # 带框图 no_det
            dst_marked = marked_root / "no_det"
            dst_marked.mkdir(exist_ok=True, parents=True)
            if marked_img is not None and marked_img.exists():
                shutil.copy2(marked_img, dst_marked / marked_img.name)

            stats["no_det"] += 1
            continue

        # ---------------------【2. 获取最高置信度框】--------------------
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()

        best_idx = int(np.argmax(confs))
        best_conf = float(confs[best_idx])
        pred_names = [model.names[int(i)] for i in cls_ids]

        # ---------------------【3. 低置信度】--------------------
        if best_conf < low_conf_threshold:
            dst_orig = orig_root / "low_confident"
            dst_orig.mkdir(exist_ok=True, parents=True)
            if orig_img.exists():
                shutil.copy2(orig_img, dst_orig / orig_img.name)

            dst_marked = marked_root / "low_confident"
            dst_marked.mkdir(exist_ok=True, parents=True)
            if marked_img is not None and marked_img.exists():
                shutil.copy2(marked_img, dst_marked / marked_img.name)

            stats["low_confident"] += 1
            continue

        # ---------------------【4. 多框优先级分类】--------------------
        selected_cls = next((p for p in priority_list if p in pred_names), "unknown")

        # ---------------------【5. 按类别保存】--------------------
        dst_orig = orig_root / selected_cls
        dst_orig.mkdir(exist_ok=True, parents=True)
        if orig_img.exists():
            shutil.copy2(orig_img, dst_orig / orig_img.name)

        dst_marked = marked_root / selected_cls
        dst_marked.mkdir(exist_ok=True, parents=True)
        if marked_img is not None and marked_img.exists():
            shutil.copy2(marked_img, dst_marked / marked_img.name)

        stats[selected_cls] += 1

    # ------------------【统计结果打印】------------------
    print("\n================ 预测统计结果 ================")
    total = 0
    for key in sorted(stats.keys()):
        print(f"{key:<15}: {stats[key]}")
        total += stats[key]
    print("----------------------------------------------")
    print(f"总图片数           : {total}")
    print("==============================================\n")


# -------------------------------------------------------------------------------------------------------------


def main():
    start_time = setup_logging_start()

    version = "v10.1"

    data = r"/root/autodl-tmp/ultralytics-main/data/WX_class/v7/v7.3/v7.3_split/data.yaml"
    predict_data = r"/root/autodl-tmp/ultralytics-main/data/WX_class/test/11-20/20_ng_91"
    checkpoint = f"/root/autodl-tmp/ultralytics-main/runs/detect/{version}/weights/best.pt"

    delete_ipynb_checkpoints(os.path.dirname(data))

    run_detection_prediction_power(
        checkpoint,
        predict_data,
        conf=0.1,
        low_conf_threshold=0.7,
        save_path=r"/root/autodl-tmp/ultralytics-main/data/WX_class/test/11-20",
    )

    setup_logging_end(start_time)


if __name__ == "__main__":
    main()
