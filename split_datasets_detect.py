import random
import shutil
from pathlib import Path

random.seed(0)


def split_data(
    image_dir,
    label_dir,
    out_dir,
    train_rate=0.8,
    val_rate=0.2,
    test_rate=0.0,
):
    """将 image_dir / label_dir 下的数据划分为 train / val (/ test) 并复制到 out_dir。.

    目标目录结构： out_dir/ images/
        train/
        val/
        test/   # 仅当 test_rate > 0 时创建
    labels/
        train/
        val/
        test/   # 仅当 test_rate > 0 时创建
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    out_dir = Path(out_dir)

    # 一些常见图片后缀
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    # 收集所有有对应 label 的图片
    pairs = []
    for img_path in image_dir.iterdir():
        if img_path.suffix.lower() not in image_exts:
            continue
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            # 如果你想知道哪些图片没有标签，可以打印出来
            # print(f"⚠️ 找不到标签文件: {label_path}")
            pass

    total = len(pairs)
    if total == 0:
        print("⚠️ 没有找到任何成对的 图片+标签，检查路径或文件名是否对应。")
        return

    print(f"共找到 {total} 对 图片+标签")

    random.shuffle(pairs)

    # ================== 按比例切分 ==================
    if test_rate <= 0:
        # 默认：只划分 train / val
        train_end = int(total * train_rate)
        train_pairs = pairs[:train_end]
        val_pairs = pairs[train_end:]
        test_pairs = []
    else:
        # train / val / test 都划分
        train_end = int(total * train_rate)
        val_end = train_end + int(total * val_rate)
        train_pairs = pairs[:train_end]
        val_pairs = pairs[train_end:val_end]
        test_pairs = pairs[val_end:]

    print(f"train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}")

    # ================== 辅助函数：复制文件 ==================
    def copy_split(split_name, split_pairs):
        if not split_pairs:
            return
        img_dst_dir = out_dir / "images" / split_name
        lbl_dst_dir = out_dir / "labels" / split_name
        img_dst_dir.mkdir(parents=True, exist_ok=True)
        lbl_dst_dir.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in split_pairs:
            shutil.copy2(img_path, img_dst_dir / img_path.name)
            shutil.copy2(lbl_path, lbl_dst_dir / lbl_path.name)

    # ================== 执行复制 ==================
    copy_split("train", train_pairs)
    copy_split("val", val_pairs)
    if test_pairs:
        copy_split("test", test_pairs)


if __name__ == "__main__":
    base_path = r"/root/autodl-tmp/ultralytics-main/data/WX_class/v10/v10.5"
    new_file_path = base_path + "/v10.5_split"
    file_path = base_path + "/images"
    txt_path = base_path + "/labels"

    # 默认：只有 train/val
    split_data(file_path, txt_path, new_file_path, train_rate=0.7, val_rate=0.2, test_rate=0.1)
