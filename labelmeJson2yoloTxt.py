import json
from pathlib import Path  # ✅ 跨平台路径

# 定义标签映射
label_map = {
    "py": 0,
    "cl": 1,
    "undefined": 2,
}


def convert_labelme_to_yolo(json_path: Path, output_dir: Path):
    """将单个 Labelme JSON 转为 YOLO txt"""

    with json_path.open('r', encoding='utf-8') as f:
        labelme_data = json.load(f)

    image_width = labelme_data['imageWidth']
    image_height = labelme_data['imageHeight']

    yolo_annotations = []

    for shape in labelme_data['shapes']:
        label = shape.get('label')
        if label not in label_map:
            continue  # 忽略未定义的标签

        class_id = label_map[label]
        points = shape['points']

        # ✅ 不再依赖画框顺序，统一通过 min/max 计算外接矩形
        # 无论是 rectangle 还是 polygon，points 都是一组点
        if shape['shape_type'] in ['rectangle', 'polygon']:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
        else:
            # 其他类型暂不处理
            continue

        # 计算 YOLO 格式（归一化）
        x_center = ((x1 + x2) / 2.0) / image_width
        y_center = ((y1 + y2) / 2.0) / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height

        # 保留一定小数位，避免太长
        yolo_annotations.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    # 输出目录确保存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 输出 txt 文件名与 json 同名
    output_file = output_dir / f"{json_path.stem}.txt"
    with output_file.open('w', encoding='utf-8') as f:
        f.write('\n'.join(yolo_annotations))


def process_folder(input_folder, output_folder):
    """批量处理文件夹中的所有 JSON"""

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for json_path in input_folder.glob("*.json"):
        convert_labelme_to_yolo(json_path, output_folder)


if __name__ == "__main__":
    # 示例使用（这里用原来的路径，换成自己的即可）
    # Windows 和 Linux 都可以用这种写法（注意路径是否存在）
    # input_folder = r"E:\work\projects\defect_detection\异常分类\PASS前300标注"
    input_folder = r"E:\work\poc_dataset\classifier\3_py_annotation\3_py_annotation\test_json"
    output_folder = r"E:\work\poc_dataset\classifier\3_py_annotation\3_py_annotation\txt2"

    process_folder(input_folder, output_folder)

    # 列出输出文件夹中的文件以确认
    out_files = list(Path(output_folder).glob("*.txt"))
    print(f"共生成 {len(out_files)} 个标注文件：")
    for p in out_files[:10]:
        print(" -", p.name)
