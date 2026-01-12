import os
import cv2
import numpy as np

def resize_images(input_dir, output_dir, max_size=1024):
    """
    调整图像大小并保持纵横比
    
    参数:
    input_dir (str): 输入图像目录
    output_dir (str): 输出图像目录
    max_size (int): 最大边的像素数
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图像文件扩展名
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.webp')
    
    # 遍历输入目录中的所有图像文件
    for filename in os.listdir(input_dir):
        # 检查文件扩展名
        if filename.lower().endswith(image_extensions):
            # 构建完整的输入和输出路径
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                # 读取图像
                img = cv2.imread(input_path)
                
                # 检查图像是否成功读取
                if img is None:
                    print(f"警告：无法读取图像 {filename}")
                    continue
                
                # 获取原始图像尺寸
                height, width = img.shape[:2]
                
                # 计算缩放比例
                if height > width:
                    scale = max_size / height
                else:
                    scale = max_size / width
                
                # 计算新的尺寸
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                # 确保尺寸至少为1
                new_width = max(new_width, 1)
                new_height = max(new_height, 1)
                
                # 调整图像大小
                resized_img = cv2.resize(
                    img, 
                    (new_width, new_height), 
                    interpolation=cv2.INTER_AREA
                )
                
                # 保存调整大小后的图像
                cv2.imwrite(output_path, resized_img)
                
                print(f"压缩 {filename}: {width}x{height} -> {new_width}x{new_height}")
            
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")

def adjust_yolo_labels(input_label_dir, output_label_dir, original_size, new_size):
    """
    调整 YOLO 格式的标签文件
    
    参数:
    input_label_dir (str): 输入标签目录
    output_label_dir (str): 输出标签目录
    original_size (tuple): 原始图像尺寸 (width, height)
    new_size (tuple): 新图像尺寸 (width, height)
    """
    # 确保输出目录存在
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 计算缩放比例
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]
    
    # 遍历标签文件
    for label_file in os.listdir(input_label_dir):
        if label_file.endswith('.txt'):
            input_path = os.path.join(input_label_dir, label_file)
            output_path = os.path.join(output_label_dir, label_file)
            
            try:
                # 读取标签文件
                with open(input_path, 'r') as f:
                    lines = f.readlines()
                
                adjusted_lines = []
                for line in lines:
                    parts = line.strip().split()
                    
                    # 确保有足够的部分
                    if len(parts) < 5:
                        print(f"警告：{label_file} 中的标签格式不正确")
                        continue
                    
                    # 调整中心坐标和宽高
                    try:
                        x_center = float(parts[1]) * scale_x
                        y_center = float(parts[2]) * scale_y
                        width = float(parts[3]) * scale_x
                        height = float(parts[4]) * scale_y
                        
                        # 确保坐标在 0-1 范围内
                        x_center = max(0, min(x_center, 1))
                        y_center = max(0, min(y_center, 1))
                        width = max(0, min(width, 1))
                        height = max(0, min(height, 1))
                        
                        adjusted_line = f"{parts[0]} {x_center} {y_center} {width} {height}\n"
                        adjusted_lines.append(adjusted_line)
                    
                    except ValueError:
                        print(f"警告：{label_file} 中的坐标转换失败")
                
                # 写入调整后的标签
                with open(output_path, 'w') as f:
                    f.writelines(adjusted_lines)
            
            except Exception as e:
                print(f"处理 {label_file} 时出错: {e}")

def process_dataset(input_image_dir, input_label_dir, output_image_dir, output_label_dir, max_size=1024):
    """
    处理整个数据集
    
    参数:
    input_image_dir (str): 输入图像目录
    input_label_dir (str): 输入标签目录
    output_image_dir (str): 输出图像目录
    output_label_dir (str): 输出标签目录
    max_size (int): 最大边的像素数
    """
    # 确保输出目录存在
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 遍历图像文件
    for filename in os.listdir(input_image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.webp')):
            # 读取原始图像
            input_image_path = os.path.join(input_image_dir, filename)
            img = cv2.imread(input_image_path)
            
            # 获取原始图像尺寸
            original_height, original_width = img.shape[:2]
            
            # 计算缩放比例
            if original_height > original_width:
                scale = max_size / original_height
            else:
                scale = max_size / original_width
            
            # 计算新的尺寸
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # 调整图像大小
            resized_img = cv2.resize(
                img, 
                (new_width, new_height), 
                interpolation=cv2.INTER_AREA
            )
            
            # 保存调整大小后的图像
            output_image_path = os.path.join(output_image_dir, filename)
            cv2.imwrite(output_image_path, resized_img)
            
            # 处理对应的标签文件
            label_filename = filename.rsplit('.', 1)[0] + '.txt'
            input_label_path = os.path.join(input_label_dir, label_filename)
            output_label_path = os.path.join(output_label_dir, label_filename)
            
            if os.path.exists(input_label_path):
                # 调整标签
                adjust_yolo_labels(
                    os.path.dirname(input_label_path),
                    os.path.dirname(output_label_path),
                    (original_width, original_height),
                    (new_width, new_height)
                )
            
            print(f"处理 {filename}: {original_width}x{original_height} -> {new_width}x{new_height}")

# 使用示例
if __name__ == "__main__":
    # 训练集
    print("处理训练集数据")
    # path = "/root/autodl-tmp/data/defection/train"
    # new_path = "/root/autodl-tmp/data/defection/train_resize"
    
    path = "/root/autodl-tmp/ultralytics-main/datas/Black_png/"
    new_path = "/root/autodl-tmp/ultralytics-main/datas/Black_png"

    train_path = f"{path}/train"
    train_save = f"{path}/train_resize"

    val_path = f"{path}/val"
    val_save = f"{path}/val_resize"

    
    process_dataset(
        input_image_dir = f"{train_path}/images",
        input_label_dir = f"{train_path}/labels",
        output_image_dir= f"{train_save}/images",
        output_label_dir= f"{train_save}/labels",
        max_size=1024
    )
    
    # 验证集
    print("处理验证集数据")
    path = "/root/autodl-tmp/data/defection/val"
    new_path = "/root/autodl-tmp/data/defection/val_resize"
    process_dataset(
        input_image_dir = f"{val_path}/images",
        input_label_dir = f"{val_path}/labels",
        output_image_dir= f"{val_save}/images",
        output_label_dir= f"{val_save}/labels",
        max_size=1024
    )
    print("运行结束")