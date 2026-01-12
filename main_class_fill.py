
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
import torchvision.transforms as T
from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator
from pathlib import Path
import datetime
import shutil

class CustomizedDataset(ClassificationDataset):
    """A customized dataset class for image classification with enhanced data augmentation transforms."""

    def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
        """Initialize a customized classification dataset with enhanced data augmentation transforms."""
        super().__init__(root, args, augment, prefix)
        print("调用自己写的 CustomizedDataset")

        print(f"【args.hsv_v】={args.hsv_v}")
        print(f"【args.hsv_s】={args.hsv_s}")
        print(f"【args.hsv_h】={args.hsv_h}")
        
        # Add your custom training transforms here
        train_transforms = T.Compose(
            [
                T.Resize((args.imgsz, args.imgsz)),
                T.RandomHorizontalFlip(p=args.fliplr),
                T.RandomVerticalFlip(p=args.flipud),
                T.RandAugment(interpolation=T.InterpolationMode.BILINEAR),
                # 使用更安全的ColorJitter配置，避免hue参数可能引起的uint8溢出问题
                T.ColorJitter(brightness=args.hsv_v, contrast=args.hsv_v, saturation=args.hsv_s),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(0), std=torch.tensor(1)),
                T.RandomErasing(p=args.erasing, inplace=True),
            ]
        )

        # Add your custom validation transforms here
        val_transforms = T.Compose(
            [
                T.Resize((args.imgsz, args.imgsz)),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(0), std=torch.tensor(1)),
            ]
        )
        self.torch_transforms = train_transforms if augment else val_transforms


class CustomizedTrainer(ClassificationTrainer):
    """A customized trainer class for YOLO classification models with enhanced dataset handling."""

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        print("调用自己写的 CustomizedTrainer")
        """Build a customized dataset for classification training and the validation during training."""
        return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)


class CustomizedValidator(ClassificationValidator):
    """A customized validator class for YOLO classification models with enhanced dataset handling."""

    def build_dataset(self, img_path: str, mode: str = "train"):
        print("调用自己写的 CustomizedValidator")
        """Build a customized dataset for classification standalone validation."""
        return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=self.args.split)




def delete_ipynb_checkpoints(root_dir):
    # 遍历文件夹中的所有文件和子文件夹
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # 如果目录名称是 .ipynb_checkpoints，删除该目录及其中的所有文件
        if os.path.basename(dirpath) == '.ipynb_checkpoints':
            print(f"【Deleting】: {dirpath}")
            shutil.rmtree(dirpath)
        if os.path.basename(dirpath) == '.cache':
            print(f"【Deleting】: {dirpath}")
            shutil.rmtree(dirpath)

def setup_logging_start():
    """设置并打印程序开始日志."""
    start_time = datetime.datetime.now()
    print(f"程序开始运行时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    return start_time


def setup_logging_end(start_time):
    """打印程序结束日志和总耗时."""
    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print("-" * 50)
    print(f"训练完成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总训练耗时: {total_time}")



#-------------------------------------------------------------------------------------------------------------------
### 【分类】
def run_classification_training(model, data):
    """运行分类训练代码"""
    print("--- 运行分类训练 ---")
    # Load a model
    # model = YOLO("yolo11n-cls.yaml")                          # build a new model from YAML
    # model = YOLO("yolo11n-cls.pt")                            # load a pretrained model (recommended for training)
    model = YOLO(f"/root/autodl-tmp/ultralytics-main/ultralytics/cfg/models/11/{model}.yaml").load(f"{model}.pt")   # build from YAML and transfer weights
    #
    model.train(data = data, 
                trainer=CustomizedTrainer,
                epochs=200,      
                patience=50,
                batch=32,
                imgsz=640,
                val=True,
                workers=8,
                # project="./ultralytics/runs/detect", 
                name="v6.4_binary",    
                # cache=True,
                # box=7.5,   
                # cls=0.5,    
                # dfl=1.5,  
               )
    print("分类训练代码已执行（当前注释状态）")
    print("-" * 50)

def run_classification_validation(checkpoint,data,split="val"):
    """运行分类预测代码"""
    print("--- 运行分类验证 ---")
    print(f"checkpoint 路径为：{checkpoint}")
    model = YOLO(checkpoint)
    model.val(data=data, 
              # validator=CustomizedValidator,
            #   batch=64,
              imgsz=640, 
              split=split,
              verbose=True,
              save_txt=True,
              save_conf=True
             )
    print("检测验证代码已执行（当前注释状态）")
    print("-" * 50)


def run_classification_prediction(checkpoint, data, defect_class, predict_dir_name="predict_v3", confidence_threshold=0.0):
    """封装分类任务的预测代码."""
    print("--- 运行分类预测 ---")
    model = YOLO(checkpoint)  
    base_predict_path = f"./runs/classify/{predict_dir_name}"
    
    results = model.predict(f"{data}/val/{defect_class}",
                  save=True,
                  name=f"{predict_dir_name}/{defect_class}"
                 )  # predict on an image
    
    # 获取真实标签索引
    true_label = int(defect_class.split("_")[0])
    
    # 创建错误预测保存目录的根目录
    incorrect_root_dir = Path(f"{base_predict_path}/{defect_class}/high_incorrect")
    incorrect_root_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建低置信度保存目录
    low_confidence_dir = Path(f"{base_predict_path}/{defect_class}/low_confidence")
    low_confidence_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建低置信度且分类错误保存目录的根目录
    low_incorrect_root_dir = Path(f"{base_predict_path}/{defect_class}/low_incorrect")
    low_incorrect_root_dir.mkdir(parents=True, exist_ok=True)
    
    # 遍历预测结果，将错误分类的图片移动到对应错误类别的目录
    error_count = 0
    low_confidence_count = 0
    low_incorrect_count = 0
    
    for result in results:
        # 获取预测概率和预测标签
        probs = result.probs
        pred_label = probs.top1
        pred_confidence = probs.top1conf
        
        # 获取图片路径
        img_path = Path(result.path)
        # 原始保存路径
        orig_save_dir = Path(f"{base_predict_path}/{defect_class}")
        orig_save_path = orig_save_dir / img_path.name
        
        # 检查图片是否已保存
        if orig_save_path.exists():
            # 如果置信度低于阈值
            if confidence_threshold > 0 and pred_confidence < confidence_threshold:
                # 如果同时分类错误
                if pred_label != true_label:
                    # 根据预测标签获取错误类别名称
                    pred_class_name = model.names[pred_label]
                    # 创建对应错误类别的目录
                    low_incorrect_class_dir = low_incorrect_root_dir / pred_class_name
                    low_incorrect_class_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 移动图片到对应错误类别的目录
                    save_path = low_incorrect_class_dir / img_path.name
                    shutil.move(str(orig_save_path), str(save_path))
                    low_incorrect_count += 1
                else:
                    # 只是置信度低
                    save_path = low_confidence_dir / img_path.name
                    shutil.move(str(orig_save_path), str(save_path))
                    low_confidence_count += 1
            # 如果预测错误且置信度高于阈值
            elif pred_label != true_label:
                # 根据预测标签获取错误类别名称
                pred_class_name = model.names[pred_label]
                # 创建对应错误类别的目录
                incorrect_class_dir = incorrect_root_dir / pred_class_name
                incorrect_class_dir.mkdir(parents=True, exist_ok=True)
                
                # 移动图片到对应错误类别的目录
                save_path = incorrect_class_dir / img_path.name
                shutil.move(str(orig_save_path), str(save_path))
                error_count += 1
    
    print("-" * 50)
    print(f"总共处理了 {len(results)} 张图片")
    print(f"预测错误的图片有 {error_count} 张，已保存至: {base_predict_path}/{defect_class}/incorrect/[错误类别名称]/")
    if confidence_threshold > 0:
        print(f"低置信度的图片有 {low_confidence_count} 张，已保存至: {base_predict_path}/{defect_class}/low_confidence/")
        print(f"低置信度且分类错误的图片有 {low_incorrect_count} 张，已保存至: {base_predict_path}/{defect_class}/low_incorrect/[错误类别名称]/")
    print(f"预测结果已保存至: {base_predict_path}/{defect_class}/")
    print("分类预测代码已执行")
    print("-" * 50)
#-----------------------------------------------------------------------------------------------------------------------------
def main():
    ## 分类
    # 设置要操作的根目录
    model = "yolo11x-cls"
    data = r"/root/autodl-tmp/ultralytics-main/datas/WX_class/v6/v6.3-multi_split"
    checkpoint = r"/root/autodl-tmp/ultralytics-main/runs/classify/v6.3_multi/weights/best.pt" #train15_v4_resize
    delete_ipynb_checkpoints(data)
    
    # run_classification_training(model, data)    
    # run_classification_validation(checkpoint, data)
    run_classification_prediction(checkpoint, 
                                  data, 
                                  defect_class="4_undefined", #0_normal  1_cl  2_fz  3_py 4_undefined
                                  predict_dir_name="predict_v6-multi", 
                                  confidence_threshold=0.9) 


if __name__ == "__main__":
    start_time = setup_logging_start()
    main()
    setup_logging_end(start_time)


# nohup python main_class.py > logs/$(date +%Y%m%d-%H%M%S)-classes_yolo11x_v4.log 2>&1 &