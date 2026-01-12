from ultralytics import YOLO


def run_detection_prediction(checkpoint, predict_data, save_name="predict"):
    """封装检测任务的预测代码."""
    print("--- 运行检测预测 ---")
    model = YOLO(checkpoint)  
    
    model.predict(predict_data,
                  save=True,
                  line_width=3,
                  project="./runs/detect",
                  name=save_name,
                  # conf=0.1
                 )


def main():
    checkpoint = r"/root/autodl-tmp/ultralytics-main/runs/detect/v7_3/weights/best.pt" 
    predict_data=r"/root/autodl-tmp/ultralytics-main/datas/WX_class/v7/v7.2/v7.2_split/images/test"

    # 检测
    run_detection_prediction(checkpoint, predict_data,save_name="v7.2_predict")  # 预测


if __name__ == "__main__":
    main()
