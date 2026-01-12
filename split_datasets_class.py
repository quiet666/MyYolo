from ultralytics.data.split import autosplit

from ultralytics.data.split import split_classify_dataset


data_path = r"/root/autodl-tmp/ultralytics-main/datas/WX_class/v8/v8-binary" 
# autosplit(path="data_path", weights=(0.8, 0.15, 0.05), annotated_only=False)

split_classify_dataset(data_path, 0.80)
