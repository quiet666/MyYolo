import os
import shutil

def delete_ipynb_checkpoints(root_dir):
    # 遍历文件夹中的所有文件和子文件夹
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # 如果目录名称是 .ipynb_checkpoints，删除该目录及其中的所有文件
        if os.path.basename(dirpath) == '.ipynb_checkpoints':
            print(f"Deleting: {dirpath}")
            shutil.rmtree(dirpath)


# 设置要操作的根目录
root_directory = '/root/autodl-tmp/ultralytics-main/datas/WX_class'
delete_ipynb_checkpoints(root_directory)
