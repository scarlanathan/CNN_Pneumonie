import os
import shutil
import random

def balance_dataset(src_dir, dst_dir, max_count):
    os.makedirs(dst_dir + "/NORMAL", exist_ok=True)
    os.makedirs(dst_dir + "/PNEUMONIA", exist_ok=True)

    for cls in ["NORMAL", "PNEUMONIA"]:
        src_cls_path = os.path.join(src_dir, cls)
        dst_cls_path = os.path.join(dst_dir, cls)

        # 清空目标目录（谨慎操作）
        for f in os.listdir(dst_cls_path):
            os.remove(os.path.join(dst_cls_path, f))

        files = os.listdir(src_cls_path)
        random.shuffle(files)
        files = files[:max_count]

        for f in files:
            shutil.copy(os.path.join(src_cls_path, f), os.path.join(dst_cls_path, f))

# 假设你有备份的原始验证集
balance_dataset("dataset/val", "dataset/val", 1200)  # 保持两类一样
balance_dataset("dataset/test", "dataset/test", 1200)

