import os
import shutil
import random
from pathlib import Path

# 设置参数
SOURCE_DIR = "dataset/train"
VAL_DIR = "dataset/val"
TEST_DIR = "dataset/test"
VAL_SPLIT = 0.15   # 15% 用作验证集
TEST_SPLIT = 0.15  # 15% 用作测试集

def prepare_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def split_data():
    for cls in os.listdir(SOURCE_DIR):
        src_class_dir = os.path.join(SOURCE_DIR, cls)
        if not os.path.isdir(src_class_dir):
            continue

        images = [f for f in os.listdir(src_class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        total = len(images)
        val_count = int(total * VAL_SPLIT)
        test_count = int(total * TEST_SPLIT)

        val_images = images[:val_count]
        test_images = images[val_count:val_count + test_count]

        # 复制到 val 和 test
        for img in val_images:
            src = os.path.join(src_class_dir, img)
            dst = os.path.join(VAL_DIR, cls)
            os.makedirs(dst, exist_ok=True)
            shutil.copy(src, os.path.join(dst, img))

        for img in test_images:
            src = os.path.join(src_class_dir, img)
            dst = os.path.join(TEST_DIR, cls)
            os.makedirs(dst, exist_ok=True)
            shutil.copy(src, os.path.join(dst, img))

        print(f"✅ {cls}: {total} total → {len(val_images)} val, {len(test_images)} test")

if __name__ == "__main__":
    prepare_dir(VAL_DIR)
    prepare_dir(TEST_DIR)
    split_data()
    print("\n📦 数据集划分完成！")
