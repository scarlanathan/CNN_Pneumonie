import os
import shutil
import random
from PIL import Image

def balance_class(src_dir, target_count):
    for cls in os.listdir(src_dir):
        cls_path = os.path.join(src_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        current_count = len(images)
        print(f"📁 Class '{cls}': {current_count} images")

        if current_count >= target_count:
            print(f"✅ Already balanced or over: {current_count} >= {target_count}")
            continue

        # 开始复制已有图片来补齐
        needed = target_count - current_count
        print(f"➡️ Need to copy {needed} images to balance class '{cls}'")

        for i in range(needed):
            src_file = random.choice(images)
            src_path = os.path.join(cls_path, src_file)
            dst_file = f"aug_{i}_{src_file}"
            dst_path = os.path.join(cls_path, dst_file)

            # 确保复制文件合法
            try:
                img = Image.open(src_path)
                img.save(dst_path)
            except Exception as e:
                print(f"❌ Failed to copy {src_file}: {e}")
                continue

    print("\n✅ Balancing completed.")

# 修改这里指定你要平衡的子集路径和目标数量
if __name__ == "__main__":
    subset_path = "dataset/val"   # 可修改为 "dataset/train" 或 "dataset/test"
    target_per_class = 500        # 每类要达到的图像数量
    balance_class(subset_path, target_per_class)
