import os
import shutil
from pathlib import Path

# 设置路径
input_dir = 'dataset/images/test'  # 包含 PNEUMONIA/ 和 NORMAL/
output_images_dir = 'images/test'
output_labels_dir = 'dataset/labels/test'

# 类别映射
class_map = {
    'PNEUMONIA': 0,
    'NORMAL': 1
}

# 创建输出目录
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# 遍历每个类别文件夹
for class_name, class_id in class_map.items():
    class_dir = Path(input_dir) / class_name
    for img_path in class_dir.glob('*.*'):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        # 复制图像到 images/train
        dst_img = Path(output_images_dir) / img_path.name
        shutil.copy(img_path, dst_img)

        # 生成标签文件
        label_path = Path(output_labels_dir) / f'{img_path.stem}.txt'
        with open(label_path, 'w') as f:
            # 整张图中心 (0.5, 0.5)，宽高 1.0
            f.write(f'{class_id} 0.5 0.5 1.0 1.0\n')

print("✅ 标签生成完成。")
