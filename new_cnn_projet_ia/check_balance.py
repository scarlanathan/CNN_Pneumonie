import os

def check_class_distribution(dataset_dir):
    print(f"\n📂 Checking dataset: {dataset_dir}")
    total = 0
    for cls in os.listdir(dataset_dir):
        cls_path = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        count = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  ➤ Class '{cls}': {count} images")
        total += count
    print(f"  🧮 Total images: {total}\n")

# 指定要检查的目录
for split in ["train", "val", "test"]:
    path = os.path.join("dataset", split)
    if os.path.exists(path):
        check_class_distribution(path)
    else:
        print(f"❌ Missing directory: {path}")
