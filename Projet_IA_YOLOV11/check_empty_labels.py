import os

# 替换为你的路径
image_dir = 'dataset/images/test'
label_dir = 'dataset/labels/test'

# 支持的图像后缀
image_exts = ['.jpg', '.jpeg', '.png']

# 记录异常情况
missing_label_files = []
empty_label_files = []

for filename in os.listdir(image_dir):
    name, ext = os.path.splitext(filename)
    if ext.lower() not in image_exts:
        continue

    label_path = os.path.join(label_dir, name + '.txt')

    # 检查是否有标签文件
    if not os.path.exists(label_path):
        missing_label_files.append(filename)
    else:
        # 检查标签文件是否为空
        if os.path.getsize(label_path) == 0:
            empty_label_files.append(filename)

# 打印报告
print("\n✅ 检查完成：\n")

if missing_label_files:
    print("❌ 缺失标签文件的图像：")
    for f in missing_label_files:
        print("  -", f)
else:
    print("✅ 所有图像都有对应标签文件。")

if empty_label_files:
    print("\n⚠️ 标签内容为空的图像（需要修复）：")
    for f in empty_label_files:
        print("  -", f)
else:
    print("✅ 所有标签文件都有内容。")
