import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    f1_score,
    accuracy_score
)
from sklearn.utils import class_weight
import os

# 参数设置
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# 加载数据集
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary", shuffle=True
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/val", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary"
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/test", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary"
)

# 数据加载优化
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 提取训练标签用于计算 class weights
y_train = np.concatenate([y.numpy().flatten() for _, y in train_ds])
# y_train = np.concatenate([y.numpy() for _, y in train_ds])
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# 数据增强
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# 构建 CNN 模型
model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=IMG_SIZE + (3,)),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 回调函数
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]

# 模型训练（使用 class_weights）
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=callbacks,
    class_weight=class_weights
)

# 评估阶段
y_true = []
y_pred = []
y_prob = []

for images, labels in test_ds:
    probs = model.predict(images).flatten()
    preds = (probs > 0.5).astype("int")
    y_true.extend(labels.numpy())
    y_pred.extend(preds)
    y_prob.extend(probs)

# 分类报告
print("\nReport:\n", classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# 准确率
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy：{acc:.4f}")

# 保存图表
def save_plot(filename):
    plt.savefig(filename)
    plt.close()

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
save_plot("confusion_matrix.png")

# F1 分数图
f1_macro = f1_score(y_true, y_pred, average="macro")
f1_weighted = f1_score(y_true, y_pred, average="weighted")
plt.bar(["F1 Macro", "F1 Weighted"], [f1_macro, f1_weighted], color=['orange', 'green'])
plt.title("F1 Score")
plt.ylim(0, 1)
save_plot("f1_score.png")

# AUC-ROC 曲线
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
save_plot("roc_curve.png")

# 准确率柱状图
plt.bar(['Accuracy'], [acc], color='skyblue')
plt.ylim(0, 1)
plt.title("Test Accuracy")
save_plot("accuracy.png")

# 保存最终模型
model.save("final_model.keras")
