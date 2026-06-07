"""
学习整理版说明：

这个文件复制自 Keras 官方示例：
https://keras.io/examples/vision/3D_image_classification/

建议学习时修改这个文件，不要直接修改 ../01_官方原始源码/3D_image_classification_official.py。
中文分段讲解见：代码分段讲解.md

=================================================================
代码功能概述：
这个代码用 3D CNN 判断胸部 CT 是否存在病毒性肺炎相关异常。
输入：CT 体数据（很多张切片堆起来的三维数组）
任务类型：二分类（0=正常CT，1=异常CT）
=================================================================
"""

"""
Title: 3D image classification from CT scans
Author: [Hasib Zunair](https://twitter.com/hasibzunair)
Date created: 2020/09/23
Last modified: 2024/01/11
Description: Train a 3D convolutional neural network to predict presence of pneumonia.
Accelerator: GPU
"""

"""
## Introduction

This example will show the steps needed to build a 3D convolutional neural network (CNN)
to predict the presence of viral pneumonia in computer tomography (CT) scans. 2D CNNs are
commonly used to process RGB images (3 channels). A 3D CNN is simply the 3D
equivalent: it takes as input a 3D volume or a sequence of 2D frames (e.g. slices in a CT scan),
3D CNNs are a powerful model for learning representations for volumetric data.

## References

- [A survey on Deep Learning Advances on Different 3D DataRepresentations](https://arxiv.org/abs/1808.01462)
- [VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition](https://www.ri.cmu.edu/pub_files/2015/9/voxnet_maturana_scherer_iros15.pdf)
- [FusionNet: 3D Object Classification Using MultipleData Representations](https://arxiv.org/abs/1607.05695)
- [Uniformizing Techniques to Process CT scans with 3D CNNs for Tuberculosis Prediction](https://arxiv.org/abs/2007.13224)
"""
"""
## Setup（导入依赖）
"""

# ==================== 模块来源说明 ====================
# os：Python标准库，用于文件路径操作
# zipfile：Python标准库，用于解压文件
# numpy：第三方库，用于处理数组（pip install numpy）
# tensorflow：第三方库，用于构建数据管道（pip install tensorflow）
# keras：第三方库，用于搭建和训练神经网络（pip install keras）
# =====================================================

import os
import zipfile
import numpy as np
import tensorflow as tf  # for data preprocessing

import keras
from keras import layers

"""
## Downloading the MosMedData: Chest CT Scans with COVID-19 Related Findings
## （下载 CT 数据）

In this example, we use a subset of the
[MosMedData: Chest CT Scans with COVID-19 Related Findings](https://www.medrxiv.org/content/10.1101/2020.05.20.20100362v1).
This dataset consists of lung CT scans with COVID-19 related findings, as well as without such findings.

We will be using the associated radiological findings of the CT scans as labels to build
a classifier to predict presence of viral pneumonia.
Hence, the task is a binary classification problem.

讲解说明：
源码会下载两个压缩包：
- CT-0.zip：正常肺部 CT
- CT-23.zip：有异常表现的 CT
下载后解压到 MosMedData 文件夹。
注意：数据体积比较大，第一次运行会花时间，并且需要网络。
"""

# Download url of normal CT scans.
# 模块来源：os.path.join - Python标准库os模块
url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-0.zip"
filename = os.path.join(os.getcwd(), "CT-0.zip")
# 模块来源：keras.utils.get_file - Keras工具函数，用于下载文件
keras.utils.get_file(filename, url)

# Download url of abnormal CT scans.
url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-23.zip"
filename = os.path.join(os.getcwd(), "CT-23.zip")
keras.utils.get_file(filename, url)

# Make a directory to store the data.
# 模块来源：os.makedirs - Python标准库os模块
os.makedirs("MosMedData")

# Unzip data in the newly created directory.
# 模块来源：zipfile.ZipFile - Python标准库zipfile模块
with zipfile.ZipFile("CT-0.zip", "r") as z_fp:
    z_fp.extractall("./MosMedData/")

with zipfile.ZipFile("CT-23.zip", "r") as z_fp:
    z_fp.extractall("./MosMedData/")

"""
## Loading data and preprocessing（读取 .nii 医学影像）

The files are provided in Nifti format with the extension .nii. To read the
scans, we use the `nibabel` package.
You can install the package via `pip install nibabel`. CT scans store raw voxel
intensity in Hounsfield units (HU). They range from -1024 to above 2000 in this dataset.
Above 400 are bones with different radiointensity, so this is used as a higher bound. A threshold
between -1000 and 400 is commonly used to normalize CT scans.

To process the data, we do the following:

* We first rotate the volumes by 90 degrees, so the orientation is fixed
* We scale the HU values to be between 0 and 1.
* We resize width, height and depth.

Here we define several helper functions to process the data. These functions
will be used when building training and validation datasets.

讲解说明：
医学 CT 常见格式不是 .jpg 或 .png，而是 .nii、.nii.gz、.dcm 等。这个示例用的是 .nii。
需要安装 nibabel 包：pip install nibabel
"""

# ==================== 模块来源说明 ====================
# nibabel：第三方库，用于读取医学影像 .nii 文件（pip install nibabel）
# scipy.ndimage：第三方库，用于旋转、缩放三维图像（pip install scipy）
# =====================================================

import nibabel as nib

from scipy import ndimage


def read_nifti_file(filepath):
    """Read and load volume（读取 .nii 医学影像文件）"""
    # 模块来源：nib.load - nibabel库函数，用于加载.nii文件
    # Read file
    scan = nib.load(filepath)
    # 模块来源：scan.get_fdata() - nibabel对象方法，获取原始数据
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """Normalize the volume（CT HU 值归一化）"""
    # 讲解说明：
    # CT 图像的像素值通常是 HU 值，不是普通图片的 0 到 255。
    # 代码把 HU 限制在：最小 -1000，最大 400
    # 然后缩放到 0 到 1。这是医学图像预处理中很重要的一步。
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    # 模块来源：volume.astype() - NumPy数组方法，用于转换数据类型
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize across z-axis（调整 CT 三维尺寸）"""
    # 讲解说明：
    # 模型需要固定输入尺寸，所以代码把每个 CT 统一调整成：
    # 128 x 128 x 64（宽 x 高 x 深度切片数）
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    # 模块来源：img.shape - NumPy数组属性，获取数组维度
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    # 模块来源：ndimage.rotate - scipy.ndimage模块，用于旋转图像
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    # 模块来源：ndimage.zoom - scipy.ndimage模块，用于缩放图像
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path):
    """Read and resize volume（读取并调整CT尺寸）"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume


"""
Let's read the paths of the CT scans from the class directories.
（读取CT扫描文件路径）
"""

# 讲解说明：
# 代码把正常和异常 CT 合并，然后分成：
# - 训练集：140 个样本
# - 验证集：60 个样本
# 标签规则：abnormal_labels = 1（异常），normal_labels = 0（正常）

# Folder "CT-0" consist of CT scans having normal lung tissue,
# no CT-signs of viral pneumonia.
# 模块来源：os.path.join - Python标准库os模块，用于路径拼接
# 模块来源：os.listdir - Python标准库os模块，用于列出目录内容
normal_scan_paths = [
    os.path.join(os.getcwd(), "MosMedData/CT-0", x)
    for x in os.listdir("MosMedData/CT-0")
]
# Folder "CT-23" consist of CT scans having several ground-glass opacifications,
# involvement of lung parenchyma.
abnormal_scan_paths = [
    os.path.join(os.getcwd(), "MosMedData/CT-23", x)
    for x in os.listdir("MosMedData/CT-23")
]

print("CT scans with normal lung tissue: " + str(len(normal_scan_paths)))
print("CT scans with abnormal lung tissue: " + str(len(abnormal_scan_paths)))


"""
## Build train and validation datasets（组成训练集和验证集）
Read the scans from the class directories and assign labels. Downsample the scans to have
shape of 128x128x64. Rescale the raw HU values to the range 0 to 1.
Lastly, split the dataset into train and validation subsets.
"""

# Read and process the scans.
# Each scan is resized across height, width, and depth and rescaled.
# 模块来源：np.array - NumPy库函数，用于创建数组
abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])
normal_scans = np.array([process_scan(path) for path in normal_scan_paths])

# For the CT scans having presence of viral pneumonia
# assign 1, for the normal ones assign 0.
abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
normal_labels = np.array([0 for _ in range(len(normal_scans))])

# Split data in the ratio 70-30 for training and validation.
# 模块来源：np.concatenate - NumPy库函数，用于数组拼接
x_train = np.concatenate((abnormal_scans[:70], normal_scans[:70]), axis=0)
y_train = np.concatenate((abnormal_labels[:70], normal_labels[:70]), axis=0)
x_val = np.concatenate((abnormal_scans[70:], normal_scans[70:]), axis=0)
y_val = np.concatenate((abnormal_labels[70:], normal_labels[70:]), axis=0)
print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)

"""
## Data augmentation（数据增强）

The CT scans also augmented by rotating at random angles during training. Since
the data is stored in rank-3 tensors of shape `(samples, height, width, depth)`,
we add a dimension of size 1 at axis 4 to be able to perform 3D convolutions on
the data. The new shape is thus `(samples, height, width, depth, 1)`. There are
different kinds of preprocessing and augmentation techniques out there,
this example shows a few simple ones to get started.

讲解说明：
训练时会随机旋转 CT 体数据，让模型见到更多变化，减少过拟合。
"""

# ==================== 模块来源说明 ====================
# random：Python标准库，用于随机选择（用于数据增强的随机角度选择）
# scipy.ndimage：第三方库，用于旋转三维图像（已导入）
# =====================================================

import random

from scipy import ndimage


def rotate(volume):
    """Rotate the volume by a few degrees（数据增强：随机旋转CT体数据）"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        # 模块来源：random.choice - Python标准库random模块，用于随机选择
        angle = random.choice(angles)
        # rotate volume
        # 模块来源：ndimage.rotate - scipy.ndimage模块，用于旋转图像
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    # 模块来源：tf.numpy_function - TensorFlow函数，用于将NumPy函数包装为TensorFlow操作
    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel.（训练数据预处理）"""
    # Rotate volume
    volume = rotate(volume)
    # 模块来源：tf.expand_dims - TensorFlow函数，用于增加维度
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel.（验证数据预处理）"""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


"""
While defining the train and validation data loader, the training data is passed through
and augmentation function which randomly rotates volume at different angles. Note that both
training and validation data are already rescaled to have values between 0 and 1.
（定义数据加载器，训练数据会进行数据增强）
"""

# Define data loaders.
# 模块来源：tf.data.Dataset.from_tensor_slices - TensorFlow数据管道函数
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 2
# Augment the on the fly during training.
# 讲解说明：
# - shuffle：打乱数据顺序
# - map：应用预处理函数
# - batch：分批处理
# - prefetch：预加载数据，提高训练效率
train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)

"""
Visualize an augmented CT scan.（可视化增强后的CT扫描）
"""

# ==================== 模块来源说明 ====================
# matplotlib：第三方库，用于画图和显示CT切片（pip install matplotlib）
# =====================================================

import matplotlib.pyplot as plt

# 模块来源：train_dataset.take(1) - TensorFlow数据集方法，获取一个批次数据
data = train_dataset.take(1)
images, labels = list(data)[0]
# 模块来源：images.numpy() - TensorFlow张量方法，转换为NumPy数组
images = images.numpy()
image = images[0]
print("Dimension of the CT scan is:", image.shape)
# 模块来源：np.squeeze - NumPy函数，用于去除维度为1的轴
# 模块来源：plt.imshow - matplotlib.pyplot函数，用于显示图像
plt.imshow(np.squeeze(image[:, :, 30]), cmap="gray")


"""
Since a CT scan has many slices, let's visualize a montage of the slices.
（由于CT扫描有很多切片，让我们可视化切片的蒙太奇）
"""


def plot_slices(num_rows, num_columns, width, height, data):
    """Plot a montage of 20 CT slices（绘制CT切片蒙太奇）"""
    # 模块来源：np.rot90 - NumPy函数，用于数组旋转90度
    data = np.rot90(np.array(data))
    # 模块来源：np.transpose - NumPy函数，用于数组转置
    data = np.transpose(data)
    # 模块来源：np.reshape - NumPy函数，用于数组重塑
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    # 模块来源：plt.subplots - matplotlib.pyplot函数，用于创建子图
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            # 模块来源：axarr[i, j].imshow - matplotlib Axes对象方法，用于显示图像
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    # 模块来源：plt.subplots_adjust - matplotlib.pyplot函数，用于调整子图间距
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


# Visualize montage of slices.
# 4 rows and 10 columns for 100 slices of the CT scan.
plot_slices(4, 10, 128, 128, image[:, :, :40])

"""
## Define a 3D convolutional neural network（构建 3D CNN）

To make the model easier to understand, we structure it into blocks.
The architecture of the 3D CNN used in this example
is based on [this paper](https://arxiv.org/abs/2007.13224).

讲解说明：
普通 CNN 用 Conv2D 处理图片；CT 体数据用 Conv3D，因为它同时看宽、高、切片深度三个方向。
模型主要结构：
- 多层 Conv3D
- 多层 MaxPool3D
- GlobalAveragePooling3D
- 全连接层
- 最后输出一个概率（sigmoid激活函数用于二分类）
"""


def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model.（构建3D CNN模型）"""

    # 模块来源：keras.Input - Keras函数，用于定义模型输入
    inputs = keras.Input((width, height, depth, 1))

    # 模块来源：layers.Conv3D - Keras层，3D卷积层
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    # 模块来源：layers.MaxPool3D - Keras层，3D最大池化层
    x = layers.MaxPool3D(pool_size=2)(x)
    # 模块来源：layers.BatchNormalization - Keras层，批归一化层
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    # 模块来源：layers.GlobalAveragePooling3D - Keras层，3D全局平均池化层
    x = layers.GlobalAveragePooling3D()(x)
    # 模块来源：layers.Dense - Keras层，全连接层
    x = layers.Dense(units=512, activation="relu")(x)
    # 模块来源：layers.Dropout - Keras层，Dropout层，用于防止过拟合
    x = layers.Dropout(0.3)(x)

    # 输出层：sigmoid激活函数用于二分类
    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    # 模块来源：keras.Model - Keras函数，用于创建模型
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
model = get_model(width=128, height=128, depth=64)
# 模块来源：model.summary() - Keras模型方法，打印模型摘要
model.summary()

"""
## Train model（训练模型）
"""

# Compile model.
# 讲解说明：
# 使用：
# - 损失函数：binary_crossentropy（二分类交叉熵）
# - 优化器：Adam
# - 指标：accuracy（准确率）

# 模块来源：initial_learning_rate - 自定义变量
initial_learning_rate = 0.0001
# 模块来源：keras.optimizers.schedules.ExponentialDecay - Keras学习率调度器
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
# 模块来源：model.compile() - Keras模型方法，用于编译模型
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
    run_eagerly=True,
)

# Define callbacks.
# 模块来源：keras.callbacks.ModelCheckpoint - Keras回调函数，用于保存最佳模型
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.keras", save_best_only=True
)
# 模块来源：keras.callbacks.EarlyStopping - Keras回调函数，用于早停
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# Train the model, doing validation at the end of each epoch
epochs = 100
# 模块来源：model.fit() - Keras模型方法，用于训练模型
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)

"""
It is important to note that the number of samples is very small (only 200) and we don't
specify a random seed. As such, you can expect significant variance in results. The full dataset
which consists of over 1000 CT scans can be found [here](https://www.medrxiv.org/content/10.1101/2020.05.20.20100362v1). Using the full
dataset, an accuracy of 83% was achieved. A variability of 6-7% in the classification
performance is observed in both cases.
"""

"""
## Visualizing model performance（查看训练效果）

Here the model accuracy and loss for the training and the validation sets are plotted.
Since the validation set is class-balanced, accuracy provides an unbiased representation
of the model's performance.

讲解说明：
代码会画出：
- 训练准确率
- 验证准确率
- 训练损失
- 验证损失
如果训练准确率很高但验证准确率很低，就是过拟合。
"""

fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    # 模块来源：model.history.history - Keras模型训练历史记录
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])

"""
## Make predictions on a single CT scan（单个 CT 预测）

讲解说明：
最后一段代码会拿一个验证集 CT 做预测，输出它属于正常或异常的概率。
"""

# Load best weights.
# 模块来源：model.load_weights() - Keras模型方法，用于加载模型权重
model.load_weights("3d_image_classification.keras")
# 模块来源：model.predict() - Keras模型方法，用于预测
# 模块来源：np.expand_dims - NumPy函数，用于增加维度
prediction = model.predict(np.expand_dims(x_val[0], axis=0))[0]
scores = [1 - prediction[0], prediction[0]]

class_names = ["normal", "abnormal"]
for score, name in zip(scores, class_names):
    print(
        "This model is %.2f percent confident that CT scan is %s"
        % ((100 * score), name)
    )

"""
## Relevant Chapters from Deep Learning with Python
- [Chapter 8: Image classification](https://deeplearningwithpython.io/chapters/chapter08_image-classification)
"""
