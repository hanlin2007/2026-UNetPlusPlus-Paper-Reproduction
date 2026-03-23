
#  UNet++ 论文复现笔记：使用SSH远程连接云服务器复现论文模型


> 这篇学习笔记，是我在2026年联创团队AI组春招期间的论文复现实践，记录了我对于UNet++(Nested U-Net) 图像分割模型的完整复现历程，并使用vscode+SSH远程连接AutoDL云服务器训练了整个模型。在这个README文档中，我将详细说明环境搭建、数据集准备、模型训练与评估，以及连接云服务器以及在云服务器中搭建环境的完整流程。  
> 本文适合想入门论文复现或学习SSH连接云服务器训练深度学习项目的读者参考。

---
## 参考链接

- [UNet++ 原论文](https://arxiv.org/abs/1807.10165)
- [官方实现：MrGiovanni/UNetPlusPlus](https://github.com/MrGiovanni/UNetPlusPlus)
- [参考实现：pytorch-nested-unet](https://github.com/4uiiurz1/pytorch-nested-unet)
- [Kaggle DSB2018 数据集](https://www.kaggle.com/c/data-science-bowl-2018/data)

## 前置知识与论文回顾

语义分割是指对图像中的**每个像素进行分类**，常用于：医学图像：肺结节、细胞核、肝脏、息肉等区域提取，以及自动驾驶：道路、行人、车辆识别等领域

UNet++ 是 [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165) 中提出的一种改进型 U-Net 结构，通过嵌套的密集连接，减少了编码器与解码器之间的语义鸿沟，在医学图像分割任务中取得了更优效果。

从 U-Net 到 UNet++：

- U-Net 通过编码器-解码器结构 + 跳跃连接，保留空间信息
- UNet++ 在跳跃连接中引入**嵌套的密集卷积块**，让不同尺度的特征更充分融合，减少语义差距

---

## 适用于本地与云服务器的环境搭建

> 本部分提供**完整、可复现**的环境配置方案，包含 **本地 Conda 环境** 与 **Docker 镜像** 两种方式。
> 需要注意，在云服务器中搭建环境时，请使用conda命令创建虚拟环境，避免直接在base环境中使用pip命令，造成base环境的污染或者依赖包的冲突。

### 环境依赖清单

| 组件          | 版本                          |
|---------------|-------------------------------|
| 操作系统      | Ubuntu 20.04 / 22.04（推荐） |
| GPU 驱动      | ≥ 470                         |
| CUDA          | 11.8                          |
| Python        | 3.9                           |
| PyTorch       | 2.0.0+cu118                   |
| 显存          | ≥ 8GB（训练 DSB2018 可低至 4GB） |

## 1️⃣ 本地 Conda 环境

### ① 创建并激活环境

```bash
conda create -n nested_unet python=3.9 -y
conda activate nested_unet
```

### ② 安装 Python 依赖

检查 `requirements.txt`文件，确保保存为为以下内容：

```
numpy==1.23.5
albumentations==0.5.2
opencv-python-headless==4.8.1.78
scikit-learn==1.3.2
scikit-image==0.21.0
pandas==2.0.3
matplotlib==3.7.1
tqdm==4.61.2
imageio==2.37.2
pyyaml==6.0.3
Pillow==11.3.0
imgaug==0.4.0
```

然后安装：

```bash
pip install -r requirements.txt
```

### ③ 安装 GPU 版 PyTorch

```bash
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

> ⚠️ 如果使用 `conda install` 安装 PyTorch，请务必指定 `cudatoolkit` 版本（如 `conda install pytorch==2.0.0 torchvision==0.15.0 cudatoolkit=11.8 -c pytorch`），但推荐使用 `pip` 以获得更精确的控制。

### ④ 导出环境文件便于复现

```bash
conda env export > environment.yml
```

以后在新机器上复现环境只需：

```bash
conda env create -f environment.yml
```

---

## 2️⃣ Docker 环境

> 为了适配需要**彻底隔离环境**的项目，或在多台服务器之间快速迁移，Docker 是最佳选择。  
> 以下步骤基于 **NVIDIA CUDA 官方镜像**，已包含所有依赖，并预置训练命令。

### ① 准备 Dockerfile

在项目根目录创建 `Dockerfile`，内容如下：

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY . /workspace

RUN pip install --upgrade pip

RUN pip install \
    numpy==1.23.5 \
    albumentations==0.5.2 \
    opencv-python-headless==4.8.1.78 \
    scikit-learn==1.3.2 \
    scikit-image==0.21.0 \
    pandas==2.0.3 \
    matplotlib==3.7.1 \
    tqdm==4.61.2 \
    imageio==2.37.2 \
    pyyaml==6.0.3 \
    Pillow==11.3.0 \
    imgaug==0.4.0

RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

CMD ["python", "train.py", "--dataset", "dsb2018_96", "--arch", "NestedUNet"]
```

### ② 构建镜像

```bash
docker build -t nested_unet .
```

> 首次构建需要下载基础镜像和依赖包，耗时约 5–10 分钟。

### ③ 运行训练，使用 GPU 加速

```bash
docker run --gpus all -it nested_unet
```

### ④ 导出镜像，用于迁移

```bash
docker save nested_unet > nested_unet.tar
```

在其他服务器上恢复镜像：

```bash
docker load < nested_unet.tar
docker run --gpus all -it nested_unet
```

---

## 数据集准备

### 使用 DSB2018 数据集

1. 从 [Kaggle Data Science Bowl 2018](https://www.kaggle.com/c/data-science-bowl-2018/data) 下载数据
2. 解压后目录结构如下：

```
inputs
└── data-science-bowl-2018
    ├── stage1_train
    │   ├── 00ae65...
    │   │   ├── images
    │   │   └── masks
    │   └── ...
```

> ⚠️ 一定要保证项目结构目录与复现要求中的一致！否则数据处理会遇到路径不匹配的问题

3. 运行预处理脚本：

```bash
python preprocess_dsb2018.py
```

---

## 模型训练

```bash
python train.py --dataset dsb2018_96 --arch NestedUNet
```

## 模型评估

```bash
python val.py --name dsb2018_96_NestedUNet_woDS
```

---

## 复现常见问题与解决方案

| 问题 | 解决方案 |
|------|----------|
| 依赖包版本冲突 | 使用 `conda` + `pip` 混合安装，固定 PyTorch 版本为 1.8+ |
| 云服务器 CUDA 版本不兼容 | 选择 4090D 显卡，安装 CUDA 11.8 对应的 PyTorch |
| 数据集传输慢 | 先压缩为 `.zip`，再上传后解压 |
| 训练时显存不足 | 减小 `--batch_size`，或使用混合精度训练 |
| 远程终端无法显示图像 | 在训练时关闭 `--vis` 参数，或使用 TensorBoard 远程查看 |

---

## 许可

本项目遵循 [MIT License](LICENSE) 协议，代码可自由使用、修改和分发，仅需保留原始版权声明。

## 贡献与交流

- 如果你在复现过程中遇到任何问题，欢迎提交 [Issue](https://github.com/your-repo/issues) 讨论  
- 如果你有改进建议、新功能或修复，欢迎发起 Pull Request  
- 如果你觉得本项目对你的学习有帮助，不妨点个 ⭐ 收藏
