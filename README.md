# YOLOv1 · PyTorch · PASCAL VOC 2012

模型按 YOLOv1 论文图 3 实现。输入为 `448 × 448` RGB 图像，经过 24 层卷积和 2 层全连接，输出 `7 × 7 × 30`。每个网格预测两个框，每个框包含中心偏移、宽高的平方根和置信度。两个框共享该网格的 20 个类别分数。

## 数据与环境

在项目根目录运行命令。训练脚本默认读取 `dataset/VOC2012_train_val`，该目录下需要有 `JPEGImages`、`Annotations` 和 `ImageSets/Main`。如有不同的存放位置，用 `--data` 指定目录。训练使用 `train.txt` 的 5,717 张图，验证使用 `val.txt` 的 5,823 张图。

```powershell
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

CUDA 可用时，训练默认使用 GPU 和混合精度。若第二条命令输出 `False`，可以先用 CPU 跑下面的样例。需要 GPU 训练时，按官方安装说明选择适配显卡驱动的 CUDA 版 PyTorch。 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)

## 训练

先用一张训练图和一张验证图检查数据、损失、验证和权重保存。

```powershell
python train.py --device cpu --fc-size 64 --epochs 1 --batch-size 1 --limit 1 --map --output runs/quickcheck/last.pt
```

完整结构默认使用 `--fc-size 4096`。原版约有 2.72 亿参数，显存紧张时可以缩小第一层全连接层。下面是 4 GB 显存可尝试的起始配置。

```powershell
python train.py --device cuda --fc-size 512 --batch-size 1 --workers 2 --map
```

`--map` 每轮计算 VOC 插值 `mAP@0.5`。验证使用完整的原始标注，保留同网格里的多个对象，评估时按 VOC 规则忽略匹配到 `difficult` 的检测。训练标签跳过 `difficult`，同网格中有多个候选时保留面积最大的框。损失由定位、前景置信度、背景置信度和分类四部分组成，前景由当前 IoU 最高的框负责。

每轮结束后更新 `runs/yolov1/last.pt`，并在第 5、10、15 轮等节点保存 `runs/yolov1/checkpoints/epoch_0005.pt`、`epoch_0010.pt` 等快照。这些文件都包含模型、优化器和混合精度状态，可以用 `--resume` 恢复。继续训练时，`--epochs` 表示训练到第几轮，模型宽度和峰值学习率从权重文件恢复。自定义 `--output` 时，快照和 TensorBoard 日志都放在该文件所在的运行目录中。

```powershell
python train.py --device cuda --batch-size 1 --workers 2 --map --resume runs/yolov1/last.pt --epochs 150
```

默认峰值学习率为 `0.001`，预热后在完成第 75 和 105 轮时各降至原来的十分之一，可用 `--lr` 调整峰值。论文还使用了 VOC2007 数据，验证结果应按这里的 VOC2012 划分解读。 [YOLOv1 论文](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html)

## 使用 ImageNet 预训练权重

将作者发布的 Darknet `extraction.weights` 放在 `pretrained/` 下，用 `--pretrained` 开始一次 VOC2012 训练。程序加载前 20 个卷积层，把 BatchNorm 参数折叠进卷积权重和偏置。后 4 层卷积和 2 层全连接从随机权重开始，所有卷积层都参与微调。 [官方 Extraction 模型](https://pjreddie.com/darknet/imagenet/#extraction)

```powershell
python train.py --pretrained pretrained/extraction.weights --device cuda --fc-size 512 --batch-size 1 --map
```

默认保存到 `runs/yolov1-pretrained/last.pt`，对应的 TensorBoard 日志位于 `runs/yolov1-pretrained/tensorboard`。继续该实验时，从生成的 PyTorch checkpoint 恢复，并指定同一个输出路径。

```powershell
python train.py --resume runs/yolov1-pretrained/last.pt --output runs/yolov1-pretrained/last.pt --device cuda --batch-size 1 --epochs 135 --map
```

## 训练中查看 TensorBoard

在另一个终端里启动 TensorBoard，打开 [http://localhost:6006](http://localhost:6006)，在 **Scalars** 页面查看曲线。自定义了输出路径时，将 `--logdir` 改成对应运行目录下的 `tensorboard`。

```powershell
python -m tensorboard.main --logdir runs/yolov1/tensorboard --port 6006
```

横轴是累计训练 batch 数。`loss/train_batch` 和 `optimizer/lr` 每个 batch 记录一次。`loss/train_epoch` 和 `loss/val_epoch` 每轮记录一次。加上 `--map` 后，每轮还会记录 `metrics/mAP50`。训练开始后首批数据立即写入，后续日志在训练中持续写入。

从新版 checkpoint 恢复到同一运行目录时，续训会接着原有日志记录。旧版 checkpoint 恢复后会创建一份新的曲线，从恢复后的训练开始记录。开启新实验时，可以用 `--output runs/新实验名/last.pt` 分开存放曲线和权重。

## 单图推理

```powershell
python predict.py dataset/VOC2012_train_val/JPEGImages/2008_000008.jpg --weights runs/yolov1/last.pt --output runs/prediction.jpg
```

推理将框置信度与类别分数相乘，按类别执行 NMS，再把检测框绘制回原图。用 `--score` 调整最低得分，用 `--iou` 调整 NMS 重叠阈值。一轮小样本试跑生成的权重用于验证流程，正式检测使用完整训练后的权重。

## 代码入口

- `voc.py` 读取 VOC XML、增强图像并编码网格标签。
- `yolo.py` 定义模型、损失及预测解码。
- `metrics.py` 计算 VOC mAP。
- `train.py` 和 `predict.py` 分别运行训练与单图推理。
