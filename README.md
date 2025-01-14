# 基于 YOLOv8 的口罩检测系统

本项目实现了一个基于 YOLOv8 的口罩检测系统，数据集来源于 Kaggle。项目包含了训练、验证和实时预测的完整实现。

### 项目简介
本项目使用口罩检测数据集作为示例，展示了 YOLO 架构的能力。仅使用 500+ 张图像训练 10 个周期，就能够准确预测人是否正确佩戴口罩、未佩戴口罩或佩戴口罩不当。通过进一步训练，准确率还可以进一步提高。

主要功能：
- 训练脚本：在口罩检测数据集上训练 YOLOv8 模型
- 验证脚本：在测试数据集上验证模型性能
- 实时预测：使用网络摄像头进行实时口罩检测

![口罩检测示例](maksssksksss0.jpg)

### 安装步骤
1. 克隆仓库：
   ```bash
   git clone https://github.com/yourusername/face-mask-detection.git
   cd face-mask-detection
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 从 Kaggle 下载数据集，解压文件并将图像和标注文件存储在 datasets 文件夹中。

### 模型训练
运行 `face_mask_detection_training` 笔记本进行模型训练。

### 模型验证
运行 `face_mask_detection_validation` 笔记本进行模型验证，并将模型保存为 onnx 格式用于部署。

### 实时预测
要使用网络摄像头进行实时预测，请执行：
```bash
python live_prediction.py 
```

### 实验结果
YOLOv8 模型在训练 10 个周期后在口罩检测任务上取得了不错的准确率。以下是验证集的混淆矩阵示例：

![混淆矩阵](confusion_matrix.png)

### 贡献指南
欢迎贡献！如有改进建议或错误修复，请提出 issue 或提交 pull request。

### 许可证
本项目采用 MIT 许可证。详情请参见 LICENSE 文件。

口罩检测系统 © 2024 作者：Supriya Rani，采用 CC BY 4.0 许可证。
查看许可证副本，请访问 https://creativecommons.org/licenses/by/4.0/
