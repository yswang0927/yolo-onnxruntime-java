# yolo-onnxruntime-java

## 使用微软的onnxruntime java库调用yolo-onnx格式模型进行推理。

## YOLOv8 输入、输出数据格式说明：

### 1. YOLOv8n detection model：
- 输入Shape格式：(1, 3, 640, 640)
> 1 

- 输出Shape格式：(1, 84, 8400)
> 1 = 单图像推理的批量大小
> 
> 84 = 80 个类 + 4 个边界框参数 (x, y, w, h)

### YOLOv8n-seg detection model：
