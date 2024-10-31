package com.fh.gdk;

import java.nio.FloatBuffer;
import java.util.*;

import ai.onnxruntime.*;
import com.fh.gdk.ai.util.*;
import com.fh.gdk.ai.yolo.KeyPoint;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Yolov8PoseTest {

    static {
        System.load(Yolov8PoseTest.class.getClassLoader().getResource("libopencv_java4100.so").getFile());
    }

    public static void main(String[] args) throws OrtException {
        String model_path = "yolov8n-pose.onnx";
        // 加载ONNX模型
        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        OrtSession session = environment.createSession(model_path, sessionOptions);
        // 输出基本信息
        session.getInputInfo().keySet().forEach(x -> {
            try {
                System.out.println("input name = " + x);
                System.out.println(session.getInputInfo().get(x).getInfo().toString());
            } catch (OrtException e) {
                throw new RuntimeException(e);
            }
        });

        OnnxModelMetadata onnxMetadata = session.getMetadata();
        Map<String, String> customMetadata = onnxMetadata.getCustomMetadata();
        String imgsz = customMetadata.get("imgsz"); // [640, 640]
        String names = customMetadata.get("names"); // {0: 'person'}
        String task = customMetadata.get("task"); // pose
        String kptShape = customMetadata.get("kpt_shape"); // [17, 3]
        String stride = customMetadata.get("stride"); // 32
        System.out.println(">> 元数据："+ customMetadata);

        // --------- 预处理图片 -------
        // 读取 image
        Mat img = Imgcodecs.imread("pose2.jpg");
        Mat image = img.clone();
        Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2RGB);

        // 在这里先定义下线的粗细、关键的半径(按比例设置大小粗细比较好一些)
        int minDwDh = Math.min(img.width(), img.height());
        int thickness = minDwDh / PEConfig.lineThicknessRatio;
        int radius = minDwDh / PEConfig.dotRadiusRatio;

        // 更改 image 尺寸
        Letterbox letterbox = new Letterbox(640, 640);
        letterbox.setStride(32);
        image = letterbox.letterbox(image);

        double ratio = letterbox.getRatio();
        double dw = letterbox.getDw();
        double dh = letterbox.getDh();
        int rows = letterbox.getHeight();
        int cols = letterbox.getWidth();
        int channels = image.channels();

        // 4. 归一化: 确保图像的RGB值被正确归一化到[0, 1]或[-1, 1]范围
        image.convertTo(image, CvType.CV_32FC3, 1.0 / 255.0);
        float[] pixels = new float[(int) (image.total() * image.channels())];
        image.get(0, 0, pixels);

        // 调整图片中的 [宽度,高度,通道] -> [通道,宽度,高度]
        pixels = ImageUtil.whc2cwh(pixels);

        // 创建OnnxTensor对象
        long[] shape = {1L, (long) channels, (long) rows, (long) cols};
        OnnxTensor tensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(pixels), shape);
        HashMap<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put(session.getInputInfo().keySet().iterator().next(), tensor);

        // 运行模型
        OrtSession.Result output = session.run(inputs);

        /*
        YOLOv8-pose 的输出结构 (float32[1, 56, 8400]):
        1：表示 batch size。
        56：每个检测结果包含的预测值：
            4 个边界框值（中心点坐标、宽、高、及置信度）。
            1 个目标置信度。
            51 个关键点信息（17 个关键点，每个关键点有 x, y, confidence）。
        8400：表示所有可能的检测点，即在特征图中每个位置的输出。
                表示所有 anchor-free 输出位置的组合（YOLOv8 采用 grid-based 结构，无需为每个特征层设计特定的 anchor）。

         解析步骤：
            a.加载 YOLOv8-pose 的输出张量
                假设输出为 output，其 shape 为 [1, 56, 8400]。
            b.提取维度与变换张量
                通常要将维度 [1, 56, 8400] 转换为 [8400, 56] 以便逐行解析。
                output = output.squeeze(0).transpose(1, 0)  # Shape: [8400, 56]
            c.提取置信度过滤结果
                为了保留高置信度的检测结果，可以过滤掉目标置信度低于某一阈值（如 0.5）的结果。
                confidence_threshold = 0.5
                valid_detections = output[output[:, 4] > confidence_threshold]  # Shape: [N, 56]
            d.解析边界框坐标与类别ID
                每一行的前 6 个值包含边界框坐标和类别信息：
                x_min = valid_detections[i, 0]
                y_min = valid_detections[i, 1]
                x_max = valid_detections[i, 2]
                y_max = valid_detections[i, 3]
                类别 ID: class_id = int(valid_detections[i, 5])
                目标置信度: object_conf = valid_detections[i, 4]
            e.提取关键点坐标与置信度
                从索引 6 开始的部分是 17 个关键点的坐标和置信度，每个关键点占 3 个值：
                x = valid_detections[i, 6 + j * 3]
                y = valid_detections[i, 7 + j * 3]
                conf = valid_detections[i, 8 + j * 3]
                ```
                keypoints = []
                for j in range(17):
                    x = valid_detections[i, 6 + j * 3]
                    y = valid_detections[i, 7 + j * 3]
                    conf = valid_detections[i, 8 + j * 3]
                    keypoints.append((x, y, conf))
               ```
         */

        // 得到结果 float32[1,56,8400]
        float[][] outputData = ((float[][][]) output.get(0).getValue())[0];

        // 转换矩阵，将 [56, 8400] 转换为 [8400, 56]
        outputData = transpose(outputData);
        float confThreshold = 0.7f;

        List<PEResult> peResults = new ArrayList<>();
        for (float[] outputDatum : outputData) {
            PEResult result = new PEResult(outputDatum);
            if (result.getScore() > PEConfig.personScoreThreshold) {
                peResults.add(result);
            }
        }

        // 对结果进行非极大值抑制
        peResults = NMS.nms(peResults, PEConfig.IoUThreshold);

        for (PEResult peResult : peResults) {
            System.out.println(peResult);
            // 画框
            Point topLeft = new Point((peResult.getX0() - dw) / ratio, (peResult.getY0() - dh) / ratio);
            Point bottomRight = new Point((peResult.getX1() - dw) / ratio, (peResult.getY1() - dh) / ratio);
            Imgproc.rectangle(img, topLeft, bottomRight, new Scalar(255, 0, 0), thickness);
            List<KeyPoint> keyPoints = peResult.getKeyPointList();
            // 画点
            keyPoints.forEach(keyPoint -> {
                if (keyPoint.getScore() > PEConfig.keyPointScoreThreshold) {
                    Point center = new Point((keyPoint.getX() - dw) / ratio, (keyPoint.getY() - dh) / ratio);
                    Scalar color = PEConfig.poseKptColor.get(keyPoint.getId());
                    Imgproc.circle(img, center, radius, color, -1); //-1表示实心
                }
            });
            // 画线
            for (int i = 0; i < PEConfig.skeleton.length; i++) {
                int indexPoint1 = PEConfig.skeleton[i][0] - 1;
                int indexPoint2 = PEConfig.skeleton[i][1] - 1;
                if (keyPoints.get(indexPoint1).getScore() > PEConfig.keyPointScoreThreshold &&
                        keyPoints.get(indexPoint2).getScore() > PEConfig.keyPointScoreThreshold) {
                    Scalar coler = PEConfig.poseLimbColor.get(i);
                    Point point1 = new Point(
                            (keyPoints.get(indexPoint1).getX() - dw) / ratio,
                            (keyPoints.get(indexPoint1).getY() - dh) / ratio
                    );
                    Point point2 = new Point(
                            (keyPoints.get(indexPoint2).getX() - dw) / ratio,
                            (keyPoints.get(indexPoint2).getY() - dh) / ratio
                    );
                    Imgproc.line(img, point1, point2, coler, thickness);
                }
            }
        }

        Imgproc.cvtColor(img, img, Imgproc.COLOR_RGB2BGR);
        // 保存图像
        Imgcodecs.imwrite("test_out.jpg", img);

    }

    /**
     * 矩阵行列转换
     */
    private static float[][] transpose(float[][] input) {
        int rows = input.length;
        int cols = input[0].length;
        float[][] transposed = new float[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = input[i][j];
            }
        }
        return transposed;
    }

    static int argmax(float[] a) {
        float re = -Float.MAX_VALUE;
        int arg = -1;
        for (int i = 0; i < a.length; i++) {
            if (a[i] >= re) {
                re = a[i];
                arg = i;
            }
        }
        return arg;
    }

}
