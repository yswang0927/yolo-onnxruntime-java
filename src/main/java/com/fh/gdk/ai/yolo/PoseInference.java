package com.fh.gdk.ai.yolo;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.FloatBuffer;
import java.util.List;
import java.util.*;
import javax.imageio.ImageIO;

import ai.onnxruntime.*;
import com.fh.gdk.ai.util.ImageUtil;
import com.fh.gdk.ai.util.OpenCVUtils;

/**
 * 姿态推理
 */
public class PoseInference {

    private OrtEnvironment env;
    private OrtSession session;
    private String inputName;
    private String outputName;
    private long[] inputShape;

    public PoseInference(String modelPath) throws OrtException {
        this.env = OrtEnvironment.getEnvironment(); //创建环境

        OrtSession.SessionOptions sessionOpts = new OrtSession.SessionOptions();
        //int gpuDeviceId = 0; // The GPU device ID to execute on
        //sessionOpts.addCUDA(gpuDeviceId);

        this.session = this.env.createSession(modelPath, sessionOpts); //创建会话
        this.inputName = this.session.getInputNames().iterator().next();
        this.outputName = this.session.getOutputNames().iterator().next();

        TensorInfo inputTensorInfo = (TensorInfo) this.session.getInputInfo().get(this.inputName).getInfo();
        // eg: [1, 3, 640, 640]
        this.inputShape = inputTensorInfo.getShape();

        /*System.out.println(">>> 输入信息：");
        for (Map.Entry<String, NodeInfo> nodeInfoEntry : this.session.getInputInfo().entrySet()) {
            System.out.println(nodeInfoEntry.getKey() + " : "+ nodeInfoEntry.getValue().toString());
        }
        System.out.println(">>> 输出信息：");
        for (Map.Entry<String, NodeInfo> nodeInfoEntry : this.session.getOutputInfo().entrySet()) {
            System.out.println(nodeInfoEntry.getKey() + " : "+ nodeInfoEntry.getValue().toString());
        }*/

        System.out.println(">>> YOLOv8 模型加载成功！");
    }

    /**
     * 姿态推理
     * @param imagePath 输入图片
     * @throws Exception
     */
    public void predict(String imagePath) throws Exception {
        // 预处理图片
        float[] imageData = preprocessImage(imagePath, (int) this.inputShape[2]);
        float[] chw = ImageUtil.whc2cwh(imageData);
        OnnxTensor inputTensor = OnnxTensor.createTensor(this.env, FloatBuffer.wrap(chw), this.inputShape);

        //byte[] imgRawBytes = Files.readAllBytes(new File(imagePath).toPath());
        //long[] shape = { imgRawBytes.length };
        //OnnxTensor inputTensor = OnnxTensor.createTensor(this.env, ByteBuffer.wrap(imgRawBytes), shape, OnnxJavaType.UINT8);

        // 执行推理
        /*
        YOLOv8 模型的输出通常是一个 (1, N, 6) 形状的张量，其中：
         - N 是检测到的目标数量。
         - 6 表示每个目标的检测信息：[x_min, y_min, x_max, y_max, score, class_id]。
         */
        Map<String, OnnxTensor> inputs = Collections.singletonMap(this.inputName, inputTensor);
        Set<String> outputs = new HashSet<>();
        outputs.add(this.outputName);

        try (OrtSession.Result results = session.run(inputs, outputs)) {
            float[][][] output = (float[][][]) results.get(0).getValue();

            // 解析检测结果
            List<PoseResult> detections = parsePoseResults(output);

            // 类别名称列表（示例，可以根据实际情况修改）
            List<String> classNames = Arrays.asList("spine");

            // 加载原始图片
            BufferedImage inputImage = ImageIO.read(new File(imagePath));
            // 绘制检测结果
            BufferedImage outputImage = drawDetections(inputImage, detections);

            // 保存结果图片
            File srcImgPath = new File(imagePath);
            File outImgPath = new File(srcImgPath.getParent(), "test_out.jpg");
            ImageIO.write(outputImage, "jpg", outImgPath);
            System.out.println("检测结果已保存：" + outImgPath);

        }
    }


    // 预处理图片：缩放到 640x640(示例)，归一化 RGB 值
    private static float[] preprocessImage(String imagePath, int trainImgsz) throws Exception {
        return OpenCVUtils.preprocessImage(imagePath, trainImgsz);

        /*BufferedImage img = ImageIO.read(new File(imagePath));
        BufferedImage resizedImage = new BufferedImage(trainImgsz, trainImgsz, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resizedImage.createGraphics();
        g.drawImage(img, 0, 0, trainImgsz, trainImgsz, null);
        g.dispose();

        // 将图像转换为float数组并进行归一化
        float[] imageData = new float[trainImgsz * trainImgsz * 3]; // 3个通道
        for (int y = 0; y < trainImgsz; y++) {
            for (int x = 0; x < trainImgsz; x++) {
                int rgb = resizedImage.getRGB(x, y);
                // 提取RGB值并归一化
                imageData[(y * trainImgsz + x) * 3 + 0] = ((rgb >> 16) & 0xFF) / 255.0f; // R
                imageData[(y * trainImgsz + x) * 3 + 1] = ((rgb >> 8) & 0xFF) / 255.0f;  // G
                imageData[(y * trainImgsz + x) * 3 + 2] = (rgb & 0xFF) / 255.0f;         // B
            }
        }
        return imageData;*/
    }

    // 解析 YOLOv8-Pose 的输出结果
    private static List<PoseResult> parsePoseResults(float[][][] output) {
        List<PoseResult> results = new ArrayList<>();
        /*
        YOLOv8-Pose 输出的张量形状为 (1, N, 57)，每个 N 表示一个检测对象，57 包括：
            - 4 个边界框坐标 (x_min, y_min, x_max, y_max)
            - 1 个类别 id
            - 1 个置信度 (score)
            - 17 个关键点，每个关键点包含 x, y, confidence（共 51 个值）
            [0,1,2,3] 4个索引是边界框的索引
            索引[4] score
            索引[5] class
            索引[6,] 关键点
         */
        final int keyPointsCount = 17;
        for (float[] detection : output[0]) {
            float score = detection[4];
            int classId = (int) detection[5];
            if (classId == 0) {
                System.out.println(">>>> " + detection[5]);
            }

            if (score > 0.5) {  // 置信度筛选
                float[] bbox = Arrays.copyOfRange(detection, 0, 4);
                float[][] keypoints = new float[keyPointsCount][3];  // 每个关键点 (x, y, confidence)
                for (int i = 0; i < keyPointsCount; i++) {
                    keypoints[i][0] = detection[6 + i * 3];     // x
                    keypoints[i][1] = detection[6 + i * 3 + 1]; // y
                    keypoints[i][2] = detection[6 + i * 3 + 2]; // confidence
                }
                results.add(new PoseResult(bbox, keypoints, classId, score));
            }
        }

        return results;
    }

    // 在图片上绘制检测结果
    private static BufferedImage drawDetections(BufferedImage image, List<PoseResult> detections) {
        Graphics2D g = image.createGraphics();
        g.setStroke(new BasicStroke(2));
        g.setFont(new Font("Arial", Font.BOLD, 16));

        for (PoseResult result : detections) {
            g.setColor(Color.RED);

            // 绘制边界框
            int xMin = (int) result.bbox[0];
            int yMin = (int) result.bbox[1];
            int width = (int) (result.bbox[2] - result.bbox[0]);
            int height = (int) (result.bbox[3] - result.bbox[1]);
            g.drawRect(xMin, yMin, width, height);

            // 绘制关键点
            g.setColor(Color.BLUE);
            for (float[] keypoint : result.keypoints) {
                int x = (int) keypoint[0];
                int y = (int) keypoint[1];
                g.fillOval(x - 3, y - 3, 6, 6);  // 在关键点位置绘制圆点
            }

            // 绘制类别和置信度
            String label = String.format("ID: %d, Score: %.2f", result.classId, result.score);
            g.drawString(label, xMin, yMin - 5);
        }
        g.dispose();
        return image;
    }

    public void close() throws OrtException {
        if (session != null) {
            session.close();
        }
        if (env != null) {
            env.close();
        }
    }

}
