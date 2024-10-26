package com.fh.gdk.ai.yolo;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.FloatBuffer;
import java.util.*;
import java.util.List;

import javax.imageio.ImageIO;

import ai.onnxruntime.*;
import com.fh.gdk.ai.util.OpenCVUtils;

public class SpineModelInference {

    private OrtEnvironment env;
    private OrtSession session;
    private int imgsz = 640;

    public SpineModelInference(String modelPath) throws OrtException {
        env = OrtEnvironment.getEnvironment(); //创建环境
        session = env.createSession(modelPath, new OrtSession.SessionOptions()); //创建会话

        System.out.println(">>> 输入信息：");
        for (Map.Entry<String, NodeInfo> nodeInfoEntry : this.session.getInputInfo().entrySet()) {
            System.out.println(nodeInfoEntry.getKey() + " : "+ nodeInfoEntry.getValue().toString());
        }

        System.out.println(">>> 输出信息：");
        for (Map.Entry<String, NodeInfo> nodeInfoEntry : this.session.getOutputInfo().entrySet()) {
            System.out.println(nodeInfoEntry.getKey() + " : "+ nodeInfoEntry.getValue().toString());
        }


        OnnxModelMetadata metadata = session.getMetadata();
        Map<String, String> customMetadata = metadata.getCustomMetadata();
        if (customMetadata != null) {
            // eg: [640, 640]
            String imgszStr = customMetadata.get("imgsz");

        }

        System.out.println(">>> YOLOv8 模型加载成功！");
    }

    // 预处理图片：缩放到 640x640(示例)，归一化 RGB 值
    public static float[] preprocessImage(String imagePath, int trainImgsz) throws Exception {
        return OpenCVUtils.preprocessImage(imagePath, trainImgsz);

        /*// 加载图片并调整为 trainImgsz x trainImgsz
        BufferedImage image = ImageIO.read(new File(imagePath));
        BufferedImage resizedImage = new BufferedImage(trainImgsz, trainImgsz, BufferedImage.TYPE_3BYTE_BGR);
        resizedImage.getGraphics().drawImage(image, 0, 0, trainImgsz, trainImgsz, null);

        // 转换为 Float 数组并进行归一化 (0-255 -> 0.0-1.0)
        float[] imageData = new float[trainImgsz * trainImgsz * 3];
        int index = 0;
        for (int y = 0; y < trainImgsz; y++) {
            for (int x = 0; x < trainImgsz; x++) {
                int rgb = resizedImage.getRGB(x, y);
                imageData[index++] = ((rgb >> 16) & 0xFF) / 255.0f; // R
                imageData[index++] = ((rgb >> 8) & 0xFF) / 255.0f;  // G
                imageData[index++] = (rgb & 0xFF) / 255.0f;         // B
            }
        }
        return imageData;*/
    }

    // 解析 YOLOv8-Pose 的输出结果
    private static List<PoseResult> parseSpinePoseResults(float[][][] output) {
        List<PoseResult> results = new ArrayList<>();
        /*
        YOLOv8-Pose-Spine 输出的张量形状为 (1, N, 300)，每个 N 表示一个检测对象，300 包括：
            - 4 个边界框坐标 (x_min, y_min, x_max, y_max)
            - 1 个类别 id
            - 1 个置信度 (score)
            - 68 个关键点，每个关键点包含 x, y, confidence（共 204 个值）
            [0,1,2,3] 4个索引是边界框的索引，索引[4] score、索引[5] - class，索引[6,] - 关键点
         */
        final int keyPointsCount = 68;
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

    /**
     * 脊柱推理
     * @param imagePath 输入的脊柱图片
     * @param trainImgsz 模型训练时的 imgsz 大小
     * @throws Exception
     */
    public void predictSpine(String imagePath, int trainImgsz) throws Exception {
        // 预处理图片
        long stime = System.currentTimeMillis();

        float[] imageData = preprocessImage(imagePath, trainImgsz);

        System.out.println(">>> 预处理图片耗时："+ (System.currentTimeMillis() - stime));

        stime = System.currentTimeMillis();

        long[] inputShape = {1, 3, trainImgsz, trainImgsz};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(imageData), inputShape);

        // 执行推理
        /*
        YOLOv8 模型的输出通常是一个 (1, N, 6) 形状的张量，其中：
         - N 是检测到的目标数量。
         - 6 表示每个目标的检测信息：[x_min, y_min, x_max, y_max, score, class_id]。

         */
        Map<String, OnnxTensor> inputs = Collections.singletonMap(session.getInputNames().iterator().next(), inputTensor);
        try (OrtSession.Result results = session.run(inputs)) {
            float[][][] output = (float[][][]) results.get(0).getValue();

            System.out.println(">>> 推理耗时："+ (System.currentTimeMillis() - stime));

            // 解析检测结果
            List<PoseResult> detections = parseSpinePoseResults(output);

            // 加载原始图片
            BufferedImage inputImage = ImageIO.read(new File(imagePath));

            // 类别名称列表（示例，可以根据实际情况修改）
            List<String> classNames = Arrays.asList("spine");

            // 绘制检测结果
            BufferedImage outputImage = drawSpineDetections(inputImage, detections);

            // 保存结果图片
            File srcImgPath = new File(imagePath);
            File outImgPath = new File(srcImgPath.getParent(), "test_out.jpg");
            ImageIO.write(outputImage, "jpg", outImgPath);
            System.out.println("检测结果已保存：" + outImgPath);

        }
    }

    // 在图片上绘制检测结果
    private static BufferedImage drawSpineDetections(BufferedImage image, List<PoseResult> detections) {
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

    /**
     * 在图片上绘制检测结果。
     * @param image 原始图片
     * @param detections 检测结果列表，每个元素为 [x_min, y_min, x_max, y_max, score, class_id]
     * @param classNames 类别名称列表
     * @return 带标注的图片
     */
    public static BufferedImage drawDetections(BufferedImage image, float[][] detections, List<String> classNames) {
        Graphics2D g = image.createGraphics();
        g.setStroke(new BasicStroke(2));  // 设置矩形框线条粗细
        g.setFont(new Font("Arial", Font.BOLD, 16));  // 设置字体

        for (float[] detection : detections) {
            // 解析检测结果
            int xMin = (int) detection[0];
            int yMin = (int) detection[1];
            int xMax = (int) detection[2];
            int yMax = (int) detection[3];
            float score = detection[4];
            int classId = (int) detection[5];

            // 随机颜色为每个框分配颜色（可选）
            g.setColor(new Color((int) (Math.random() * 0x1000000)));

            // 绘制矩形框
            g.drawRect(xMin, yMin, xMax - xMin, yMax - yMin);

            // 绘制类别名称和置信度分数
            String label = String.format("%s: %.2f", classNames.get(classId), score);
            g.drawString(label, xMin, yMin - 5);  // 在矩形框上方显示文本
        }
        g.dispose();  // 释放资源
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
