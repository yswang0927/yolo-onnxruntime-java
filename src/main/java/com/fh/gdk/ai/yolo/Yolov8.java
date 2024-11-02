package com.fh.gdk.ai.yolo;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.*;
import java.util.stream.Collectors;

import ai.onnxruntime.*;
import com.fh.gdk.ai.AiException;
import com.fh.gdk.ai.util.Letterbox;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Yolov8 implements AutoCloseable {

    static {
        // 加载 opencv 依赖的动态链接库
        System.load(Yolov8.class.getClassLoader().getResource("libopencv_java4100.so").getFile());
    }

    // 默认置信度值
    public static final float CONFIDENCE_THRESHOLD_DEFAULT = 0.7f;
    public static final float IOU_THRESHOLD_DEFAULT = 0.5f;

    // 根据图像大小按比例控制点大小及线粗
    public static final Integer DOT_RADIUS_RATIO = 268;
    public static final Integer LINE_THICKNESS_RATIO = 333;

    protected final OrtEnvironment env;
    protected final OrtSession session;

    protected final String inputName;
    // FLOAT or UINT8
    protected final OnnxJavaType inputType;
    // eg: tensor: float32[1,3,640,640]
    protected final long[] inputShape;
    protected final long inputNumElements;

    protected final String outputName;
    // eg: tensor: float32[1,56,8400]
    protected final long[] outputShape;
    protected final OnnxJavaType outputType;
    protected final long outputNumElements;

    // 置信度
    protected float confThreshold;
    protected float iouThreshold;


    public Yolov8(String modelPath) {
        this(modelPath, CONFIDENCE_THRESHOLD_DEFAULT, IOU_THRESHOLD_DEFAULT, -1);
    }

    public Yolov8(String modelPath, float confThreshold, float iouThreshold) {
        this(modelPath, confThreshold, iouThreshold, -1);
    }

    public Yolov8(String modelPath, float confThreshold, float iouThreshold, int gpuDeviceId) {
        this.confThreshold = confThreshold;
        this.iouThreshold = iouThreshold;

        try {
            this.env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();

            // 使用GPU
            if (gpuDeviceId >= 0) {
                sessionOptions.addCPU(false);
                sessionOptions.addCUDA(gpuDeviceId);
                sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            }

            this.session = this.env.createSession(modelPath, sessionOptions);

            // 解析模型输入参数元数据信息
            this.inputName = this.session.getInputNames().iterator().next();
            Map<String, NodeInfo> inputInfo = this.session.getInputInfo();
            NodeInfo inputNodeInfo = inputInfo.get(this.inputName);
            TensorInfo inputTensorInfo = (TensorInfo) inputNodeInfo.getInfo();
            // eg: FLOAT, UINT8 等
            this.inputType = inputTensorInfo.type;
            // eg: (1, 3, 640, 640)
            this.inputShape = inputTensorInfo.getShape();
            this.inputNumElements = inputTensorInfo.getNumElements();

            // 解析模型输出参数元数据信息
            this.outputName = this.session.getOutputNames().iterator().next();
            Map<String, NodeInfo> outputInfo = this.session.getOutputInfo();
            NodeInfo outputNodeInfo = outputInfo.get(this.outputName);
            TensorInfo ouputTensorInfo = (TensorInfo) outputNodeInfo.getInfo();
            this.outputShape = ouputTensorInfo.getShape();
            this.outputType = ouputTensorInfo.type;
            this.outputNumElements = ouputTensorInfo.getNumElements();

        } catch (Exception e) {
            throw new AiException("Failed to init YOLO-onnx model environment!", e);
        }
    }

    /**
     * 预处理下输入的图像为模型输入要求的格式。
     * <p>当前预处理过程如下：</p>
     * <ul>
     * <li>转换 BGR -> RGB</li>
     * <li>根据模型参数调整图像大小</li>
     * <li>归一化: 确保图像的RGB值被正确归一化到[0, 1]或[-1, 1]范围</li>
     * <li>将图像数据展平为一维数组</li>
     * <li>调整图片数据的 [宽度,高度,通道] 为 [通道,宽度,高度] 格式</li>
     * </ul>
     *
     * @param inputImage 输入图像
     * @return 预处理后的图像
     */
    protected float[] preprocessImage(Mat inputImage) {
        Mat image = inputImage.clone();
        // 1.转换 BGR -> RGB
        Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2RGB);

        // 2.调整图像大小 inputShape=(1,3,640,640)
        final int netw = (int) this.inputShape[2];
        final int neth = (int) this.inputShape[3];
        Letterbox letterbox = new Letterbox(netw, neth);
        letterbox.setStride(32);
        image = letterbox.letterbox(image);

        // 3.归一化: 确保图像的RGB值被正确归一化到[0, 1]或[-1, 1]范围
        image.convertTo(image, CvType.CV_32FC3, 1.0 / 255.0);

        // 4.将图像数据展平为一维数组
        float[] imageData = new float[(int) (image.total() * image.channels())];
        image.get(0, 0, imageData);

        // 5.调整图片中的 [宽度,高度,通道] -> [通道,宽度,高度]
        imageData = hwc2chw(imageData);
        return imageData;
    }

    /**
     * 姿态推理
     * @param imagePath 输入图片
     * @return 推理结果
     * @throws AiException
     */
    public List<PosePredictResult> predictPose(String imagePath) throws AiException {
        return this.predictPose(Imgcodecs.imread(imagePath));
    }

    public List<PosePredictResult> predictPose(Mat inputImg) throws AiException {
        try {
            // 预处理图像
            Mat image = inputImg.clone();
            // 1.转换 BGR -> RGB
            Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2RGB);

            // 2.调整图像大小 inputShape=(1,3,640,640)
            Letterbox letterbox = new Letterbox((int) this.inputShape[2], (int) this.inputShape[3]);
            letterbox.setStride(32);
            image = letterbox.letterbox(image);

            // 保存下输入图片预处理后的相关元数据，可以用于自定义绘图
            final double ratio = letterbox.getRatio();
            final double dw = letterbox.getDw();
            final double dh = letterbox.getDh();
            //final int rows = letterbox.getHeight();
            //final int cols = letterbox.getWidth();
            ImageMetaData imgMetaData = new ImageMetaData();
            imgMetaData.setDw(dw);
            imgMetaData.setDh(dh);
            imgMetaData.setRatio(ratio);
            imgMetaData.setSrcWith(inputImg.width());
            imgMetaData.setSrcHeight(inputImg.height());

            // 创建输入 OnnxTensor 对象
            OnnxTensor inputTensor;
            if (OnnxJavaType.UINT8 == this.inputType) {
                byte[] whc = new byte[(int) this.inputNumElements];
                image.get(0, 0, whc);
                // 调整图片中的 [宽度,高度,通道] -> [通道,宽度,高度]
                byte[] chw = hwc2chw(whc);
                inputTensor = OnnxTensor.createTensor(this.env, ByteBuffer.wrap(chw), this.inputShape, this.inputType);
            }
            else if (OnnxJavaType.FLOAT == this.inputType) {
                // 归一化: 确保图像的RGB值被正确归一化到[0, 1]或[-1, 1]范围
                image.convertTo(image, CvType.CV_32FC1, 1. / 255);
                float[] whc = new float[(int) this.inputNumElements];
                image.get(0, 0, whc);
                // 调整图片中的 [宽度,高度,通道] -> [通道,宽度,高度]
                float[] chw = hwc2chw(whc);
                inputTensor = OnnxTensor.createTensor(this.env, FloatBuffer.wrap(chw), this.inputShape);
            }
            else {
                throw new AiException("Unsupported onnx-input-type: "+ this.inputType);
            }

            HashMap<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put(this.inputName, inputTensor);

            // 运行模型
            OrtSession.Result output = session.run(inputs);
            // 得到结果 float32[1, n, 8400]
            float[][] outputData = ((float[][][]) output.get(0).getValue())[0];
            // 转换矩阵，将 [n, 8400] 转换为 [8400, n]
            outputData = transposeMatrix(outputData);

            List<PosePredictResult> poseResults = new ArrayList<>();

            // 先使用预置的置信度过滤掉一批低置信度的
            float[] res;
            for (int i = 0, len = outputData.length; i < len; i++) {
                res = outputData[i];
                // (bbox.x, bbox.y, bbox.w, bbox.h, conf, ...)
                // res[4] = 当前框的置信度
                if (res[4] > this.confThreshold) {
                    poseResults.add(new PosePredictResult(res, 0).setImageMetaData(imgMetaData));
                }
            }

            // 对结果进行非极大值抑制：从剩下的一组存在重叠的边界框中选择最佳的边界框
            poseResults = nms(poseResults, this.iouThreshold);
            return poseResults;
        } catch (Exception e) {
            throw new AiException(e);
        }
    }

    /**
     * 调整 src中的 [宽度,高度,通道] ->[通道,宽度,高度]
     * @param src 待调整的数组
     * @return 调整后的结果
     */
    public static float[] hwc2chw(float[] src) {
        float[] chw = new float[src.length];
        int j = 0;
        for (int ch = 0; ch < 3; ++ch) {
            for (int i = ch; i < src.length; i += 3) {
                chw[j] = src[i];
                j++;
            }
        }
        return chw;
    }

    public static byte[] hwc2chw(byte[] src) {
        byte[] chw = new byte[src.length];
        int j = 0;
        for (int ch = 0; ch < 3; ++ch) {
            for (int i = ch; i < src.length; i += 3) {
                chw[j] = src[i];
                j++;
            }
        }
        return chw;
    }

    /**
     * 矩阵行列转换
     */
    private static float[][] transposeMatrix(float[][] input) {
        final int rows = input.length;
        final int cols = input[0].length;
        float[][] transposed = new float[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = input[i][j];
            }
        }
        return transposed;
    }

    /**
     * 进行非最大抑制
     */
    private static List<PosePredictResult> nms(List<PosePredictResult> poses, final float iouThreshold) {
        List<PosePredictResult> bestPoses = new ArrayList<>();
        // 先按照置信度 confidence 降序排序
        poses.sort(Comparator.comparing(a -> a.score));
        // standard nms
        while (!poses.isEmpty()) {
            PosePredictResult bestPose = poses.remove(poses.size() - 1);  // 弹出当前置信度最高的框
            bestPoses.add(bestPose);
            poses = poses.stream().filter(a -> Yolov8.computeIOU(a.bbox, bestPose.bbox) < iouThreshold).collect(Collectors.toList());
        }

        return bestPoses;
    }

    /*protected static List<float[]> nonMaxSuppression(List<float[]> bboxes, float iouThreshold) {
        // output boxes
        List<float[]> bestBboxes = new ArrayList<>();
        // confidence 按顺序排序
        bboxes.sort(Comparator.comparing(a -> a[4]));
        // standard nms
        while (!bboxes.isEmpty()) {
            float[] bestBbox = bboxes.remove(bboxes.size() - 1);  // 弹出当前置信度最高的框
            bestBboxes.add(bestBbox);
            bboxes = bboxes.stream().filter(a -> computeIOU(a, bestBbox) < iouThreshold).collect(Collectors.toList());
        }

        return bestBboxes;
    }*/

    public static float computeIOU(float[] box1, float[] box2) {
        float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
        float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);

        float left = Math.max(box1[0], box2[0]);
        float top = Math.max(box1[1], box2[1]);
        float right = Math.min(box1[2], box2[2]);
        float bottom = Math.min(box1[3], box2[3]);

        float interArea = Math.max(right - left, 0) * Math.max(bottom - top, 0);
        float unionArea = area1 + area2 - interArea;

        return Math.max(interArea / unionArea, 1e-8f);
    }

    protected void xywh2xyxy(float[] bbox) {
        float x = bbox[0];
        float y = bbox[1];
        float w = bbox[2];
        float h = bbox[3];

        bbox[0] = x - w * 0.5f;
        bbox[1] = y - h * 0.5f;
        bbox[2] = x + w * 0.5f;
        bbox[3] = y + h * 0.5f;
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

    public void setConfidenceThreshold(float confThreshold) {
        this.confThreshold = confThreshold;
    }

    public void setIouThreshold(float iouThreshold) {
        this.iouThreshold = iouThreshold;
    }

    @Override
    public void close() throws Exception {
        if (this.session != null) {
            this.session.close();
        }

        if (this.env != null) {
            this.env.close();
        }
    }

}