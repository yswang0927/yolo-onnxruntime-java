package com.fh.gdk;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.fh.gdk.ai.util.*;
import com.fh.gdk.ai.yolo.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Yolov7PoseTest {

    static {
        System.load(Yolov7PoseTest.class.getClassLoader().getResource("libopencv_java4100.so").getFile());
    }

    public static void main(String[] args) throws OrtException {
        String model_path = "yolov7-w6-pose.onnx";
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

        // 读取 image
        Mat img = Imgcodecs.imread("person.jpg");
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2RGB);
        Mat image = img.clone();

        // 在这里先定义下线的粗细、关键的半径(按比例设置大小粗细比较好一些)
        int minDwDh = Math.min(img.width(), img.height());
        int thickness = minDwDh / PEConfig.lineThicknessRatio;
        int radius = minDwDh / PEConfig.dotRadiusRatio;

        // 更改 image 尺寸
        Letterbox letterbox = new Letterbox(960, 960);
        letterbox.setStride(32);
        image = letterbox.letterbox(image);

        double ratio = letterbox.getRatio();
        double dw = letterbox.getDw();
        double dh = letterbox.getDh();
        int rows = letterbox.getHeight();
        int cols = letterbox.getWidth();
        int channels = image.channels();

        // 将Mat对象的像素值赋值给Float[]对象
        float[] pixels = new float[channels * rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double[] pixel = image.get(j, i);
                for (int k = 0; k < channels; k++) {
                    // 这样设置相当于同时做了image.transpose((2, 0, 1))操作
                    pixels[rows * cols * k + j * cols + i] = (float) pixel[k] / 255.0f;
                }
            }
        }

        // 创建OnnxTensor对象
        long[] shape = {1L, (long) channels, (long) rows, (long) cols};
        OnnxTensor tensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(pixels), shape);
        HashMap<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put(session.getInputInfo().keySet().iterator().next(), tensor);

        // 运行模型
        OrtSession.Result output = session.run(inputs);

        // 得到结果
        float[][] outputData = ((float[][][]) output.get(0).getValue())[0];

        List<PEResult> peResults = new ArrayList<>();
        for (float[] outputDatum : outputData) {
            PEResult result = new PEResult(outputDatum);
            if (result.getScore() > PEConfig.personScoreThreshold) {
                peResults.add(result);
            }
        }

        // 对结果进行非极大值抑制
        peResults = NMS.nms(peResults, PEConfig.IoUThreshold);

        for (PEResult peResult: peResults) {
            System.out.println(peResult);
            // 画框
            Point topLeft = new Point((peResult.getX0()-dw)/ratio, (peResult.getY0()-dh)/ratio);
            Point bottomRight = new Point((peResult.getX1()-dw)/ratio, (peResult.getY1()-dh)/ratio);
            Imgproc.rectangle(img, topLeft, bottomRight, new Scalar(255,0,0), thickness);
            List<KeyPoint> keyPoints = peResult.getKeyPointList();
            // 画点
            keyPoints.forEach(keyPoint->{
                if (keyPoint.getScore()>PEConfig.keyPointScoreThreshold) {
                    Point center = new Point((keyPoint.getX()-dw)/ratio, (keyPoint.getY()-dh)/ratio);
                    Scalar color = PEConfig.poseKptColor.get(keyPoint.getId());
                    Imgproc.circle(img, center, radius, color, -1); //-1表示实心
                }
            });
            // 画线
            for (int i = 0; i< PEConfig.skeleton.length; i++){
                int indexPoint1 = PEConfig.skeleton[i][0]-1;
                int indexPoint2 = PEConfig.skeleton[i][1]-1;
                if ( keyPoints.get(indexPoint1).getScore()>PEConfig.keyPointScoreThreshold &&
                        keyPoints.get(indexPoint2).getScore()>PEConfig.keyPointScoreThreshold ) {
                    Scalar coler = PEConfig.poseLimbColor.get(i);
                    Point point1 = new Point(
                            (keyPoints.get(indexPoint1).getX()-dw)/ratio,
                            (keyPoints.get(indexPoint1).getY()-dh)/ratio
                    );
                    Point point2 = new Point(
                            (keyPoints.get(indexPoint2).getX()-dw)/ratio,
                            (keyPoints.get(indexPoint2).getY()-dh)/ratio
                    );
                    Imgproc.line(img, point1, point2, coler, thickness);
                }
            }
        }

        Imgproc.cvtColor(img, img, Imgproc.COLOR_RGB2BGR);
        // 保存图像
        Imgcodecs.imwrite("test_out.jpg", img);

    }

}
