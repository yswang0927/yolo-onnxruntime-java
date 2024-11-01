package com.fh.gdk.ai.yolo.spine;

import java.util.Arrays;
import java.util.List;

import com.fh.gdk.ai.AiException;
import com.fh.gdk.ai.yolo.ImageMetaData;
import com.fh.gdk.ai.yolo.PosePredictResult;
import com.fh.gdk.ai.yolo.Yolov8;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class SpineYolov8 extends Yolov8 {

    public SpineYolov8(String modelPath) {
        super(modelPath, 0.7f, 0.5f);
    }

    public void predictSpine(String imagePath) {
        List<PosePredictResult> posePredictResults = this.predictPose(imagePath);
        // TODO 处理通用的姿态结果为spine结果
        // 脊柱X-Ray影像上只会有一个
        PosePredictResult posResult = posePredictResults.get(0);
        // 边界框
        float[] bbox = posResult.bbox;
        // 关键点，应该是：68 个点(17节椎骨 * 4个角顶点)
        // [68][x,y,conf]
        float[][] keypoints = posResult.keypoints;

        if (keypoints.length != 68) {
            throw new AiException("未检测到有效的 68 个关键点");
        }

        // 图片元数据
        ImageMetaData imgMetaData = posResult.getImageMetaData();


        // ================================================================================
        // 测试绘图
        // 在这里先定义下线的粗细、关键的半径(按比例设置大小粗细比较好一些)
        int minDwDh = Math.min(imgMetaData.getSrcWith(), imgMetaData.getSrcHeight());
        int thickness = minDwDh / LINE_THICKNESS_RATIO;
        int radius = minDwDh / DOT_RADIUS_RATIO;
        double dw = imgMetaData.getDw();
        double dh = imgMetaData.getDh();
        double ratio = imgMetaData.getRatio();
        Mat inputImg = Imgcodecs.imread(imagePath);

        // 画边界框
        Point topLeft = new Point((posResult.getX0() - dw) / ratio, (posResult.getY0() - dh) / ratio);
        Point bottomRight = new Point((posResult.getX1() - dw) / ratio, (posResult.getY1() - dh) / ratio);
        Imgproc.rectangle(inputImg, topLeft, bottomRight, new Scalar(255, 0, 0), thickness);

        // 椎骨索引号
        int vertebraIndex = 0;

        float[] kp;
        for (int p = 0; p < keypoints.length; p++) {
            kp = keypoints[p];

            Point center = new Point((kp[0] - dw) / ratio, (kp[1] - dh) / ratio);
            Scalar color = new Scalar( 255, 0, 0 );
            Imgproc.circle(inputImg, center, radius, color, -1); //-1表示实心

            // 每一节椎骨（每节椎骨4个关键点）
            if (p % 4 == 0) {
                vertebraIndex++;

                Point bp1 = new Point((kp[0] - dw) / ratio, (kp[1] - dh) / ratio);
                Point bp2 = new Point((keypoints[p + 3][0] - dw) / ratio, (keypoints[p + 3][1] - dh) / ratio);
                //Imgproc.rectangle(inputImg, bp1, bp2, new Scalar(255, 255, 0), 2);

                // Z字形连接 0-1-2-3 每节上的4个点
                Point[] vertePoints = new Point[4];
                for (int j = 0; j < 4; j++) {
                    vertePoints[j] = new Point((keypoints[p + j][0] - dw) / ratio, (keypoints[p + j][1] - dh) / ratio);
                }
                MatOfPoint matOfPoint = new MatOfPoint(vertePoints);
                Imgproc.polylines(inputImg, Arrays.asList(matOfPoint), false, new Scalar(0, 255, 0), 2);

                // 绘制椎骨名称
                Imgproc.putText(inputImg, Spine.VERTEBRAE_NAMES[vertebraIndex-1], new Point((bp1.x + bp2.x)/2 - 10, (bp1.y + bp2.y)/2 + 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, color, 2);

                // 计算每个椎骨的上斜率和下斜率

            }

        }
        // 保存图像
        Imgproc.cvtColor(inputImg, inputImg, Imgproc.COLOR_RGB2BGR);
        Imgcodecs.imwrite("test_out_spine_cobb.jpg", inputImg);

    }

}
