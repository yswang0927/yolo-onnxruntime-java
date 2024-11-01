package com.fh.gdk.ai.yolo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/**
 * 人体姿态
 */
public class PersonPoseYolov8 extends Yolov8 {

    public static final List<Scalar> palette= new ArrayList<>(Arrays.asList(
            new Scalar( 255, 128, 0 ),
            new Scalar( 255, 153, 51 ),
            new Scalar( 255, 178, 102 ),
            new Scalar( 230, 230, 0 ),
            new Scalar( 255, 153, 255 ),
            new Scalar( 153, 204, 255 ),
            new Scalar( 255, 102, 255 ),
            new Scalar( 255, 51, 255 ),
            new Scalar( 102, 178, 255 ),
            new Scalar( 51, 153, 255 ),
            new Scalar( 255, 153, 153 ),
            new Scalar( 255, 102, 102 ),
            new Scalar( 255, 51, 51 ),
            new Scalar( 153, 255, 153 ),
            new Scalar( 102, 255, 102 ),
            new Scalar( 51, 255, 51 ),
            new Scalar( 0, 255, 0 ),
            new Scalar( 0, 0, 255 ),
            new Scalar( 255, 0, 0 ),
            new Scalar( 255, 255, 255 )
    ));

    public static final int[][] skeleton = {
            {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12},
            {7, 13}, {6, 7}, {6, 8}, {7, 9}, {8, 10}, {9, 11}, {2, 3},
            {1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}
    };

    public static final List<Scalar> poseLimbColor = new ArrayList<>(Arrays.asList(
            palette.get(9), palette.get(9), palette.get(9), palette.get(9), palette.get(7),
            palette.get(7), palette.get(7), palette.get(0), palette.get(0), palette.get(0),
            palette.get(0), palette.get(0), palette.get(16), palette.get(16), palette.get(16),
            palette.get(16), palette.get(16), palette.get(16), palette.get(16)));

    public static final List<Scalar> poseKptColor = new ArrayList<>(Arrays.asList(
            palette.get(16), palette.get(16), palette.get(16), palette.get(16), palette.get(16),
            palette.get(0), palette.get(0), palette.get(0), palette.get(0), palette.get(0),
            palette.get(0), palette.get(9), palette.get(9), palette.get(9), palette.get(9),
            palette.get(9), palette.get(9)));


    public PersonPoseYolov8(String modelPath) {
        super(modelPath);
    }

    public void predictPersonPose(String imagePath) {
        List<PosePredictResult> poseResults = this.predictPose(imagePath);

        // =================================================================
        // 测试绘图
        Mat inputImg = Imgcodecs.imread(imagePath);

        for (PosePredictResult posResult : poseResults) {
            ImageMetaData imgMetaData = posResult.getImageMetaData();
            // 在这里先定义下线的粗细、关键的半径(按比例设置大小粗细比较好一些)
            int minDwDh = Math.min(imgMetaData.getSrcWith(), imgMetaData.getSrcHeight());
            int thickness = minDwDh / LINE_THICKNESS_RATIO;
            int radius = minDwDh / DOT_RADIUS_RATIO;
            double dw = imgMetaData.getDw();
            double dh = imgMetaData.getDh();
            double ratio = imgMetaData.getRatio();

            // 画边界框
            Point topLeft = new Point((posResult.getX0() - dw) / ratio, (posResult.getY0() - dh) / ratio);
            Point bottomRight = new Point((posResult.getX1() - dw) / ratio, (posResult.getY1() - dh) / ratio);
            Imgproc.rectangle(inputImg, topLeft, bottomRight, new Scalar(255, 0, 0), thickness);

            // 画关键点
            float[][] keypoints = posResult.keypoints;
            float[] kp;
            for (int p = 0; p < keypoints.length; p++) {
                kp = keypoints[p];
                Point center = new Point((kp[0] - dw) / ratio, (kp[1] - dh) / ratio);
                Scalar color = new Scalar( 255, 0, 0 );
                Imgproc.circle(inputImg, center, radius, color, -1); //-1表示实心
            }

            // 姿态17个点的骨架帘线
            for (int i = 0; i < skeleton.length; i++) {
                int indexPoint1 = skeleton[i][0] - 1;
                int indexPoint2 = skeleton[i][1] - 1;
                Scalar coler = poseLimbColor.get(i);
                Point point1 = new Point(
                        (keypoints[indexPoint1][0] - dw) / ratio,
                        (keypoints[indexPoint1][1] - dh) / ratio
                );
                Point point2 = new Point(
                        (keypoints[indexPoint2][0] - dw) / ratio,
                        (keypoints[indexPoint2][1] - dh) / ratio
                );
                Imgproc.line(inputImg, point1, point2, coler, thickness);
            }
        }
        // 保存图像
        Imgproc.cvtColor(inputImg, inputImg, Imgproc.COLOR_RGB2BGR);
        Imgcodecs.imwrite("test_out_person_pose.jpg", inputImg);
    }

}
