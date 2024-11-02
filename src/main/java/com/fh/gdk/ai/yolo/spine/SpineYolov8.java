package com.fh.gdk.ai.yolo.spine;

import java.nio.charset.StandardCharsets;
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

/**
 * 脊柱侧弯凸评估。
 * 柱侧凸评估需要 近胸角(PT)、主胸角(MT) 和 胸腰角(TL)三个Cobb角
 */
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
        // 颜色 Scalar(Blue, Green, Red) BGR格式
        Point topLeft = new Point((posResult.getX0() - dw) / ratio, (posResult.getY0() - dh) / ratio);
        Point bottomRight = new Point((posResult.getX1() - dw) / ratio, (posResult.getY1() - dh) / ratio);
        Imgproc.rectangle(inputImg, topLeft, bottomRight, new Scalar(255, 0, 0), 2);

        // 椎骨索引号
        int vertebraIndex = 0;

        float[] kp;
        double[] test_x = new double[68];
        double[] test_y = new double[68];
        for (int p = 0; p < keypoints.length; p++) {
            kp = keypoints[p];

            Point center = new Point((kp[0] - dw) / ratio, (kp[1] - dh) / ratio);
            Scalar color = new Scalar( 255, 0, 0 );
            Imgproc.circle(inputImg, center, radius, color, -1); //-1表示实心

            test_x[p] = center.x;
            test_y[p] = center.y;
            System.out.println(String.format("第 %d 个关键点：(%f, %f)", p+1, center.x, center.y));

            // 每一节椎骨（每节椎骨4个关键点）
            if (p % 4 == 0) {
                vertebraIndex++;

                Point bp1 = new Point((kp[0] - dw) / ratio, (kp[1] - dh) / ratio);
                Point bp2 = new Point((keypoints[p + 3][0] - dw) / ratio, (keypoints[p + 3][1] - dh) / ratio);
                //Imgproc.rectangle(inputImg, bp1, bp2, new Scalar(255, 255, 0), 2);

                // Z字形连接 0-1-2-3 每节上的4个点
                // 0 1
                // 2 3
                Point[] vertePoints = new Point[4];
                for (int j = 0; j < 4; j++) {
                    vertePoints[j] = new Point((keypoints[p + j][0] - dw) / ratio, (keypoints[p + j][1] - dh) / ratio);
                }
                // 划Z线
                //MatOfPoint matOfPoint = new MatOfPoint(vertePoints);
                //Imgproc.polylines(inputImg, Arrays.asList(matOfPoint), false, new Scalar(0, 255, 0), 2);
                // 填充多边形
                Point temp = vertePoints[2];
                vertePoints[2] = vertePoints[3];
                vertePoints[3] = temp;
                MatOfPoint matOfPoint = new MatOfPoint(vertePoints);
                Imgproc.polylines(inputImg, Arrays.asList(matOfPoint), true, new Scalar(255, 235, 0, 50));


                // 绘制椎骨名称
                Imgproc.putText(inputImg, Spine.VERTEBRAE_NAMES[vertebraIndex-1], new Point((bp1.x + bp2.x)/2 - 10, (bp1.y + bp2.y)/2 + 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, color, 2);

                // 计算每个椎骨的上斜率和下斜率
            }
        }

        // 测试绘制cobb角度线，spine2.jpg的测试数据
        double[][][] midLines = {
                {{464, 536}, {560, 537}},
                {{471, 591}, {560, 594}},
                {{482, 655}, {569, 649}},
                {{494, 721}, {576, 708}},
                {{509, 783}, {591, 765}},
                {{529, 846}, {609, 828}},
                {{538, 905}, {624, 897}},
                {{537, 967}, {631, 969}},
                {{528, 1027}, {626, 1038}},
                {{509, 1091}, {610, 1115}},
                {{484, 1165}, {591, 1194}},
                {{455, 1253}, {570, 1273}},
                {{439, 1351}, {555, 1355}},
                {{438, 1455}, {554, 1440}},
                {{450, 1562}, {578, 1533}},
                {{474, 1672}, {612, 1636}},
                {{494, 1767}, {654, 1741}}
        } ;

        for (double[][] mp : midLines) {
            Point p1 = new Point(mp[0]);
            Point p2 = new Point(mp[1]);
            Imgproc.line(inputImg, p1, p2, new Scalar(0, 255, 0), 2);
        }

        double[] pt = {27.5, 4, 10};
        double[] mt = {29.6, 10, 15};
        double[] tl = {5.2, 15, 15};

        double[][] cobbAngles = {pt, mt, tl};
        Scalar[] colors = {new Scalar(0, 255, 255), new Scalar(255, 0, 255), new Scalar(0, 0, 255)};
        int c = 0;
        for (double[] cobb : cobbAngles) {
            // 角度，保留一位小数
            String angle = String.format("%.1f", cobb[0]); // + "°";

            int line1 = (int) cobb[1];
            int line2 = (int) cobb[2];

            // 绘制夹角线
            double[][] topLine = SpineUtils.createLongLine(midLines[line1][0][0], midLines[line1][0][1], midLines[line1][1][0], midLines[line1][1][1], imgMetaData.getSrcWith());
            Point tp1 = new Point(topLine[0]);
            Point tp2 = new Point(topLine[1]);
            Imgproc.line(inputImg, tp1, tp2, colors[c], 3);

            double[][] bottomLine = SpineUtils.createLongLine(midLines[line2][0][0], midLines[line2][0][1], midLines[line2][1][0], midLines[line2][1][1], imgMetaData.getSrcWith());
            Point bp1 = new Point(bottomLine[0]);
            Point bp2 = new Point(bottomLine[1]);
            Imgproc.line(inputImg, bp1, bp2, colors[c], 3);

            // 绘制角度文本
            // 判断文本是在脊柱左侧？还是右侧？根据上下线的夹角是在左边还是右边
            Point anglePoint = null;
            // 夹角在右侧
            if (bp2.y - tp2.y < bp1.y - tp1.y) {
                anglePoint = new Point(bp2.x - 200, (bp2.y + tp2.y) / 2);
            }
            // 夹角在左侧
            else if (bp2.y - tp2.y > bp1.y - tp1.y) {
                anglePoint = new Point(tp1.x + 100, (bp1.y + tp1.y) / 2);
            }

            if (anglePoint != null) {
                Imgproc.putText(inputImg, angle, anglePoint, Imgproc.FONT_HERSHEY_SIMPLEX, 2, new Scalar(255, 0, 0), 2);
            }

            c++;
        }

        // 保存图像
        Imgproc.cvtColor(inputImg, inputImg, Imgproc.COLOR_RGB2BGR);
        Imgcodecs.imwrite("test_out_spine_cobb.jpg", inputImg);

        System.out.println(Arrays.toString(test_x));
        System.out.println(Arrays.toString(test_y));

    }

}
