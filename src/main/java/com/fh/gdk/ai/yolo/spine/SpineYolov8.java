package com.fh.gdk.ai.yolo.spine;

import java.util.*;

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
        // 脊柱X-Ray影像上只会有一个
        PosePredictResult posResult = posePredictResults.get(0);
        // 边界框 [x_min, y_min, x_max, y_max]
        float[] bbox = posResult.bbox;
        // 关键点，应该是：68 个点(17节椎骨 * 4个角顶点)
        // [68][x,y,conf]
        float[][] keypoints = posResult.keypoints;

        if (keypoints.length != 68) {
            throw new AiException("未检测到有效的 68 个关键点");
        }

        // 图片元数据
        ImageMetaData imgMetaData = posResult.getImageMetaData();
        final double dw = imgMetaData.getDw();
        final double dh = imgMetaData.getDh();
        final double ratio = imgMetaData.getRatio();

        // 将tensor点位转换为用于绘图的UI点位
        // 矩形区域坐标
        double[] uiBboxCoord = {
                (bbox[0] - dw) / ratio, (bbox[1] - dh) / ratio,
                (bbox[2] - dw) / ratio, (bbox[3] - dh) / ratio
        };

        Spine spine = new Spine(uiBboxCoord);

        // 17节椎骨的四个边角坐标: [[lft_top_x, lft_top_y, rgt_top_x, rgt_top.y, lft_btm_x, lft_btm_y, rgt_btm_x, rgt_btm_y], ]
        List<Vertebrae> vertebraes = new ArrayList<>(24);
        // 椎骨索引号
        int vertebraIndex = 0;
        for (int p = 0; p < keypoints.length; p += 4) {
            double[] cornerPoints = {
                    (keypoints[p][0] - dw) / ratio, (keypoints[p][1] - dh) / ratio,
                    (keypoints[p + 1][0] - dw) / ratio, (keypoints[p + 1][1] - dh) / ratio,
                    (keypoints[p + 2][0] - dw) / ratio, (keypoints[p + 2][1] - dh) / ratio,
                    (keypoints[p + 3][0] - dw) / ratio, (keypoints[p + 3][1] - dh) / ratio
            };
            vertebraes.add(new Vertebrae(cornerPoints).setLabel(Spine.VERTEBRAE_NAMES[vertebraIndex]));
            vertebraIndex++;
        }

        spine.setVertebraes(vertebraes);


        final int vertesCount = vertebraes.size();

        // 参照 coronal_plane_cobb.py 计算
        /*List<Map<String, Object>> vertebrae_information = new ArrayList<>();
        for (int i = 0; i < vertesCount; i++) {
            // 判断定点01的位置，是横着的还是竖着的
            Vertebrae verte = vertebraes.get(i);
            // 0 1
            // 3 2
            double[][] box = verte.getCornerPoints();

            double[] point0 = {box[0][0], box[0][1]};
            double[] point1 = {box[1][0], box[1][1]};
            double[] point2 = {box[2][0], box[2][1]};
            double distance_01 = Math.sqrt(Math.pow(point0[0] - point1[0], 2) + Math.pow(point0[1] - point1[1], 2));
            double distance_12 =  Math.sqrt(Math.pow(point1[0] - point2[0], 2) +  Math.pow(point1[1] - point2[1], 2));

            double[] center = {
                    (point0[0] + point1[0]) / 2,
                    (point1[1] + point2[1]) / 2
            };
            if (distance_01 > distance_12) {
                // up_slope = (box[3][1] - box[2][1]) / (box[3][0] - box[2][0])
                // down_slope = (box[0][1] - box[1][1]) / (box[0][0] - box[1][0])
                double up_slope = SpineUtils.lineSlope(box[2][0], box[2][1], box[3][0], box[3][1]);
                double down_slope = SpineUtils.lineSlope(box[1][0], box[1][1], box[0][0], box[0][1]);
                Map<String, Object> info = new HashMap<>();
                info.put("index", i);
                info.put("location", true);
                info.put("vertexes", box);
                info.put("center", center);
                info.put("up_slope", up_slope);
                info.put("down_slope", down_slope);

                vertebrae_information.add(info);
            }
            else {
                // up_slope = (box[2][1] - box[1][1]) / (box[2][0] - box[1][0])
                // down_slope = (box[3][1] - box[0][1]) / (box[3][0] - box[0][0])
                double up_slope = SpineUtils.lineSlope(box[1][0], box[1][1], box[2][0], box[2][1]);
                double down_slope = SpineUtils.lineSlope(box[0][0], box[0][1], box[3][0], box[3][1]);
                Map<String, Object> info = new HashMap<>();
                info.put("index", i);
                info.put("location", false);
                info.put("vertexes", box);
                info.put("center", center);
                info.put("up_slope", up_slope);
                info.put("down_slope", down_slope);

                vertebrae_information.add(info);
            }

        }

        // 找出拐点
        double[] flag_index = {-1, -1};
        List<Map<String, Object>> turning_location = new ArrayList<>();
        for (int i = 0; i < vertesCount; i++) {
            if (i + 1 < vertesCount) {
                boolean location1 = (boolean) vertebrae_information.get(i).get("location");
                boolean location2 = (boolean) vertebrae_information.get(i+1).get("location");
                if (!location1 && location2) {
                    if (flag_index[1] != i) {
                        flag_index = new double[]{ i, i + 1 };
                        Map<String, Object> turning_index = new HashMap<>();
                        turning_index.put("previous", i);
                        turning_index.put("last", i + 1);
                        turning_location.add(turning_index);
                    }
                }
                if (location1 && !location2) {
                    if (flag_index[1] != i) {
                        flag_index = new double[]{ i, i + 1 };
                        Map<String, Object> turning_index = new HashMap<>();
                        turning_index.put("previous", i);
                        turning_index.put("last", i + 1);
                        turning_location.add(turning_index);
                    }
                }
            }
        }

        // 找出上下端锥绝对值最大的椎骨
        // 并计算最大cobb角
        List<Map<String, Object>> slope_decline = new ArrayList<>(vertebrae_information);
        slope_decline.sort((a, b) -> {
            return Double.compare((double)b.get("up_slope"), (double)a.get("up_slope"));
        });*/

        // ================================================================================
        // 测试绘图
        // 在这里先定义下线的粗细、关键的半径(按比例设置大小粗细比较好一些)
        int minDwDh = Math.min(imgMetaData.getSrcWith(), imgMetaData.getSrcHeight());
        int thickness = minDwDh / LINE_THICKNESS_RATIO;
        int radius = minDwDh / DOT_RADIUS_RATIO;

        Mat inputImg = Imgcodecs.imread(imagePath);

        // 画边界框
        // 颜色 Scalar(Blue, Green, Red) BGR格式
        Imgproc.rectangle(inputImg, new Point(uiBboxCoord[0], uiBboxCoord[1]), new Point(uiBboxCoord[2], uiBboxCoord[3]), new Scalar(0,0,255), 2);

        List<Double> allX = new ArrayList<>();
        List<Double> allY = new ArrayList<>();
        for (int v = 0; v < vertesCount; v++) {
            Vertebrae vert = vertebraes.get(v);
            System.out.println(vert);
            // 绘制68个关键点，每块椎骨4个关键点
            Imgproc.circle(inputImg, new Point(vert.getLeftTopPoint()), radius, new Scalar( 255, 0, 0 ), -1); //-1表示实心
            Imgproc.circle(inputImg, new Point(vert.getRightTopPoint()), radius, new Scalar( 255, 0, 0 ), -1);
            Imgproc.circle(inputImg, new Point(vert.getLeftBottomPoint()), radius, new Scalar( 255, 0, 0 ), -1);
            Imgproc.circle(inputImg, new Point(vert.getRightBottomPoint()), radius, new Scalar( 255, 0, 0 ), -1);
            
            // 连接四个点，顺时针绘制矩形
            // 0 1
            // 3 2
            Point[] boxPoints = new Point[4];
            boxPoints[0] = new Point(vert.getLeftTopPoint());
            boxPoints[1] = new Point(vert.getRightTopPoint());
            boxPoints[2] = new Point(vert.getRightBottomPoint());
            boxPoints[3] = new Point(vert.getLeftBottomPoint());
            Imgproc.polylines(inputImg, Arrays.asList(new MatOfPoint(boxPoints)), true, new Scalar(0, 255, 0));
            // 绘制椎骨名称
            Point vertMidPoint = new Point(vert.getCenterPoint());
            Imgproc.putText(inputImg, vert.getLabel(), vertMidPoint, Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(255, 0, 0), 2);
            Imgproc.circle(inputImg, vertMidPoint, radius + 2, new Scalar(0, 255, 0), -1);

            allX.add(boxPoints[0].x);
            allX.add(boxPoints[1].x);
            allX.add(boxPoints[3].x);
            allX.add(boxPoints[2].x);

            allY.add(boxPoints[0].y);
            allY.add(boxPoints[1].y);
            allY.add(boxPoints[3].y);
            allY.add(boxPoints[2].y);
        }

        allX.addAll(allY);
        System.out.println(allX);


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

        double[] pt = {27.5, 4, 10};
        double[] mt = {29.6, 10, 15};
        double[] tl = {5.2, 15, 15};

        double[][] cobbAngles = {pt, mt};
        Scalar[] colors = {new Scalar(0, 255, 255), new Scalar(255, 0, 255), new Scalar(0, 0, 255)};
        int c = 0;
        for (double[] cobb : cobbAngles) {
            // 角度，保留一位小数
            String angle = String.format("%.1f", cobb[0]); // + "°";

            int line1 = (int) cobb[1];
            int line2 = (int) cobb[2];

            //Point[] vert1Points = vertebraUIPoints.get(line1);
            //Point[] vert2Points = vertebraUIPoints.get(line2);

            // 绘制夹角线
            double[][] topLine = SpineUtils.createLongLine(midLines[line1][0][0], midLines[line1][0][1], midLines[line1][1][0], midLines[line1][1][1], imgMetaData.getSrcWith());
            //double[][] topLine = SpineUtils.createLongLine(vert1Points[0].x, vert1Points[0].y, vert1Points[1].x, vert1Points[1].y, imgMetaData.getSrcWith());
            Point tp1 = new Point(topLine[0]);
            Point tp2 = new Point(topLine[1]);
            Imgproc.line(inputImg, tp1, tp2, colors[c], 3);

            double[][] bottomLine = SpineUtils.createLongLine(midLines[line2][0][0], midLines[line2][0][1], midLines[line2][1][0], midLines[line2][1][1], imgMetaData.getSrcWith());
            //double[][] bottomLine = SpineUtils.createLongLine(vert2Points[2].x, vert2Points[2].y, vert2Points[3].x, vert2Points[3].y, imgMetaData.getSrcWith());
            Point bp1 = new Point(bottomLine[0]);
            Point bp2 = new Point(bottomLine[1]);
            Imgproc.line(inputImg, bp1, bp2, colors[c], 3);

            //System.out.println("夹角： "+ SpineUtils.getDegress(vert1Points[0].x, vert1Points[0].y, vert1Points[1].x, vert1Points[1].y,vert2Points[3].x, vert2Points[3].y, vert2Points[2].x, vert2Points[2].y));

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


    }

}
