package com.fh.gdk.ai.yolo.spine;

public class SpineUtils {

    private SpineUtils() {}

    /**
     * 给定四个点坐标，计算这两条直线交点坐标。
     * @param x1 第一条直线的第一个点坐标
     * @param y1 第一条直线的第一个点坐标
     * @param x2 第一条直线的第二个点坐标
     * @param y2 第一条直线的第二个点坐标
     * @param x3 第二条直线的第一个点坐标
     * @param y3 第二条直线的第一个点坐标
     * @param x4 第二条直线的第二个点坐标
     * @param y4 第二条直线的第二个点坐标
     * @return 交点坐标
     */
    public static double[] intersectionPoint(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4) {
        double a1 = y2 - y1;
        double b1 = x1 - x2;
        double c1 = a1 * x1 + b1 * y1;

        double a2 = y4 - y3;
        double b2 = x3 - x4;
        double c2 = a2 * x3 + b2 * y3;

        double denominator = a1 * b2 - a2 * b1;
        // 平行线
        if (denominator == 0) {
            return new double[0];
        }

        double x = (b2 * c1 - b1 * c2) / denominator;
        double y = (a1 * c2 - a2 * c1) / denominator;
        return new double[] {x, y};
    }

    /**
     * 计算线段的斜率。
     * @return
     */
    public static double lineSlope(double x1, double y1, double x2, double y2) {
        if ((x2 - x1) == 0) {
            return 0;
        }
        return (y2 - y1) / (x2 - x1);
    }

    /**
     * 计算两条线的夹角。
     * <pre>夹角 = arctan((斜率1 – 斜率2) / (1 + 斜率1 * 斜率2))。</pre>
     * @return
     */
    public static double linesAngle(double s1, double s2) {
        return Math.toDegrees(Math.atan((s2 - s1) / (1 + (s2 * s1))));
    }

    // p1和p2 组合成线段1，p3和p4组合成线段2
    public static double getDegress(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4){
        // 这里是p2-p1也可以是p1-p2 位置是无所谓的，只是要统一。如果x轴是p2-p1，那么y轴也得是p2-p1
        double d1x = x2 - x1;
        double d1y = y2 - y1;
        // 这个的逻辑和上面一样，p3-p4或者p4-p3都可以
        double d2x = x4 - x3;
        double d2y = y4 - y3;
        // 然后通过atan2 得到弧度，要注意了这个方法中必须是y轴值在前面，x轴值在后面
        // 两个弧度相减，就是两个线段的夹角弧度
        double angle = Math.atan2(d1y, d1x) - Math.atan2(d2y, d2x);
        // 将弧度，转为角度。并通过绝对值去除正负符号
        angle = Math.abs(Math.toDegrees(angle));
        if (angle > 180) {
            // 因为线段夹角内角+外角=360°，
            // 如果超过180°了说明我们得到的是最大的外角了，而夹角应该是最小的角度，所以进行了360-angile
            angle = 360 - angle;
        }
        return angle;
    }

    /**
     * 计算矩形内2点连线和矩形两边的交点，用于绘制延长线段
     */
    public static double[][] createLongLine(double x1, double y1, double x2, double y2, double imageWidth) {
        double m = lineSlope(x1, y1, x2, y2);
        double nx1 = 20, nx2 = imageWidth - 20;
        double ny1 = y1 - m * (x1 - nx1);
        double ny2 = y2 + m * (nx2 - x2);
        return new double[][] { {nx1, ny1}, {nx2, ny2} };
    }

}
