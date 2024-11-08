package com.fh.gdk.ai.yolo.spine;

/**
 * 椎骨
 */
public class Vertebrae {

    private int index;

    // 椎骨名称：T1~T12、L1~L5
    private String label;

    // 每块椎骨的四角顶点位置坐标
    // [0:lft_top_x, 1:lft_top_y, 2:rgt_top_x, 3:rgt_top.y, 4:lft_btm_x, 5:lft_btm_y, 6:rgt_btm_x, 7:rgt_btm_y]
    // P0 P1
    // P2 P3
    private double[] cornerPoints;

    // 椎骨的中心点坐标（最小外接矩形的中心点）
    private double[] centerPoint;

    // 椎骨平均倾斜角度
    private double slopeAngle;

    Vertebrae(double[] cornerPoints) {
        this.cornerPoints = cornerPoints;

        // 通过最小外接矩形，计算椎骨的中心点坐标
        double minX = cornerPoints[0];
        double maxX = cornerPoints[0];
        double minY = cornerPoints[1];
        double maxY = cornerPoints[1];
        for (int i = 2; i < cornerPoints.length; i += 2) {
            if (cornerPoints[i] < minX) {
                minX = cornerPoints[i];
            }
            if (cornerPoints[i] > maxX) {
                maxX = cornerPoints[i];
            }
            if (cornerPoints[i + 1] < minY) {
                minY = cornerPoints[i + 1];
            }
            if (cornerPoints[i + 1] > maxY) {
                maxY = cornerPoints[i + 1];
            }
        }

        this.centerPoint = new double[] {
                (minX + maxX) / 2,
                (minY + maxY) / 2
        };

        //System.out.println(String.format("{%.2f, %.2f}", this.centerPoint[0], this.centerPoint[1]));

        this.slopeAngle = this.calcSlopeAngle();
    }

    /**
     * 计算椎骨的平均倾斜角度：上边缘和下边缘的倾斜角度
     * @return
     */
    private double calcSlopeAngle() {
        double upSlope = SpineUtils.lineSlope(this.cornerPoints[0], this.cornerPoints[1], this.cornerPoints[2], this.cornerPoints[3]);
        double dwnSlope = SpineUtils.lineSlope(this.cornerPoints[4], this.cornerPoints[5], this.cornerPoints[6], this.cornerPoints[7]);
        double upAngle = Math.toDegrees(Math.atan(upSlope));
        double dwnAngle = Math.toDegrees(Math.atan(dwnSlope));
        return (upAngle + dwnAngle) / 2;
    }

    public String getLabel() {
        return label;
    }

    public Vertebrae setLabel(String label) {
        this.label = label;
        return this;
    }

    /**
     * 顺时针方向的四角坐标：
     * 0 1
     * 3 2
     * @return [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
     */
    public double[][] getCornerPoints() {
        return new double[][] {
            { this.cornerPoints[0], this.cornerPoints[1] },
            { this.cornerPoints[2], this.cornerPoints[3] },
            { this.cornerPoints[6], this.cornerPoints[7] },
            { this.cornerPoints[4], this.cornerPoints[5] }
        };
    }

    public double[] getCenterPoint() {
        return this.centerPoint;
    }

    public double getSlopeAngle() {
        return this.slopeAngle;
    }

    public double[] getLeftTopPoint() {
        return new double[] { this.cornerPoints[0], this.cornerPoints[1] };
    }

    public double[] getLeftBottomPoint() {
        return new double[] { this.cornerPoints[4], this.cornerPoints[5] };
    }

    public double[] getRightTopPoint() {
        return new double[] { this.cornerPoints[2], this.cornerPoints[3] };
    }

    public double[] getRightBottomPoint() {
        return new double[] { this.cornerPoints[6], this.cornerPoints[7] };
    }

    public double getUpSlope() {
        return SpineUtils.lineSlope(this.cornerPoints[0], this.cornerPoints[1], this.cornerPoints[2], this.cornerPoints[3]);
    }

    public double getDownSlope() {
        return SpineUtils.lineSlope(this.cornerPoints[4], this.cornerPoints[5], this.cornerPoints[6], this.cornerPoints[7]);
    }

    @Override
    public String toString() {
        return "Vertebrae {" +
                "label='" + label + '\'' +
                ", angle=" + String.format("%.2f", this.getSlopeAngle()) +
                '}';
    }
}
