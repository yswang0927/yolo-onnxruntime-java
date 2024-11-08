package com.fh.gdk.ai.yolo.spine;

import java.util.ArrayList;
import java.util.List;

import com.fh.gdk.ai.util.CurveInflectionPoints;
import org.apache.commons.math3.analysis.polynomials.PolynomialFunction;

/**
 * 脊柱
 */
public class Spine {

    // 椎骨名称：12节胸椎(T1~T12)、5节腰椎(L1~L5)
    public static final String[] VERTEBRAE_NAMES = { "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5" };

    // 脊柱的矩形边界区域坐标：[x_min, y_min, x_max, y_max]
    private double[] bbox;

    // 脊柱中的椎骨
    private List<Vertebrae> vertebraes;

    private String curveType;
    private int[] ptRegion;
    private int[] mtRegion;
    private int[] tlRegion;

    Spine(double[] bbox) {
        this.bbox = bbox;
    }

    public List<Vertebrae> getVertebraes() {
        return vertebraes;
    }

    public void setVertebraes(List<Vertebrae> vertebraes) {
        this.vertebraes = vertebraes;
        this.determineSpineShapeAndRegions();
        //a();
    }

    private void determineSpineShapeAndRegions() {
        List<Integer> directionChanges = new ArrayList<>();
        for (int i = 1; i < this.vertebraes.size(); i++) {
            if ((this.vertebraes.get(i).getAngle() > 0 && this.vertebraes.get(i-1).getAngle() < 0)
                || (this.vertebraes.get(i).getAngle() < 0 && this.vertebraes.get(i-1).getAngle() > 0)
            ) {
                directionChanges.add(i);
            }
        }

        if (directionChanges.size() >= 2) {
            this.curveType = "S";
            this.ptRegion = new int[]{0, directionChanges.get(0)};
            this.mtRegion = new int[]{directionChanges.get(0), directionChanges.get(1)};
            this.tlRegion = new int[]{directionChanges.get(1), 17 };
        } else {
            this.curveType = "C";
        }
    }

    private void a() {
        // 示例数据，这里假设已经有了68个平面坐标点，实际应用中需要替换为真实数据
        List<double[]> pointsList = new ArrayList<>();

        for (Vertebrae v : vertebraes) {
            double[][] cornerPoints = v.getCornerPoints();
            double[] center = {
                    (cornerPoints[0][0] + cornerPoints[1][0]) / 2,
                    (cornerPoints[1][1] + cornerPoints[2][1]) / 2
            };

            pointsList.add(center);
        }

        System.out.println(pointsList.size());

        CurveInflectionPoints curve = new CurveInflectionPoints(pointsList);

        // 进行三次多项式拟合
        PolynomialFunction fittedPolynomial = curve.polynomialFit(5);

        // 计算拟合多项式的二阶导数
        PolynomialFunction secondDerivative = curve.secondDerivative(fittedPolynomial);

        // 找到可能的拐点
        double[] possibleInflectionPoints = curve.findPossibleInflectionPoints(secondDerivative);

        // 检查并确定真正的拐点
        List<double[]> inflectionPoints = curve.checkInflectionPoints(secondDerivative, possibleInflectionPoints);

        System.out.println("找到的拐点坐标：" + inflectionPoints.size());
        for (double[] point : inflectionPoints) {
            System.out.println("(" + point[0] + ", " + point[1] + ")");
        }
    }

}
