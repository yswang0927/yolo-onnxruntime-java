package com.fh.gdk.ai.util;

import org.apache.commons.math3.analysis.polynomials.PolynomialFunction;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.apache.commons.math3.analysis.solvers.BrentSolver;
import org.apache.commons.math3.analysis.solvers.UnivariateSolver;
import org.apache.commons.math3.fitting.PolynomialCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoints;
import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;
import org.apache.commons.math3.analysis.differentiation.FiniteDifferencesDifferentiator;

import java.util.ArrayList;
import java.util.List;

public class CurveInflectionPoints {

    // 存储平面坐标点的列表
    private List<double[]> points = new ArrayList<>();

    public CurveInflectionPoints(List<double[]> points) {
        this.points = points;
    }

    // 使用多项式拟合曲线
    public PolynomialFunction polynomialFit(int degree) {
        WeightedObservedPoints obs = new WeightedObservedPoints();
        for (double[] point : points) {
            obs.add(point[0], point[1]);
        }

        PolynomialCurveFitter fitter = PolynomialCurveFitter.create(degree);
        double[] coefficients = fitter.fit(obs.toList());

        return new PolynomialFunction(coefficients);
    }

    // 计算多项式函数的二阶导数
    public PolynomialFunction secondDerivative(PolynomialFunction polynomial) {
        double[] coefficients = polynomial.getCoefficients();
        double[] secondDerivativeCoefficients = new double[coefficients.length - 2];

        for (int i = 2; i < coefficients.length; i++) {
            secondDerivativeCoefficients[i - 2] = coefficients[i] * i * (i - 1);
        }

        return new PolynomialFunction(secondDerivativeCoefficients);
    }

    // 找到二阶导数为零的点（可能的拐点）
    public double[] findPossibleInflectionPoints(PolynomialFunction secondDerivative) {
        UnivariateSolver solver = new BrentSolver();
        double[] possiblePoints = new double[secondDerivative.degree() - 1];

        for (int i = 0; i < secondDerivative.degree() - 1; i++) {
            possiblePoints[i] = solver.solve(100, secondDerivative, 0, (i + 1) * 1.0);
        }

        return possiblePoints;
    }

    // 检查可能的拐点处二阶导数的符号是否改变
    public List<double[]> checkInflectionPoints(PolynomialFunction secondDerivative, double[] possiblePoints) {
        List<double[]> inflectionPoints = new ArrayList<>();
        double epsilon = 0.0001;

        for (double point : possiblePoints) {
            double valueBefore = secondDerivative.value(point - epsilon);
            double valueAfter = secondDerivative.value(point + epsilon);

            if (valueBefore * valueAfter < 0) {
                // 找到对应的原曲线的y值
                double yValue = polynomialFit(3).value(point);
                inflectionPoints.add(new double[]{point, yValue});
            }
        }

        return inflectionPoints;
    }

    public static void main(String[] args) {
        // 示例数据，这里假设已经有了68个平面坐标点，实际应用中需要替换为真实数据
        List<double[]> pointsList = new ArrayList<>();
        // 这里添加68个点的坐标，示例添加几个简单点
        pointsList.add(new double[]{0, 0});
        pointsList.add(new double[]{1, 1});
        pointsList.add(new double[]{2, 4});
        pointsList.add(new double[]{3, 9});

        CurveInflectionPoints curve = new CurveInflectionPoints(pointsList);

        // 进行三次多项式拟合
        PolynomialFunction fittedPolynomial = curve.polynomialFit(3);

        // 计算拟合多项式的二阶导数
        PolynomialFunction secondDerivative = curve.secondDerivative(fittedPolynomial);

        // 找到可能的拐点
        double[] possibleInflectionPoints = curve.findPossibleInflectionPoints(secondDerivative);

        // 检查并确定真正的拐点
        List<double[]> inflectionPoints = curve.checkInflectionPoints(secondDerivative, possibleInflectionPoints);

        System.out.println("找到的拐点坐标：");
        for (double[] point : inflectionPoints) {
            System.out.println("(" + point[0] + ", " + point[1] + ")");
        }
    }
}
