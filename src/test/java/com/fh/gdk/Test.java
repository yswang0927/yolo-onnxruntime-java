package com.fh.gdk;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

import Jama.Matrix;
import org.apache.commons.math3.analysis.polynomials.PolynomialFunction;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.apache.commons.math3.analysis.solvers.BrentSolver;
import org.apache.commons.math3.analysis.solvers.UnivariateSolver;
import org.apache.commons.math3.fitting.PolynomialCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoints;
import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;
import org.apache.commons.math3.analysis.differentiation.FiniteDifferencesDifferentiator;


public class Test {

    public static void main(String[] args) {
        // 假设points是一个包含17个点坐标的二维数组
        double[][] points = {
                {511.89, 537.15},
                {515.31, 591.82},
                {523.81, 651.04},
                {534.88, 716.12},
                {550.43, 776.70},
                {568.44, 836.70},
                {582.76, 900.31},
                {584.93, 969.00},
                {576.99, 1035.09},
                {558.32, 1105.80},
                {537.28, 1180.91},
                {512.51, 1263.18},
                {497.45, 1353.41},
                {496.48, 1450.10},
                {515.10, 1549.76},
                {546.28, 1654.19},
                {577.44, 1754.64}
        };

        // 构建矩阵A和向量b
        double[][] A = new double[17][6];
        double[] b = new double[17];
        for (int i = 0; i < 17; i++) {
            A[i][0] = 1;
            A[i][1] = points[i][0];
            A[i][2] = points[i][0] * points[i][0];
            A[i][3] = points[i][0] * points[i][0] * points[i][0];
            A[i][4] = points[i][0] * points[i][0] * points[i][0] * points[i][0];
            A[i][5] = points[i][0] * points[i][0] * points[i][0] * points[i][0] * points[i][0];
            b[i] = points[i][1];
        }

        // 使用最小二乘法求解方程组
        Matrix AMatrix = new Matrix(A);
        Matrix bMatrix = new Matrix(b, 17);
        Matrix xMatrix = AMatrix.solve(bMatrix);

        // 获取系数
        double[] coefficients = xMatrix.getColumnPackedCopy();

        // 计算一阶导数和二阶导数
        double[] firstDerivative = new double[6];
        double[] secondDerivative = new double[6];
        for (int i = 0; i < 6; i++) {
            firstDerivative[i] = (i == 0) ? 0 : i * coefficients[i];
            secondDerivative[i] = (i <= 1) ? 0 : i * (i - 1) * coefficients[i];
        }

        // 构建并求解二阶导数为零的方程
        double[][] secondDerivativeMatrix = new double[17][4];
        double[] secondDerivativeVector = new double[17];
        for (int i = 0; i < 17; i++) {
            secondDerivativeMatrix[i][0] = secondDerivative[2];
            secondDerivativeMatrix[i][1] = secondDerivative[3] * points[i][0];
            secondDerivativeMatrix[i][2] = secondDerivative[4] * points[i][0] * points[i][0];
            secondDerivativeMatrix[i][3] = secondDerivative[5] * points[i][0] * points[i][0] * points[i][0];
            secondDerivativeVector[i] = 0;
        }

        Matrix secondDerivativeAMatrix = new Matrix(secondDerivativeMatrix);
        Matrix secondDerivativeBMatrix = new Matrix(secondDerivativeVector, 17);
        Matrix secondDerivativeXMatrix = secondDerivativeAMatrix.solve(secondDerivativeBMatrix);

        // 获取拐点坐标
        double[] inflectionPoints = secondDerivativeXMatrix.getColumnPackedCopy();

        // 输出拐点坐标
        for (int i = 0; i < inflectionPoints.length; i++) {
            System.out.println("拐点坐标: (" + points[i][0] + ", " + coefficients[0] + coefficients[1] * points[i][0] + coefficients[2] * points[i][0] * points[i][0] + coefficients[3] * points[i][0] * points[i][0] * points[i][0] + coefficients[4] * points[i][0] * points[i][0] * points[i][0] * points[i][0] + coefficients[5] * points[i][0] * points[i][0] * points[i][0] * points[i][0] * points[i][0] + ")");
        }
    }

}
