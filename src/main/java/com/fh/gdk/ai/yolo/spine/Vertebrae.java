package com.fh.gdk.ai.yolo.spine;

/**
 * 椎骨
 */
public class Vertebrae {

    private int index;
    // 椎骨名称：T1~T12、L1~L5
    private String label;
    // 每块椎骨的四角顶点位置坐标
    private double[] cornerPoints;
    // 上端斜率
    private double upSlope;
    // 下端斜率
    private double downSlope;


}
