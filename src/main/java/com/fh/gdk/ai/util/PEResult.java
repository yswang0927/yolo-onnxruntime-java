package com.fh.gdk.ai.util;

import java.util.ArrayList;
import java.util.List;

public class PEResult {
    // 框的4个坐标点
    private final Float x0;
    private final Float y0;
    private final Float x1;
    private final Float y1;
    private final Float score;
    private final Integer clsId;
    private final List<KeyPoint> keyPointList;

    public PEResult(float[] peResult) {
        float x = peResult[0];
        float y = peResult[1];
        float w = peResult[2] / 2.0f;
        float h = peResult[3] / 2.0f;
        this.x0 = x - w;
        this.y0 = y - h;
        this.x1 = x + w;
        this.y1 = y + h;
        this.score = peResult[4];
        this.clsId = 0; //(int) peResult[5];
        this.keyPointList = new ArrayList<>();
        // 6 = 前面框坐标点(4个) + 1个置信度 + 1个类别
        // /3 - 因为每个关键点是3个数值表示(x,y,v)
        int skipPos = 5;
        int keyPointNum = (peResult.length - skipPos) / 3;
        for (int i = 0; i < keyPointNum; i++) {
            this.keyPointList.add(new KeyPoint(i, peResult[skipPos + 3 * i], peResult[skipPos + 3 * i + 1], peResult[skipPos + 3 * i + 2]));
        }
    }

    public Float getX0() {
        return x0;
    }

    public Float getY0() {
        return y0;
    }

    public Float getX1() {
        return x1;
    }

    public Float getY1() {
        return y1;
    }

    public Float getScore() {
        return score;
    }

    public Integer getClsId() {
        return clsId;
    }

    public List<KeyPoint> getKeyPointList() {
        return keyPointList;
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder("PEResult:" +
                "  x0=" + x0 +
                ", y0=" + y0 +
                ", x1=" + x1 +
                ", y1=" + y1 +
                ", score=" + score +
                ", clsId=" + clsId +
                "\n");
        for (KeyPoint x : keyPointList) {
            result.append(x.toString());
        }
        return result.toString();
    }
}