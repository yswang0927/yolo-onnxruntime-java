package com.fh.gdk.ai.yolo;

/**
 * Yolo姿态检测结果
 */
public class PosePredictResult {
    // 边界框 [x_min, y_min, x_max, y_max]
    public final float[] bbox;

    // 关键点，每个关键点包含 (x, y, confidence)
    public final float[][] keypoints;

    // 类别 ID
    public final int classId;

    // 置信度
    public final float score;

    private ImageMetaData imageMetaData;

    public PosePredictResult(float[] tensorResult, int classId) {
        // 前5个值含义：[box中心点x, box中心点y, box宽度, box高度，置信度分数]
        float cx = tensorResult[0];
        float cy = tensorResult[1];
        float w = tensorResult[2] / 2.0f;
        float h = tensorResult[3] / 2.0f;
        float x0 = cx - w;
        float y0 = cy - h;
        float x1 = cx + w;
        float y1 = cy + h;

        this.bbox = new float[] {x0, y0, x1, y1};
        this.score = tensorResult[4];
        this.classId = classId;

        // Pose 输出是格式是 [1,56,8400], 56 = bbox(4) + conf(1) + 17kpt * 3
        // 因此，关键点数据要跳过的 5 个特殊数据
        final int skipPos = 5;
        // 由于每个关键点是由 3 个值(x,y,conf)组成，因此关键点个数 = 所有点的坐标数量 / 3
        final int kptsNum = (tensorResult.length - skipPos) / 3;

        this.keypoints = new float[kptsNum][3];
        for (int i = 0; i < kptsNum; i++) {
            this.keypoints[i] = new float[] {
                    tensorResult[skipPos + 3 * i],
                    tensorResult[skipPos + 3 * i + 1],
                    tensorResult[skipPos + 3 * i + 2]
            };
        }
    }

    public ImageMetaData getImageMetaData() {
        return imageMetaData;
    }

    public PosePredictResult setImageMetaData(ImageMetaData imageMetaData) {
        this.imageMetaData = imageMetaData;
        return this;
    }

    public float getX0() {
        return this.bbox[0];
    }

    public float getY0() {
        return this.bbox[1];
    }

    public float getX1() {
        return this.bbox[2];
    }

    public float getY1() {
        return this.bbox[3];
    }

}
