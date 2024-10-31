package com.fh.gdk;

import java.util.List;

import com.fh.gdk.ai.yolo.KeyPoint;

public class PoseResult {
    public final float[] bbox; // 边界框 [x_min, y_min, x_max, y_max]
    public final List<KeyPoint> keypoints; // 关键点，每个关键点包含 (x, y, confidence)
    public final int classId; // 类别 ID
    public final float score; // 置信度

    public PoseResult(float[] bbox, List<KeyPoint> keypoints, int classId, float score) {
        this.bbox = bbox;
        this.keypoints = keypoints;
        this.classId = classId;
        this.score = score;
    }
}
