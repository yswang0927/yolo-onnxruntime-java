package com.fh.gdk.ai.yolo.spine;

import java.util.List;

import com.fh.gdk.ai.AiException;
import com.fh.gdk.ai.yolo.PosePredictResult;
import com.fh.gdk.ai.yolo.Yolov8;

public class SpineYolov8 extends Yolov8 {

    public SpineYolov8(String modelPath) {
        super(modelPath, 0.7f, 0.5f);
    }

    public void predictSpine(String imagePath) {
        List<PosePredictResult> posePredictResults = this.predictPose(imagePath);
        // TODO 处理通用的姿态结果为spine结果

    }

}
