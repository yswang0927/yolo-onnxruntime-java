package com.fh.gdk;

import com.fh.gdk.ai.yolo.PoseInference;

public class Test {

    public static void main(String[] args) throws Exception {
        String modelPath = "best.onnx";
        String imgPath = "1.jpg";
        int trainImgsz = 2048;


        /*SpineModelInference modelInference = new SpineModelInference(modelPath);
        modelInference.predictSpine(imgPath, trainImgsz);
        modelInference.close();*/

        modelPath = "yolov8n-pose.onnx";
        imgPath = "person.jpg";

        PoseInference poseInference = new PoseInference(modelPath);
        poseInference.predict(imgPath);
        poseInference.close();
    }

}
