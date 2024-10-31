package com.fh.gdk;


import com.fh.gdk.ai.yolo.Yolov8;

public class Yolov8PoseTest2 {

    public static void main(String[] args) {
        String modelPath = "yolov8n-pose.onnx";
        String image = "person.jpg";

        modelPath = "yolov8-spine-pose.onnx";
        image = "spine1.jpg";

        Yolov8 yolo = new Yolov8(modelPath);
        yolo.predictPose(image);

    }

}
