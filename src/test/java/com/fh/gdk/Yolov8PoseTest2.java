package com.fh.gdk;

import com.fh.gdk.ai.yolo.PersonPoseYolov8;
import com.fh.gdk.ai.yolo.spine.SpineYolov8;

public class Yolov8PoseTest2 {

    public static void main(String[] args) {

        //personPosePredict();
        spineCobbPredict();

    }

    public static void spineCobbPredict() {
        String modelPath = "yolov8-spine-pose.onnx";
        String image = "spine1.jpg";

        SpineYolov8 spiner = new SpineYolov8(modelPath);
        spiner.predictSpine(image);
    }

    public static void personPosePredict() {
        String modelPath = "yolov8n-pose.onnx";
        String image = "person.jpg";

        PersonPoseYolov8 yolo = new PersonPoseYolov8(modelPath);
        yolo.predictPersonPose(image);
    }

}
