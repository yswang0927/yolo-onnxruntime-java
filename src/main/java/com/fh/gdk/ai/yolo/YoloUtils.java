package com.fh.gdk.ai.yolo;

import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class YoloUtils {

    private YoloUtils() {}

    public static void drawPredictions(Mat img, List<Detection> detectionList) {
        for (Detection detection : detectionList) {
            float[] bbox = detection.bbox;
            Scalar color = new Scalar(249, 218, 60);
            Imgproc.rectangle(img,                    //Matrix obj of the image
                    new Point(bbox[0], bbox[1]),        //p1
                    new Point(bbox[2], bbox[3]),       //p2
                    color,     //Scalar object for color
                    2                        //Thickness of the line
            );
            Imgproc.putText(
                    img,
                    detection.label,
                    new Point(bbox[0] - 1, bbox[1] - 5),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    .5, color,
                    1);
        }
    }

}
