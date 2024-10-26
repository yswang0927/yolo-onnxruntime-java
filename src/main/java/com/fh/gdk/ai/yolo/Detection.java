package com.fh.gdk.ai.yolo;

public class Detection {

    public final String label;
    public final float[] bbox;
    public final float confidence;

    public Detection(String label, float[] bbox, float confidence) {
        this.label = label;
        this.bbox = bbox;
        this.confidence = confidence;
    }

}
