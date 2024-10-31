package com.fh.gdk.ai.yolo;

public class ImageMetaData {

    public final double dw;
    public final double dh;
    public final double ratio;

    ImageMetaData(double dw, double dh, double ratio) {
        this.dw = dw;
        this.dh = dh;
        this.ratio = ratio;
    }

}
