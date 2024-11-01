package com.fh.gdk.ai.yolo;

public class ImageMetaData {

    private int srcWith;
    private int srcHeight;

    private double dw;
    private double dh;
    private double ratio;

    public ImageMetaData() {
    }

    public int getSrcWith() {
        return srcWith;
    }

    public void setSrcWith(int srcWith) {
        this.srcWith = srcWith;
    }

    public int getSrcHeight() {
        return srcHeight;
    }

    public void setSrcHeight(int srcHeight) {
        this.srcHeight = srcHeight;
    }

    public double getDw() {
        return dw;
    }

    public void setDw(double dw) {
        this.dw = dw;
    }

    public double getDh() {
        return dh;
    }

    public void setDh(double dh) {
        this.dh = dh;
    }

    public double getRatio() {
        return ratio;
    }

    public void setRatio(double ratio) {
        this.ratio = ratio;
    }



}
