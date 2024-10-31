package com.fh.gdk.ai.util;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public final class ImageUtil {

    private ImageUtil() {
    }

    public static Mat resizeWithPadding(Mat src, int width, int height) {
        Mat dst = new Mat();
        int oldW = src.width();
        int oldH = src.height();

        double r = Math.min((double) width / oldW, (double) height / oldH);

        int newUnpadW = (int) Math.round(oldW * r);
        int newUnpadH = (int) Math.round(oldH * r);

        int dw = (width - newUnpadW) / 2;
        int dh = (height - newUnpadH) / 2;

        int top = (int) Math.round(dh - 0.1);
        int bottom = (int) Math.round(dh + 0.1);
        int left = (int) Math.round(dw - 0.1);
        int right = (int) Math.round(dw + 0.1);

        Imgproc.resize(src, dst, new Size(newUnpadW, newUnpadH));
        Core.copyMakeBorder(dst, dst, top, bottom, left, right, Core.BORDER_CONSTANT);

        return dst;
    }

    public static void resizeWithPadding(Mat src, Mat dst, int width, int height) {
        int oldW = src.width();
        int oldH = src.height();

        double r = Math.min((double) width / oldW, (double) height / oldH);

        int newUnpadW = (int) Math.round(oldW * r);
        int newUnpadH = (int) Math.round(oldH * r);

        int dw = (width - newUnpadW) / 2;
        int dh = (height - newUnpadH) / 2;

        int top = (int) Math.round(dh - 0.1);
        int bottom = (int) Math.round(dh + 0.1);
        int left = (int) Math.round(dw - 0.1);
        int right = (int) Math.round(dw + 0.1);

        Imgproc.resize(src, dst, new Size(newUnpadW, newUnpadH));
        Core.copyMakeBorder(dst, dst, top, bottom, left, right, Core.BORDER_CONSTANT);
    }

    public static void xywh2xyxy(float[] bbox) {
        float x = bbox[0];
        float y = bbox[1];
        float w = bbox[2];
        float h = bbox[3];

        bbox[0] = x - w * 0.5f;
        bbox[1] = y - h * 0.5f;
        bbox[2] = x + w * 0.5f;
        bbox[3] = y + h * 0.5f;
    }

    public void scaleCoords(float[] bbox, float orgW, float orgH, float padW, float padH, float gain) {
        // xmin, ymin, xmax, ymax -> (xmin_org, ymin_org, xmax_org, ymax_org)
        bbox[0] = Math.max(0, Math.min(orgW - 1, (bbox[0] - padW) / gain));
        bbox[1] = Math.max(0, Math.min(orgH - 1, (bbox[1] - padH) / gain));
        bbox[2] = Math.max(0, Math.min(orgW - 1, (bbox[2] - padW) / gain));
        bbox[3] = Math.max(0, Math.min(orgH - 1, (bbox[3] - padH) / gain));
    }

    /**
     * 调整 src中的 [宽度,高度,通道] ->[通道,宽度,高度]
     * @param src 待调整的数组
     * @return 调整后的结果
     */
    public static float[] whc2cwh(float[] src) {
        float[] chw = new float[src.length];
        int j = 0;
        for (int ch = 0; ch < 3; ++ch) {
            for (int i = ch; i < src.length; i += 3) {
                chw[j] = src[i];
                j++;
            }
        }
        return chw;
    }

    public static void whc2cwh(float[] src, float[] dst, int start) {
        int j = start;
        for (int ch = 0; ch < 3; ++ch) {
            for (int i = ch; i < src.length; i += 3) {
                dst[j] = src[i];
                j++;
            }
        }
    }

    public static byte[] whc2cwh(byte[] src) {
        byte[] chw = new byte[src.length];
        int j = 0;
        for (int ch = 0; ch < 3; ++ch) {
            for (int i = ch; i < src.length; i += 3) {
                chw[j] = src[i];
                j++;
            }
        }
        return chw;
    }

}
