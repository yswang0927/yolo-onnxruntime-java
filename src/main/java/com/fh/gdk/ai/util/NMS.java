package com.fh.gdk.ai.util;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

public class NMS {

    public static List<float[]> nonMaxSuppression(List<float[]> bboxes, float iouThreshold) {
        // output boxes
        List<float[]> bestBboxes = new ArrayList<>();
        // confidence 按顺序排序
        bboxes.sort(Comparator.comparing(a -> a[4]));
        // standard nms
        while (!bboxes.isEmpty()) {
            float[] bestBbox = bboxes.remove(bboxes.size() - 1);  // 弹出当前置信度最高的框
            bestBboxes.add(bestBbox);
            bboxes = bboxes.stream().filter(a -> computeIOU(a, bestBbox) < iouThreshold).collect(Collectors.toList());
        }

        return bestBboxes;
    }

    private static float computeIOU(float[] box1, float[] box2) {
        float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
        float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);

        float left = Math.max(box1[0], box2[0]);
        float top = Math.max(box1[1], box2[1]);
        float right = Math.min(box1[2], box2[2]);
        float bottom = Math.min(box1[3], box2[3]);

        float interArea = Math.max(right - left, 0) * Math.max(bottom - top, 0);
        float unionArea = area1 + area2 - interArea;

        return Math.max(interArea / unionArea, 1e-8f);
    }

    public static List<PEResult> nms(List<PEResult> boxes, float iouThreshold) {
        // 根据score从大到小对List进行排序
        boxes.sort((b1, b2) -> Float.compare(b2.getScore(), b1.getScore()));
        List<PEResult> resultList = new ArrayList<>();
        for (int i = 0; i < boxes.size(); i++) {
            PEResult box = boxes.get(i);
            boolean keep = true;
            // 从i+1开始，遍历之后的所有boxes，移除与box的IOU大于阈值的元素
            for (int j = i + 1; j < boxes.size(); j++) {
                PEResult otherBox = boxes.get(j);
                float iou = getIntersectionOverUnion(box, otherBox);
                if (iou > iouThreshold) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                resultList.add(box);
            }
        }

        return resultList;
    }

    private static float getIntersectionOverUnion(PEResult box1, PEResult box2) {
        float x1 = Math.max(box1.getX0(), box2.getX0());
        float y1 = Math.max(box1.getY0(), box2.getY0());
        float x2 = Math.min(box1.getX1(), box2.getX1());
        float y2 = Math.min(box1.getY1(), box2.getY1());
        float intersectionArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        float box1Area = (box1.getX1() - box1.getX0()) * (box1.getY1() - box1.getY0());
        float box2Area = (box2.getX1() - box2.getX0()) * (box2.getY1() - box2.getY0());
        float unionArea = box1Area + box2Area - intersectionArea;
        return intersectionArea / unionArea;
    }

}
