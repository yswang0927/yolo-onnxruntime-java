package com.fh.gdk;

import java.io.IOException;

import com.fh.gdk.ai.util.OpenCVUtils;

public class OpenCVTest {

    public static void main(String[] args) throws IOException {
        // 示例：预处理图片并打印数组长度
        String imagePath = "1.jpg";
        int[] bytes = OpenCVUtils.preprocessImageBytes(imagePath, 2048);
        System.out.println("归一化后的数据长度：" + bytes.length);
    }


}
