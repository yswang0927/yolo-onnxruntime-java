package com.fh.gdk.ai.util;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

import javax.imageio.ImageIO;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public final class OpenCVUtils {

    static {
        System.load(OpenCVUtils.class.getClassLoader().getResource("libopencv_java4100.so").getFile());
    }

    private OpenCVUtils() {}

    /**
     * 使用 OpenCV 进行图像预处理，缩放并归一化为 640x640。
     * @param imagePath 输入图片路径
     * @param imgsz 图片缩放到的尺寸
     * @return 归一化后的浮点数组
     */
    public static float[] preprocessImage(String imagePath, int imgsz) {
        // 1. 读取原始图片（BGR 格式）
        Mat image = Imgcodecs.imread(imagePath);

        // 2. 缩放: 确保图像被缩放到模型期望的输入尺寸（如640x640）
        Mat resizedImage = new Mat();
        Imgproc.resize(image, resizedImage, new Size(imgsz, imgsz));

        // 3. 转换 BGR -> RGB
        Mat rgbImage = new Mat();
        Imgproc.cvtColor(resizedImage, rgbImage, Imgproc.COLOR_BGR2RGB);

        // 4. 归一化: 确保图像的RGB值被正确归一化到[0, 1]或[-1, 1]范围
        rgbImage.convertTo(rgbImage, CvType.CV_32FC3, 1.0 / 255.0);

        // 5. 将图像数据展平为一维数组 (640 * 640 * 3)
        float[] imageData = new float[(int) (rgbImage.total() * rgbImage.channels())];
        rgbImage.get(0, 0, imageData);
        return imageData;
    }

    public static int[] preprocessImageBytes(String imagePath, int imgsz) throws IOException {

        byte[] byteArray = Files.readAllBytes(new File(imagePath).toPath());
        int[] ss = new int[byteArray.length];
        for (int i = 0; i < byteArray.length; i++) {
            ss[i] = (byteArray[i] & 0xFF);
        }

        BufferedImage img = ImageIO.read(new File(imagePath));
        // 将图像转换为float数组并进行归一化
        int[] imageData = new int[img.getWidth() * img.getHeight() * 3]; // 3个通道
        int index = 0;
        for (int y = 0; y < img.getHeight(); y++) {
            for (int x = 0; x < img.getWidth(); x++) {
                int rgb = img.getRGB(x, y);
                imageData[index++] = (byte) ((rgb >> 16) & 0xFF); // 红色通道
                imageData[index++] = (byte) ((rgb >> 8) & 0xFF);  // 绿色通道
                imageData[index++] = (byte) (rgb & 0xFF);         // 蓝色通道
            }
        }
        return imageData;
    }


}
