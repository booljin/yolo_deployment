#include <opencv2/opencv.hpp>
#include <yolo_utils.h>

/*
  * @brief: resize image
  * @detail: 来源图像可以不同，但模型输入尺寸是固定的，一个典型的尺寸是640*640
  * @param: input: 输入图像
  * @param: output: 输出图像，被缩放到适合模型输入的尺寸，且长宽比保持一致
  * @param: dst_h: 模型输入的图像高度
  * @param: dst_w: 模型输入的图像宽度
*/
void preprocess_resize_img_by_cpu(cv::Mat& input, cv::Mat& output, int dst_h, int dst_w){
    // 构造一个灰色的画布（比如典型尺寸是640*640）
    output = cv::Mat(dst_h, dst_w, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat scale_img;
    int src_w = input.cols;
    int src_h = input.rows;
    // 计算长宽各自需要缩放的比例,选其中小的那个，这样原图可以在640*640的范围内完整显示
    float scale = std::min((float)dst_h / (float)src_h, (float)dst_w / (float)src_w);
    int scale_w = (int)(scale * src_w);
    int scale_h = (int)(scale * src_h);
    scale_w = scale_w > dst_w ? dst_w : scale_w;
    scale_h = scale_h > dst_h ? dst_h : scale_h;
    cv::resize(input, scale_img, cv::Size(scale_w, scale_h), 0, 0, cv::INTER_LINEAR);
    // 缩放后的图填充到640*640画布的中央
    scale_img.copyTo(output(cv::Rect((dst_w - scale_w) / 2, (dst_h - scale_h) / 2, scale_w, scale_h)));
}

/*
  * @brief: 对图像做通道转换，并进行归一化
  * @param: img: 输入图像
  * @param: blob: 输出数据，事先开辟好dest_h * dest_w * 3 * sizeof(float)的内存
*/
void preprocess_hwc2chw_normalization_by_cpu(cv::Mat& img, float* blob){
    int img_h = img.rows;
    int img_w = img.cols;
    uchar* src_data = img.data;
    // 不需要对原图进行通道转换，逐像素处理时可以如此自然完成
    float* r = blob;
    float* g = blob + img_h * img_w;
    float* b = blob + img_h * img_w * 2;
    for(int i = 0; i < img_h * img_w; i++){
        // cv::Mat的通道顺序是BGR，因此需要交换位置
        *r++ = (float)src_data[2] / 255.0f;
        *g++ = (float)src_data[1] / 255.0f;
        *b++ = (float)src_data[0] / 255.0f;
        src_data += img.channels();
    }
}


void preprocess_by_cpu(cv::Mat& img, float* blob, int dst_h, int dst_w){
    cv::Mat scaled_img;
    preprocess_resize_img_by_cpu(img, scaled_img, dst_h, dst_w);
    preprocess_hwc2chw_normalization_by_cpu(scaled_img, blob);
}