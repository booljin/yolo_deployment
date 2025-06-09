#include <opencv2/opencv.hpp>
#include <yolo_utils.h>

void preprocess_resize_img_by_cpu(cv::Mat& input, cv::Mat& output, int dst_h, int dst_w){
    output = cv::Mat(dst_h, dst_w, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat scale_img;
    int src_w = input.cols;
    int src_h = input.rows;
    float scale = std::min((float)dst_h / (float)src_h, (float)dst_w / (float)src_w);
    int scale_w = (int)(scale * src_w);
    int scale_h = (int)(scale * src_h);
    scale_w = scale_w > dst_w ? dst_w : scale_w;
    scale_h = scale_h > dst_h ? dst_h : scale_h;
    cv::resize(input, scale_img, cv::Size(scale_w, scale_h), 0, 0, cv::INTER_LINEAR);
    scale_img.copyTo(output(cv::Rect((dst_w - scale_w) / 2, (dst_h - scale_h) / 2, scale_w, scale_h)));
}

void preprocess_hwc2chw_normalization_by_cpu(cv::Mat& img, float* blob){
    int img_h = img.rows;
    int img_w = img.cols;
    uchar* src_data = img.data;
    float* r = blob;
    float* g = blob + img_h * img_w;
    float* b = blob + img_h * img_w * 2;
    for(int i = 0; i < img_h * img_w; i++){
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