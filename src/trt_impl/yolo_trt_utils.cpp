#include "yolo_trt_utils.h"
#include <opencv2/opencv.hpp>

namespace YOLO {
namespace UTILS{
namespace TRT{
void dump_preprocess_img(CudaMemory<float>& img, const std::string& path){
    img.memcpy_to_cpu_sync();
    cv::Mat dst(640, 640, CV_8UC3);
    for(int h = 0; h < 640; h++){
        for(int w = 0; w < 640; w++){
            dst.at<cv::Vec3b>(h, w) = cv::Vec3b(
                img.cpu()[640 * 640 * 2 + h * 640 + w] * 255,
                img.cpu()[640 * 640 + h * 640 + w] * 255,
                img.cpu()[h * 640 + w] * 255
            );

        }
    }
    cv::imwrite(path, dst);
}

}}}