#ifndef __YOLO_BASE_H__
#define __YOLO_BASE_H__

#include <NvInfer.h>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "yolo_utils.h"

namespace YOLO{
class TrtEngine;

class YoloBase{
public:
    struct ResizeInfo{  // 原图会等比例缩放到指定大小，real_x 和 real_y 为原图在缩放后的图像中的起始
        int real_w;
        int real_h;
        int real_x;
        int real_y;
        float scale;
        int ori_w;
        int ori_h;
    };

public:
    YoloBase();
    virtual ~YoloBase();
    YoloBase(const YoloBase&) = delete;
    YoloBase operator=(const YoloBase&) = delete;

    virtual bool load(const std::string& engine_file);
public:
    inline TrtEngine* engine() {return _engine;}

public:
	WorkSpace getNewContext();

protected:
    // 考虑到实际缩放以及后续处理都会用到cuda加速，都在显存中，所以计算缩放信息相关内容就单独提出来先做
    void pre_resize_img(const cv::Mat& input, ResizeInfo& resize_info, std::vector<std::pair<std::string, std::chrono::system_clock::time_point>>& tracer);

private:
    TrtEngine* _engine = nullptr;
protected:
	float _input_w;
	float _input_h;
    int _input_c;
    int _batch_size;
	WorkSpace _default_workspace;
};


}
#endif // __YOLO_BASE_H__