#ifndef __YOLO_DETECTION_TRT_H__
#define __YOLO_DETECTION_TRT_H__

// @brief 基于CUDA/TensorRT实现的YOLOv8-Detection模型

#include "yolo_base.h"
#include "yolo_struct_def.h"
#include <chrono>

namespace YOLO{
    

class YoloDetectionTRT: public YoloBase{

public:
    YoloDetectionTRT();
    YoloDetectionTRT(const YoloDetectionTRT&) = delete;
    YoloDetectionTRT operator=(const YoloDetectionTRT&) = delete;
    virtual ~YoloDetectionTRT() override;

    bool load(const std::string& engine_file);

public:
	std::vector<DetectionResult> detect(cv::Mat& img, WorkSpace* work_space = nullptr);


private:
    int _class_count;
    int _box_count;

    int _input_size;
    int _output0_size;
};


}

#endif //__YOLO_DETECTION_TRT_H__