#ifndef __YOLO_CLASSIFICATION_TRT_H__
#define __YOLO_CLASSIFICATION_TRT_H__

// @brief 基于CUDA/TensorRT实现的YOLOv5-Classification模型

#include "yolo_base.h"
#include "yolo_struct_def.h"
#include <chrono>

namespace YOLO{

class YoloClassificationTRT: public YoloBase{

public:
    YoloClassificationTRT();
    YoloClassificationTRT(const YoloClassificationTRT&) = delete;
    YoloClassificationTRT operator=(const YoloClassificationTRT&) = delete;
    virtual ~YoloClassificationTRT() override;

    bool load(const std::string& engine_file);

public:
	std::vector<ClassificationResult> detect(cv::Mat& img, WorkSpace* work_space = nullptr);


private:
    int _class_count;

    int _input_size;
    int _output0_size;
};


}

#endif //__YOLO_CLASSIFICATION_TRT_H__