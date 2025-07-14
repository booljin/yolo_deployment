#ifndef __YOLO_TASK_DETECT_TRT_H__
#define __YOLO_TASK_DETECT_TRT_H__

// @brief 基于CUDA/TensorRT实现的YOLOv8-Detection模型

#include "task_trt.h"
#include "taskflow_trt.h"
#include "yolo_defines.h"
#include "model_trt.h"
#include "workspace_trt.h"
#include <memory>

namespace YOLO{namespace TASK{namespace TRT{

class Detect: public TaskTRT{
public:
    Detect(YOLO::MODEL::TRT::Model* model);
    ~Detect() override;
public:
    TaskResult inference(TaskFlowContext*, void* work_space);
    TaskResult inference(TaskFlowTRTContext*, nvinfer1::IExecutionContext* work_space);
private:
    int _class_count;
    int _box_count;
    int _output0_size;

    int _shape; // 0-nfb, 1-nbf
};


}}}
#endif //__YOLO_TASK_DETECT_TRT_H__