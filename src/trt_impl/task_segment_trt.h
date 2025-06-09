#ifndef __YOLO_TASK_SEGMENT_TRT_H__
#define __YOLO_TASK_SEGMENT_TRT_H__

// @brief 基于CUDA/TensorRT实现的YOLOv8-Segmentation模型

#include "task_trt.h"
#include "taskflow_trt.h"
#include "yolo_defines.h"
#include "model_trt.h"
#include "workspace_trt.h"
#include <memory>

namespace YOLO{namespace TASK{namespace TRT{

class Segment : public TaskTRT{
public:
    Segment(std::shared_ptr<YOLO::TRT::Model> model);
    ~Segment() override;
public:
    TaskResult inference(TaskFlowContext*, void* work_space);
    TaskResult inference(TaskFlowTRTContext*, nvinfer1::IExecutionContext* work_space);
private:
    // output0
    int _class_count;
    int _box_count;
    // output1
    int _mask_dim;
    int _mask_width;
    int _mask_height;

    int _output0_size;
    int _output1_size;
};

}}}
#endif //__YOLO_TASK_SEGMENT_TRT_H__