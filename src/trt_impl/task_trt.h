#ifndef __YOLO_TASK_TRT_H__
#define __YOLO_TASK_TRT_H__

#include "task_base.h"
#include <NvInfer.h>
#include "yolo_trt_utils.h"

namespace YOLO{
namespace TASK{
namespace TRT{

const int BBOX_LIMIT = 1024;
// @brief: 每个bbox的元素结构
// @details: 每个bbox包含的元素数量：left, top, right, bottom, confidence, class, keep_flag  随后再附加所有维度的权重
const int NUM_OF_BOX_ELEMENTS = 7;

struct TaskFlowTRTContext;

class TaskTRT : public TaskAny{
public:
	TaskTRT(const std::string& alias, YOLO::TaskType task_type) :TaskAny(alias, task_type) {}
	~TaskTRT() override {};

public:
    virtual TaskResult inference(TaskFlowTRTContext*, nvinfer1::IExecutionContext*) = 0;
};



}}}

#endif // __YOLO_TASK_TRT_H__