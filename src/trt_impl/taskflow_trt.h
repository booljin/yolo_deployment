#ifndef __YOLO_TASKFLOW_TRT_H__
#define __YOLO_TASKFLOW_TRT_H__

#include "task_trt.h"
#include <NvInfer.h>
#include "yolo_trt_utils.h"

namespace YOLO{
namespace TASK{
namespace TRT{

struct TaskFlowTRTContext : public TaskFlowContext{
public:
    cudaStream_t stream;
    YOLO::TRT::CudaMemory<float> d2s_matrix;
    YOLO::TRT::CudaMemory<float> s2d_matrix;
    YOLO::TRT::CudaMemory<float> input_buffer;  // img进过预处理后得到的缓存。这个缓存是直接在GPU上申请的
public:
    TaskFlowTRTContext();
    ~TaskFlowTRTContext();
};

class TaskFlow : public TaskFlowAny{
public:
    TaskFlow();
    ~TaskFlow() override;

public:
    bool binding(const std::vector<std::shared_ptr<ModelAny>>& models) override;
    std::vector<TaskResult> execute(cv::Mat& img, WorkSpaceAny*) override;

private:
    void release_all();

private:
    std::vector<std::shared_ptr<TaskTRT>> _tasks;
};



}
}
}

#endif // __YOLO_TASKFLOW_TRT_H__