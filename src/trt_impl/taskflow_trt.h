#ifndef __YOLO_TASKFLOW_TRT_H__
#define __YOLO_TASKFLOW_TRT_H__

#include "task_trt.h"
#include "workspace_trt.h"
#include "model_trt.h"
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <map>
#include "yolo_trt_utils.h"

namespace YOLO{
namespace TASK{
namespace TRT{

struct PreprocessingCache{
    YOLO::UTILS::TRT::CudaMemory<float> d2s_matrix;
    YOLO::UTILS::TRT::CudaMemory<float> s2d_matrix;
    YOLO::UTILS::TRT::CudaMemory<float> blob;          // img进过预处理后得到的缓存。这个缓存是直接在GPU上申请的
};

struct TaskFlowTRTContext : public TaskFlowContext{
public:
    cudaStream_t stream;
    cv::Mat img;
    YOLO::UTILS::TRT::CudaMemory<unsigned char> img_data;
    std::map<int, PreprocessingCache> preprocessing_cache;  // 不同尺寸的预处理缓存。key是图片尺寸（H*W的值）
    float* input;
	int key;
    cv::Mat mask;
    YOLO::UTILS::TRT::CudaMemory<float> roi;
public:
    TaskFlowTRTContext();
    ~TaskFlowTRTContext();
};

class TaskFlow : public TaskFlowAny{
public:
    TaskFlow();
    ~TaskFlow() override;

public:
    bool binding(const std::vector<std::tuple<std::unique_ptr<YOLO::MODEL::TRT::Model>, std::string, int>>& models);

    std::vector<TaskResult> execute(cv::Mat& img, YOLO::WORKSPACE::TRT::WorkSpace*);

private:
    void release_all();
private:
    static void get_preprocess_input(TaskFlowTRTContext& ctx, int H, int W, int C, cv::Mat& cvmask);

private:
    std::vector<std::shared_ptr<TaskTRT>> _tasks;
    std::unique_ptr<YOLO::TASK::TaskUnit> _task_header = nullptr;
};



}}}

#endif // __YOLO_TASKFLOW_TRT_H__