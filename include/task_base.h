#ifndef __YOLO_TASK_ANY_H__
#define __YOLO_TASK_ANY_H__

#include "model_base.h"
#include <opencv2/opencv.hpp>
#include <variant>
#include <vector>
#include <chrono>
#include <iostream>
#include "yolo_defines.h"
#include "workspace_base.h"

namespace YOLO{
namespace TASK{

struct SegmentResult{
    float left;
    float top;
    float right;
    float bottom;
    float confidence;
    int label;
    cv::Mat mask;
};

struct ClassifyResult {
    int id;          // 分类ID
    float confidence; // 分类置信度
};

struct DetectResult {
    float left;
    float top;
    float right;
    float bottom;
    float confidence;
    int label;
};

cv::Mat draw_mask(const std::vector<SegmentResult>& bboxes, cv::Mat& mask_img);

struct TaskFlowContext{
    //float* d2s_matrix = nullptr;
    //float* s2d_matrix = nullptr;
    //float* input_data = nullptr;
	std::vector<std::pair<std::string, std::chrono::system_clock::time_point>> tracer;
};
#ifdef _DEBUG
#define TRACE(point, ctx)\
(ctx)->tracer.emplace_back(std::make_pair(point, std::chrono::system_clock::now()));
#else
#define TRACE(point, ctx)\
(ctx)->tracer.emplace_back(std::make_pair(point, std::chrono::system_clock::now()));
#endif

using TaskResult = std::variant<std::monostate, std::vector<ClassifyResult>, std::vector<DetectResult>, std::vector<SegmentResult>>;

const int RECT_LEN = 4;

class TaskAny{
public:
    virtual ~TaskAny() = default;
public:
    virtual TaskResult inference(TaskFlowContext*, void*) = 0;
public:
    int _input_batch = 0;
    int _input_channel = 0;
    int _input_height = 0;
    int _input_width = 0;
	int _input_size = 0;
public:
    double _confidence_threshold = 0.3;
    double _nms_threshold = 0.5;
    double _mask_threshold = 0.5;

};

class TaskFlowAny{
public:
    virtual ~TaskFlowAny() = default;
public:
    virtual bool binding(const std::vector<std::shared_ptr<ModelAny>>& model) = 0;
    virtual std::vector<TaskResult> execute(cv::Mat& img, WorkSpaceAny*) = 0;
public:
    inline ModelType type() const { return _type; }
protected:
    ModelType _type = MODEL_TYPE_UNKNOWN;
    // 这几个参数都是从模型的"images"中读取的。
    // 每个模型都有自己的配置，但是工作流中的模型的输入必然是一致的（在binding中保证）
    // 所以从第一个模型中备份一份数据，方便做预处理
    int _input_batch = 0;
    int _input_channel = 0;
    int _input_height = 0;
    int _input_width = 0;
    int _input_size = 0;
};


}
}

std::string dump_tracer(YOLO::TASK::TaskFlowContext* ctx);
#endif // __YOLO_TASK_ANY_H__