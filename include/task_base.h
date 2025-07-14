#ifndef __YOLO_TASK_ANY_H__
#define __YOLO_TASK_ANY_H__

#include "model_base.h"
#include <opencv2/opencv.hpp>
#include <variant>
#include <vector>
#include <chrono>
#include <memory>
#include <iostream>
#include "yolo_defines.h"
#include "workspace_base.h"
#include "model_base.h"

namespace YOLO{
namespace TASK{

struct SegmentResultItem {
    float left;
    float top;
    float right;
    float bottom;
    float confidence;
    int label;
    cv::Mat mask;
};

struct ClassifyResultItem {
    int id;          // 分类ID
    float confidence; // 分类置信度
};

struct DetectResultItem {
    float left;
    float top;
    float right;
    float bottom;
    float confidence;
    int label;
};

struct SegmentResult {
    std::vector<SegmentResultItem> bboxes;
    std::vector<std::pair<int, cv::Mat>> masks;
};

using ClassifyResult = std::vector<ClassifyResultItem>;
using DetectResult = std::vector<DetectResultItem>;
using TaskResult = std::variant<std::monostate, ClassifyResult, DetectResult, SegmentResult>;

cv::Mat draw_mask(const SegmentResult& bboxes, cv::Mat& mask_img, bool origin = false);
cv::Mat draw_box(const DetectResult& bboxes, cv::Mat& mask_img);

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

const int RECT_LEN = 4;

class TaskAny {
public:
	TaskAny(const std::string& alias, YOLO::TaskType task_type) :_alias(alias), _task_type(task_type) {}
	virtual ~TaskAny() = default;
public:
	virtual TaskResult inference(TaskFlowContext*, void*) = 0;
	inline std::string alias() { return _alias; }
	inline YOLO::TaskType task_type(){ return _task_type; }
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
protected:
    std::string _alias;
	YOLO::TaskType _task_type;

};

/* 
    * @brief   任务单元
    * @details 任务执行单元由当前任务以及它的后继任务组成。
    *          一般来说后续任务只有一个，如果是class的话，就会有多个 
*/
struct TaskUnit{
    std::unique_ptr<TaskAny> task;
    std::vector<std::unique_ptr<TaskUnit>> successors;  // 后继节点
};


class TaskFlowAny{
public:
    virtual ~TaskFlowAny() = default;
public:
    TaskUnit* get_TU_by_alias(TaskUnit* header, const std::string& alias){
        if(!header)
            return nullptr;
        if(header->task->alias() == alias)
            return header;
        for(auto& it : header->successors){
            TaskUnit* ret = get_TU_by_alias(it.get(), alias);
            if(ret)
                return ret;
        }
        return nullptr;
    }

public:
    inline YOLO::ModelType type() const { return _type; }
protected:
    YOLO::ModelType _type = MODEL_TYPE_UNKNOWN;
    // 这几个参数都是从模型的"images"中读取的。
    // 每个模型都有自己的配置，但是工作流中的模型的输入必然是一致的（在binding中保证）
    // 所以从第一个模型中备份一份数据，方便做预处理
    int _input_batch = 0;
    int _input_channel = 0;
    int _input_height = 0;
    int _input_width = 0;
    int _input_size = 0;

    std::unique_ptr<TaskUnit> _task_header = nullptr;
};


}}

std::string dump_tracer(YOLO::TASK::TaskFlowContext* ctx);
#endif // __YOLO_TASK_ANY_H__