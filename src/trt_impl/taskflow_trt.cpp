#include "taskflow_trt.h"

#include <opencv2/opencv.hpp>

#include "yolo_defines.h"
#include "yolo_trt_utils.h"
#include "workspace_trt.h"
#include "model_trt.h"
#include "task_classify_trt.h"
#include "task_detect_trt.h"
#include "task_segment_trt.h"

#include "spdlog/spdlog.h"

void preprocess_by_cuda(cv::Mat& img, float* input_buffer, int dst_h, int dst_w, float* d2s_matrix, YOLO::TASK::TRT::TaskFlowTRTContext* ctx);

using namespace YOLO::TASK::TRT;

TaskFlowTRTContext::TaskFlowTRTContext(){
    CUDA_CHECK(cudaStreamCreate(&stream));
}

TaskFlowTRTContext::~TaskFlowTRTContext(){
    CUDA_CHECK(cudaStreamDestroy(stream));
}





TaskFlow::TaskFlow(){
}

TaskFlow::~TaskFlow(){
}

bool TaskFlow::binding(const std::vector<std::shared_ptr<YOLO::ModelAny>>& models){
    if(_type == MODEL_TYPE_TRT){
        // taskflow已经绑定
        return false;
    }
    if(models.size() == 0){
        // 没有模型
        return false;
    }
    int init = 0;
    for(auto model : models){
        if(model->status() == YOLO::ModelAny::STATUS_NotReady){
            // 模型状态不正确
            release_all();
            return false;
        }
        if(model->type() != MODEL_TYPE_TRT){
            // 模型类型错误
            release_all();
            return false;
        }
        std::shared_ptr<YOLO::TRT::Model> trt_model = std::dynamic_pointer_cast<YOLO::TRT::Model>(model);
        std::shared_ptr<TaskTRT> task;
        switch(trt_model->task_type()){
        case YOLO::ModelAny::TASK_CLASSIFY:
            task = std::make_shared<Classify>(trt_model);
            break;
        case YOLO::ModelAny::TASK_DETECT:
            task = std::make_shared<Detect>(trt_model);
            break;
        case YOLO::ModelAny::TASK_SEGMENT:
            task = std::make_shared<Segment>(trt_model);
            break;
        default:
            // 模型任务识别错误
            release_all();
            return false;
        }
        if(init == 0){
            _input_batch = task->_input_batch;
            _input_channel = task->_input_channel;
            _input_height = task->_input_height;
            _input_width = task->_input_width;
            _input_size = task->_input_size;
        } else {
            if(task->_input_batch != _input_batch
                || task->_input_channel != _input_channel
                || task->_input_height != _input_height
                || task->_input_width != _input_width){
            // 任务流里每个模型的输入shape必须是相同的
                release_all();
                return false;
            }
        }
        _tasks.emplace_back(task);
    }
    return true;
}

std::vector<YOLO::TASK::TaskResult> TaskFlow::execute(cv::Mat& img, YOLO::WorkSpaceAny* wsa){
    std::vector<YOLO::TASK::TaskResult> results;
	YOLO::TRT::WorkSpace* ws = dynamic_cast<YOLO::TRT::WorkSpace*>(wsa);
    if(ws == nullptr){
        // 工作区类型不对，不应该这样
        return results;
    }
    TaskFlowTRTContext ctx;

    TRACE("begin", &ctx);

    // 计算仿射矩阵
    /*
    令缩放比例为s,原坐标表示为:{x, y},缩放后坐标变为{x', y'}，即 {x', y'} = {x*s, y*s}
    在缩放的同时，需要进行位移操作，假设x需要修正a，y需要修正b， 即 {x', y'} = {x*s + a, y*s + b} 
    此操作可以用仿射矩阵来表示：
    |x'|   |s 0 a|   |x|
    |y'| = |0 s b| * |y|
    |1 |   |0 0 1|   |1|  （齐次项）
    */
    int src_w = img.cols;
    int src_h = img.rows;
    float scale = std::min((float)_input_height / (float)src_h, (float)_input_width / (float)src_w);
    // 这个是从原图坐标到目标图的缩放矩阵A
    float s2d_matrix_t[6] = {scale, 0, (_input_width - (scale * src_w)) / 2, 0, scale, (_input_height - (scale * src_h)) / 2};
    cv::Mat s2d_mat(2, 3, CV_32F, s2d_matrix_t);
    cv::Mat d2s_mat(2, 3, CV_32F);
    // 获取逆变换矩阵，即目标图坐标到原图的变换矩阵A'
    cv::invertAffineTransform(s2d_mat, d2s_mat);
    ctx.d2s_matrix.malloc(6);
    ctx.s2d_matrix.malloc(6);
	ctx.input_buffer.malloc(_input_size);
    memcpy(ctx.d2s_matrix.cpu(), d2s_mat.ptr<float>(0), sizeof(float) * ctx.d2s_matrix.len());
    memcpy(ctx.s2d_matrix.cpu(), s2d_matrix_t, sizeof(float) * ctx.s2d_matrix.len());
	TRACE("create affine matrix", &ctx);

    // 预处理
    preprocess_by_cuda(img, ctx.input_buffer.gpu(), _input_height, _input_width, ctx.d2s_matrix.cpu(), &ctx);
	TRACE("preprocess_by_cuda", &ctx);

	int idx = 0;
	for (auto& task : _tasks) {
		results.emplace_back(task->inference(&ctx, ws->at(idx++)));
	}
	TRACE("postprocess finish", &ctx);
    // 遍历任务，进行推理和后处理
    results.emplace_back(TaskResult());
    results.emplace_back(TaskResult{std::vector<YOLO::TASK::ClassifyResult>{}});

	SPDLOG_INFO(dump_tracer(&ctx));
    return results;
}

void TaskFlow::release_all(){
    _tasks.clear();
}