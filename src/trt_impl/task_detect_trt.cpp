#include "task_detect_trt.h"
#include "yolo_trt_utils.h"
#include "model_trt.h"

#include <algorithm>

void postprocess_detect_by_cuda(
        // 检测头相关
        float* predict, int box_count, int class_count,
        // 配置相关
        float confidence_threshold, float nms_threshold, int ret_limit,
        float* d2s_matrix,
        std::vector<YOLO::TASK::DetectResult>& output,
        YOLO::TASK::TRT::TaskFlowTRTContext* ctx);

using namespace YOLO::TASK::TRT;

Detect::Detect(std::shared_ptr<YOLO::TRT::Model> model) : TaskTRT(){
    auto dims_i = model->engine()->getTensorShape("images");
    _input_batch = dims_i.d[0];
    _input_channel = dims_i.d[1];
    _input_height = dims_i.d[2];
    _input_width = dims_i.d[3];

    auto dims_o0 = model->engine()->getTensorShape("output0");
    _class_count = dims_o0.d[1] - YOLO::TASK::RECT_LEN;
    _box_count = dims_o0.d[2];

    _input_size = _input_batch * _input_channel * _input_height * _input_width;
    _output0_size = _input_batch * dims_o0.d[1] * _box_count;

    _confidence_threshold = model->confidence_threshold();
    _nms_threshold = model->nms_threshold();
}

Detect::~Detect() {
    
}

YOLO::TASK::TaskResult Detect::inference(YOLO::TASK::TaskFlowContext* ctx, void* wsa){
    TaskFlowTRTContext* trt_ctx = static_cast<TaskFlowTRTContext*>(ctx);
	nvinfer1::IExecutionContext* trt_ws = reinterpret_cast<nvinfer1::IExecutionContext*>(wsa);
    return inference(trt_ctx, trt_ws);
}

YOLO::TASK::TaskResult Detect::inference(TaskFlowTRTContext* ctx, nvinfer1::IExecutionContext* work_space){
	YOLO::TRT::CudaMemory<float> output;
    output.malloc(_output0_size);
    work_space->setTensorAddress("images", ctx->input_buffer.gpu());
	work_space->setTensorAddress("output0", output.gpu());
    work_space->enqueueV3(ctx->stream);
    std::vector<YOLO::TASK::DetectResult> result;
    postprocess_detect_by_cuda(output.gpu(), _box_count, _class_count,
        _confidence_threshold, _nms_threshold, YOLO::TASK::TRT::BBOX_LIMIT,
        ctx->d2s_matrix.cpu(), result, ctx);
    return TaskResult(std::move(result));
}