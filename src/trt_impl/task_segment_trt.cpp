#include "task_segment_trt.h"
#include "yolo_trt_utils.h"
#include "model_trt.h"

#include <algorithm>

void postprocess_segment_by_cuda(
        // 检测头相关
        float* predict, int box_count, int class_count,
        // mask头相关
        float* mask_predict, int mask_width, int mask_height, int mask_dim,
        // 配置相关
        float confidence_threshold, float nms_threshold, float mask_threshold, int ret_limit,
        float* d2s_matrix, float* s2d_matrix,
        int input_w, int input_h,
        std::vector<YOLO::TASK::SegmentResult>& output,
        YOLO::TASK::TRT::TaskFlowTRTContext* ctx);

using namespace YOLO::TASK::TRT;

Segment::Segment(std::shared_ptr<YOLO::TRT::Model> model) : TaskTRT(){
    auto dims_i = model->engine()->getTensorShape("images");
    _input_batch = dims_i.d[0];
    _input_channel = dims_i.d[1];
    _input_height = dims_i.d[2];
    _input_width = dims_i.d[3];

    auto dims_o1 = model->engine()->getTensorShape("output1");
    _mask_dim = dims_o1.d[1];
    _mask_height = dims_o1.d[2];
    _mask_width = dims_o1.d[3];

    auto dims_o0 = model->engine()->getTensorShape("output0");
    _class_count = dims_o0.d[1] - YOLO::TASK::RECT_LEN - _mask_dim;
    _box_count = dims_o0.d[2];

    _input_size = _input_batch * _input_channel * _input_height * _input_width;
    _output0_size = _input_batch * dims_o0.d[1] * _box_count;
    _output1_size = _input_batch * _mask_dim * _mask_height * _mask_width;

    _confidence_threshold = model->confidence_threshold();
    _nms_threshold = model->nms_threshold();
    _mask_threshold = model->mask_threshold();
}

Segment::~Segment() {
    
}

YOLO::TASK::TaskResult Segment::inference(YOLO::TASK::TaskFlowContext* ctx, void* wsa){
    TaskFlowTRTContext* trt_ctx = static_cast<TaskFlowTRTContext*>(ctx);
	nvinfer1::IExecutionContext* trt_ws = reinterpret_cast<nvinfer1::IExecutionContext*>(wsa);
    return inference(trt_ctx, trt_ws);
}

YOLO::TASK::TaskResult Segment::inference(TaskFlowTRTContext* ctx, nvinfer1::IExecutionContext* work_space){
	YOLO::TRT::CudaMemory<float> output0, output1;
    output0.malloc(_output0_size);
    output1.malloc(_output1_size);
    work_space->setTensorAddress("images", ctx->input_buffer.gpu());
	work_space->setTensorAddress("output0", output0.gpu());
    work_space->setTensorAddress("output1", output1.gpu());
	TRACE("binding address", ctx);
    work_space->enqueueV3(ctx->stream);
	TRACE("inference", ctx);
    std::vector<YOLO::TASK::SegmentResult> result;
    postprocess_segment_by_cuda(output0.gpu(), _box_count, _class_count,
        output1.gpu(), _mask_width, _mask_height, _mask_dim,
        _confidence_threshold, _nms_threshold, _mask_threshold, YOLO::TASK::TRT::BBOX_LIMIT,
        ctx->d2s_matrix.cpu(), ctx->s2d_matrix.cpu(),
        _input_width, _input_height, result, ctx);
    return TaskResult(std::move(result));
}