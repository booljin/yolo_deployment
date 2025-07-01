#include "task_classify_trt.h"
#include "yolo_trt_utils.h"
#include "model_trt.h"

#include <algorithm>

void postprocess_classify_by_cuda(
	// 检测头相关
	float* predict, int class_count,
	YOLO::TASK::ClassifyResult& output,
	YOLO::TASK::TRT::TaskFlowTRTContext* ctx);

using namespace YOLO::TASK::TRT;
Classify::Classify(YOLO::MODEL::TRT::Model* model) : TaskTRT(model->alias(), model->task_type()){
    auto dims_i = model->engine()->getTensorShape("images");
    _input_batch = dims_i.d[0];
    _input_channel = dims_i.d[1];
    _input_height = dims_i.d[2];
    _input_width = dims_i.d[3];

    auto dims_o0 = model->engine()->getTensorShape("output0");
    _class_count = dims_o0.d[1];

    _input_size = _input_batch * _input_channel * _input_height * _input_width;
    _output0_size = _input_batch * _class_count;

    _confidence_threshold = model->confidence_threshold();
}

Classify::~Classify() {
    
}

YOLO::TASK::TaskResult Classify::inference(YOLO::TASK::TaskFlowContext* ctx, void* wsa){
    TaskFlowTRTContext* trt_ctx = static_cast<TaskFlowTRTContext*>(ctx);
	nvinfer1::IExecutionContext* trt_ws = reinterpret_cast<nvinfer1::IExecutionContext*>(wsa);
    return inference(trt_ctx, trt_ws);
}

YOLO::TASK::TaskResult Classify::inference(TaskFlowTRTContext* ctx, nvinfer1::IExecutionContext* work_space){
	YOLO::UTILS::TRT::CudaMemory<float> output;
    output.malloc(_output0_size);
    work_space->setTensorAddress("images", ctx->input);
	work_space->setTensorAddress("output0", output.gpu());
    work_space->enqueueV3(ctx->stream);
    output.memcpy_to_cpu_sync();
    YOLO::TASK::ClassifyResult result;
	postprocess_classify_by_cuda(output.cpu(), _class_count, result, ctx);

    return TaskResult(std::move(result));
}