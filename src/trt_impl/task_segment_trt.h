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
    Segment(YOLO::MODEL::TRT::Model* model);
    ~Segment() override;
public:
    TaskResult inference(TaskFlowContext*, void* work_space);
    TaskResult inference(TaskFlowTRTContext*, nvinfer1::IExecutionContext* work_space);

private:
	static void postprocess_segment_normal(
		// 检测头相关
		float* predict, int box_count, int class_count,
		// mask头相关
		float* mask_predict, int mask_w, int mask_h, int mask_dim,
		// 配置相关
		float confidence_threshold, float nms_threshold, float mask_threshold, int ret_limit,
		float* d2s_matrix, float* s2d_matrix,
		int input_w, int input_h,
		YOLO::TASK::SegmentResult& output,
		YOLO::TASK::TRT::TaskFlowTRTContext* ctx);
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