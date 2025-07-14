#include "task_segment_trt.h"
#include "yolo_trt_utils.h"
#include "model_trt.h"

#include <algorithm>

void decode_boxes(float* predict, int shape, int box_count, int class_count, float confidence_threshold, float nms_threshold, float* d2s_matrix, float* temparray, int ret_limit, int mask_dim, YOLO::TASK::TRT::TaskFlowTRTContext* ctx);
void decode_mask(
        float* mask_predict, int mask_width, int mask_height, int mask_dim, float mask_threshold,
        float left, float top, float* mask_weights,
        unsigned char* mask_out, int out_width, int out_height,
        YOLO::TASK::TRT::TaskFlowTRTContext* ctx);

using namespace YOLO::TASK::TRT;

Segment::Segment(YOLO::MODEL::TRT::Model* model) : TaskTRT(model->alias(), model->task_type()){
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
    if(dims_o0.d[1] < 1000){
        _shape = 0;
        _class_count = dims_o0.d[1] - YOLO::TASK::RECT_LEN - _mask_dim;
        _box_count = dims_o0.d[2];
    } else {
        _shape = 1;
        _class_count = dims_o0.d[2] - YOLO::TASK::RECT_LEN - _mask_dim;
        _box_count = dims_o0.d[1];

    }

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
	YOLO::UTILS::TRT::CudaMemory<float> output0, output1;
    output0.malloc(_output0_size);
    output1.malloc(_output1_size);
    work_space->setTensorAddress("images", ctx->input);
	work_space->setTensorAddress("output0", output0.gpu());
    work_space->setTensorAddress("output1", output1.gpu());
	cudaStreamSynchronize(ctx->stream);
	TRACE("binding address", ctx);
    work_space->enqueueV3(ctx->stream);
	cudaStreamSynchronize(ctx->stream);
	TRACE("inference", ctx);
    YOLO::TASK::SegmentResult result;
    Segment::postprocess_segment_normal(output0.gpu(), _box_count, _class_count, _shape,
        output1.gpu(), _mask_width, _mask_height, _mask_dim,
        _confidence_threshold, _nms_threshold, _mask_threshold, YOLO::TASK::TRT::BBOX_LIMIT,
        ctx->preprocessing_cache[ctx->key].d2s_matrix.cpu(), ctx->preprocessing_cache[ctx->key].s2d_matrix.cpu(),
        _input_width, _input_height, result, ctx);
    return TaskResult(std::move(result));
}

struct bbox_buffer{
    YOLO::UTILS::TRT::CudaMemory<unsigned char> mask;
    int left;
    int top;
    int width;
    int height;
    bbox_buffer(int l, int t, int w, int h){
        left = l;
        top = t;
        width = w;
        height = h;
        mask.malloc(w*h);
    }
    bbox_buffer(const bbox_buffer& s){
        left = s.left;
        top = s.top;
        width = s.width;
        height = s.height;
        mask.malloc(width*height);
    }
    bbox_buffer(bbox_buffer&& s){
        left = s.left;
        top = s.top;
        width = s.width;
        height = s.height;
        mask.malloc(width*height);
    }
};

void Segment::postprocess_segment_normal(
        // 检测头相关
        float* predict, int box_count, int class_count, int shape,
        // mask头相关
        float* mask_predict, int mask_w, int mask_h, int mask_dim,
        // 配置相关
        float confidence_threshold, float nms_threshold, float mask_threshold, int ret_limit,
        float* d2s_matrix, float* s2d_matrix,
        int input_w, int input_h,
        YOLO::TASK::SegmentResult& output,
        YOLO::TASK::TRT::TaskFlowTRTContext* ctx)
{
    YOLO::UTILS::TRT::CudaMemory<float> temparray;
    temparray.malloc((YOLO::TASK::TRT::NUM_OF_BOX_ELEMENTS + mask_dim) * ret_limit + 1);
    // 解析检测头，将合适的box写入temparray，并进行nms操作。此时可以从temparray中提取所有有效的bbox
    decode_boxes(predict, shape, box_count, class_count, confidence_threshold, nms_threshold, d2s_matrix, temparray.gpu(), ret_limit, mask_dim, ctx);
    TRACE("postprocess --- decode_boxes", ctx);
    // 将temparray从device拷贝到host，准备尽心下一步mask头解析
    temparray.memcpy_to_cpu_sync();
    TRACE("postprocess --- download predict", ctx);
    
    int count = std::min((int)temparray.cpu()[0], ret_limit);
    output.bboxes.reserve(count);
    std::list<bbox_buffer> temp_buffer;
    //temp_buffer.reserve(count);

    for(int i = 0; i < count; ++i){
        float* cur_item = temparray.cpu() + 1 + i * (YOLO::TASK::TRT::NUM_OF_BOX_ELEMENTS + mask_dim);
        if((int)(cur_item[6]) == 0){
            // 这个box被抑制了，不需要输出
            continue;
        }
        YOLO::TASK::SegmentResultItem box;
        box.left = cur_item[0];
        box.top = cur_item[1];
        box.right = cur_item[2];
        box.bottom = cur_item[3];
        box.confidence = cur_item[4];
        box.label = (int)(cur_item[5]);

        float left_t = box.left * s2d_matrix[0] + s2d_matrix[2];
        float top_t = box.top * s2d_matrix[4] + s2d_matrix[5];
        float right_t = box.right * s2d_matrix[0] + s2d_matrix[2];
        float bottom_t = box.bottom * s2d_matrix[4] + s2d_matrix[5];
        if(left_t < 0) left_t = 0;
        if(top_t < 0) top_t = 0;
		if (right_t > input_w) right_t = input_w;
		if (bottom_t > input_h) bottom_t = input_h;
        float box_w = right_t - left_t;
        float box_h = bottom_t - top_t;
        float scale_x = mask_w / (float)input_w;
        float scale_y = mask_h / (float)input_h;
        int mask_out_w = (int)(box_w * scale_x + 0.5f);
        int mask_out_h = (int)(box_h * scale_y + 0.5f);
    
        if(mask_out_w > 0 && mask_out_h > 0){
            // bbox_buffer t(left_t * scale_x, top_t * scale_y, mask_out_w, mask_out_h);
            // temp_buffer.emplace_back(t);
            //temp_buffer.emplace_back(bbox_buffer{(int)(left_t * scale_x), int(top_t * scale_y), mask_out_w, mask_out_h});
            temp_buffer.emplace_back((int)(left_t * scale_x), int(top_t * scale_y), mask_out_w, mask_out_h);
            bbox_buffer& box_buf = temp_buffer.back();
            decode_mask(mask_predict, mask_w, mask_h, mask_dim, mask_threshold,
                    left_t * scale_x, top_t * scale_y, cur_item + 7, box_buf.mask.gpu(), mask_out_w, mask_out_h, ctx);
            box_buf.mask.memcpy_to_cpu_async(ctx->stream);
			output.bboxes.emplace_back(box);
        }
        
    }

    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    TRACE("postprocess --- decode mask", ctx);

	int i = 0;
    for(auto& box:temp_buffer){
        //bbox_buffer& box = temp_buffer[i];
        int class_id = output.bboxes[i].label;
        cv::Mat mask;
        for(int j = 0; j < output.masks.size(); ++j){
            if(output.masks[j].first == class_id){
                mask = output.masks[j].second;
                //cv::Mat mask = cv::Mat(box.height, box.width, CV_8UC1, box.mask.cpu());
                //output.masks[j].second.setTo(mask, mask);
                //done = true;
                break;
            }
        }
        if(mask.empty()){
            mask = cv::Mat(mask_w, mask_h, CV_8UC1, cv::Scalar(0));
            output.masks.emplace_back(std::make_pair(class_id, mask));
        }
        cv::Mat box_mask = cv::Mat(box.height, box.width, CV_8UC1, box.mask.cpu());
        mask(cv::Rect(box.left, box.top, box.width, box.height)).setTo(cv::Scalar(255), box_mask);
		++i;
    }
    TRACE("postprocess --- clear tempbuffer", ctx);
}