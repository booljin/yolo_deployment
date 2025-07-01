#include "taskflow_trt.h"

#include <opencv2/opencv.hpp>
#include <tuple>

#include "yolo_defines.h"
#include "yolo_trt_utils.h"
#include "workspace_trt.h"
#include "model_trt.h"
#include "task_classify_trt.h"
#include "task_detect_trt.h"
#include "task_segment_trt.h"

#include "spdlog/spdlog.h"

void preprocess_by_cuda(cv::Mat& img, float* input_buffer, int dst_h, int dst_w, float* d2s_matrix, unsigned char* mask, int mask_h, int mask_w, YOLO::TASK::TRT::TaskFlowTRTContext* ctx);

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

bool TaskFlow::binding(const std::vector<std::tuple<std::unique_ptr<YOLO::MODEL::TRT::Model>, std::string, int>>& models){
    if(_type == MODEL_TYPE_TRT){
        // taskflow已经绑定
        return false;
    }
    if(models.size() == 0){
        // 没有模型
        return false;
    }
    if(_task_header)
        return false;
    // 临时header。如果创建过程没过程，会保存到_task_header中，否则离开作用域会自动析构
    std::unique_ptr<YOLO::TASK::TaskUnit> header;
    for(auto& model_ : models){
        auto* model = std::get<0>(model_).get();
        if(model->status() == YOLO::MODEL::Status::STATUS_NotReady){
            // 模型状态不正确
            return false;
        }
        if(model->type() != MODEL_TYPE_TRT){
            // 模型类型错误
            return false;
        }
        std::unique_ptr<YOLO::TASK::TaskUnit> cur_tu = std::make_unique<YOLO::TASK::TaskUnit>();
        
        YOLO::MODEL::TRT::Model* trt_model = dynamic_cast<YOLO::MODEL::TRT::Model*>(model);
        switch(trt_model->task_type()){
		case YOLO::TaskType::TASK_CLASSIFY:
            cur_tu->task = std::make_unique<Classify>(trt_model);
            cur_tu->successors.resize(dynamic_cast<Classify*>(cur_tu->task.get())->_class_count);
            break;
        case YOLO::TaskType::TASK_DETECT:
            cur_tu->task = std::make_unique<Detect>(trt_model);
            cur_tu->successors.resize(1);
            break;
        case YOLO::TaskType::TASK_SEGMENT:
            cur_tu->task = std::make_unique<Segment>(trt_model);
            cur_tu->successors.resize(1);
            break;
        default:
            return false;
        }
        std::string parent = std::get<1>(model_);
        unsigned int branch_id = std::get<2>(model_);
        if(parent == ""){
            // 没有前驱节点，这个应该是任务流的首节点
            if(header){
                throw(YOLO::YoloException("taskflow logic error"));
                return false;
            }
            header = std::move(cur_tu);
        } else {
            if(!header){
                throw(YOLO::YoloException("taskflow logic error"));
                return false;
            }
            auto* p = get_TU_by_alias(header.get(), parent);
            if(p == nullptr){
                throw(YOLO::YoloException("taskflow parent not exist"));
                return false;
            }
            if(branch_id >= p->successors.size() ||  p->successors[branch_id] != nullptr){
                throw(YOLO::YoloException("taskflow logic error"));
                return false;
            }
            p->successors[branch_id] = std::move(cur_tu);
        }
        
    }
    _task_header = std::move(header);

    return true;
}

void TaskFlow::get_preprocess_input(TaskFlowTRTContext& ctx, int H, int W, int C, cv::Mat& cvmask){
    int key = H * W;
    if(ctx.preprocessing_cache.find(key) != ctx.preprocessing_cache.end()){
        ctx.input = ctx.preprocessing_cache[key].blob.gpu();
		ctx.key = key;
        // 如果预处理的blob有效，就可以跳过预处理。 在模型串连时，可能会要清除原blob，根据mask重新生成
        if(ctx.input)
            return;
    }

    // 计算仿射矩阵
    /*
    令缩放比例为s,原坐标表示为:{x, y},缩放后坐标变为{x', y'}，即 {x', y'} = {x*s, y*s}
    在缩放的同时，需要进行位移操作，假设x需要修正a，y需要修正b， 即 {x', y'} = {x*s + a, y*s + b} 
    此操作可以用仿射矩阵来表示：
    |x'|   |s 0 a|   |x|
    |y'| = |0 s b| * |y|
    |1 |   |0 0 1|   |1|  （齐次项）
    */
    int src_w = ctx.img.cols;
    int src_h = ctx.img.rows;
    float scale = std::min((float)H / (float)src_h, (float)W / (float)src_w);
    // 这个是从原图坐标到目标图的缩放矩阵A
    float s2d_matrix_t[6] = {scale, 0, (W - (scale * src_w)) / 2, 0, scale, (H - (scale * src_h)) / 2};
    cv::Mat s2d_mat(2, 3, CV_32F, s2d_matrix_t);
    cv::Mat d2s_mat(2, 3, CV_32F);
    // 获取逆变换矩阵，即目标图坐标到原图的变换矩阵A'
    cv::invertAffineTransform(s2d_mat, d2s_mat);
    ctx.preprocessing_cache[key].d2s_matrix.malloc(6);
    ctx.preprocessing_cache[key].s2d_matrix.malloc(6);
    ctx.preprocessing_cache[key].blob.malloc(H * W * C);
    memcpy(ctx.preprocessing_cache[key].d2s_matrix.cpu(), d2s_mat.ptr<float>(0), sizeof(float) * ctx.preprocessing_cache[key].d2s_matrix.len());
    memcpy(ctx.preprocessing_cache[key].s2d_matrix.cpu(), s2d_matrix_t, sizeof(float) * ctx.preprocessing_cache[key].s2d_matrix.len());
	TRACE("create affine matrix", &ctx);
    YOLO::UTILS::TRT::CudaMemory<unsigned char> mask;
    int mask_w = 0, mask_h = 0;
    if(!cvmask.empty()){
        mask.malloc(cvmask.total() * cvmask.elemSize());
        memcpy(mask.cpu(), cvmask.data, mask.len());
        mask.memcpy_to_gpu_sync();
        mask_w = cvmask.cols;
        mask_h = cvmask.rows;
    }

    preprocess_by_cuda(ctx.img, ctx.preprocessing_cache[key].blob.gpu(), H, W, ctx.preprocessing_cache[key].d2s_matrix.cpu(), mask.gpu(), mask_h, mask_w, &ctx);

	ctx.input = ctx.preprocessing_cache[key].blob.gpu();
	ctx.key = key;
    ctx.mask.release();
    TRACE("preprocess_by_cuda", &ctx);
}

std::vector<YOLO::TASK::TaskResult> TaskFlow::execute(cv::Mat& img, YOLO::WORKSPACE::TRT::WorkSpace* wsa){
    std::vector<YOLO::TASK::TaskResult> results;
	YOLO::WORKSPACE::TRT::WorkSpace* ws = dynamic_cast<YOLO::WORKSPACE::TRT::WorkSpace*>(wsa);
    if(ws == nullptr){
        // 工作区类型不对，不应该这样
        return results;
    }
    TaskFlowTRTContext ctx;

    TRACE("begin", &ctx);
    ctx.img = img;

    YOLO::TASK::TaskUnit* cur_task = _task_header.get();
    YOLO::TASK::TaskUnit* parent_task = nullptr;
    while(cur_task){
        get_preprocess_input(ctx, cur_task->task->_input_height, cur_task->task->_input_width, cur_task->task->_input_channel, ctx.mask);
        auto result = cur_task->task->inference(&ctx, ws->at(cur_task->task->alias()));
        if(cur_task->task->task_type() == YOLO::TaskType::TASK_CLASSIFY){
            // 分类任务需要根据结果选择后续任务
            auto class_ret = std::get_if<YOLO::TASK::ClassifyResult>(&result);
            if(!class_ret) {
                throw YoloException("taskflow.execute() - ClassifyResult error");
            }
            int best_class = (*class_ret)[0].id;
            if(best_class >= cur_task->successors.size()){
                throw YoloException("taskflow.execute() - ClassifyResult error");
            }
            parent_task = cur_task;
            cur_task = cur_task->successors[best_class].get();
        } else {
            parent_task = cur_task;
            cur_task = cur_task->successors[0].get();
            //TODO: 不同类型模型的后处理不一样
            if(parent_task->task->task_type() == YOLO::TaskType::TASK_SEGMENT){
                if(cur_task){
                    // 分割后接任何模型，后续模型需要根据mask重新生成blob
					auto seg_ret = std::get_if<YOLO::TASK::SegmentResult>(&result);
					if (seg_ret->masks.size() > 0) {
						int new_key = cur_task->task->_input_height * cur_task->task->_input_width;
						if (ctx.preprocessing_cache.find(new_key) != ctx.preprocessing_cache.end()) {
							ctx.preprocessing_cache[new_key].blob.release();
						}
						ctx.mask = seg_ret->masks[0].second;
					}
                }
            }
        }
        results.emplace_back(std::move(result));
    }

    return results;
}

void TaskFlow::release_all(){
    _tasks.clear();
    _task_header.reset();
}